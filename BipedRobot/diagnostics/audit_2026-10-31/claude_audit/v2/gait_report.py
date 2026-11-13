"""Why does a gait look off? Measures it from foot forces at every physics step: real steps against contact chatter, left-right balance, step length, turning, body sway and joint motion.

Usage: gait_report.py LABEL CHECKPOINT RESOLVED_CONFIG OUT_JSON REALISM SECONDS SEEDS(start:stop) BUDGET [COMMAND]
Same env and policy loading as survive_eval.py. No pushes. Only episodes that stay up count.
Loaded-foot shares: whole sole (the sole's flat face within 2 mm of level, ValidLocoEnv.sole_gap), and, for the rest, heel, toe or edge by which way
the sole tilts most. flat_stance_share and stance_tilt_deg_median are the older measure, tilt from the episode's start pose.
Lean (added 6 Nov 2026): torso pitch and roll from the base quaternion (pitch > 0 = nose down, leaning forward), pitch0 at the first control step,
and com_ahead_mm, the whole robot's centre of mass ahead of the midpoint of the two foot bodies along the heading, all averaged after the first second.
Off-clock contact (added 6 Nov 2026, policies with a gait clock only, else None): a foot carrying more than 5% of the robot's weight while the clock
swings it (right foot in the second half of each cycle, left in the first; the phase the policy saw for that step). For margin 0 and 0.05 of the cycle
left out at each end of the swing half: taps_per_s (separate loaded runs), offclock_steps_per_s (loaded foot-substeps / substeps per step / s, which is
what shaping.swing_contact at weight 1 would charge per second; the budget's alive bonus pays 10 per second).
Environment variables: GAIT_DUTY (servo PWM duty, validity.duty_range [x, x]), GAIT_ALPHA (action filter alpha at test time), GAIT_HOLD=1 (gyro
heading hold, as survive_eval hold_gyro), GAIT_DELAY (servo delay fixed at this many seconds for every episode),
GAIT_CURVE "stall,noload,cap" (replace validity.servo_curve, N m, rad/s, N m; "2.94,5.3,2.94" is the designer's capped-stall line),
GAIT_COM "x,y" (torso centre of mass shifted per episode by up to x m fore-aft and y m sideways, validity.com_offset_m).
Symmetry and rhythm (7 Nov 2026): sym_stance, sym_swing, sym_lift, sym_stride = |right - left| / mean(right, left) of mean stance time, swing time, swing
lift and stride (forward travel of a foot between its own touchdowns); step_interval_cv = std / mean of the time between consecutive touchdowns of either
foot; alternation = share of consecutive touchdowns made by the other foot; clock_lock (clock policies) = share of touchdowns within 0.1 of the cycle
of the clock's beat (right at phase 0, left at 0.5). Survival is episodes_up / episodes.
whole_sole_mid_share (7 Nov 2026, clock policies only): whole_sole_share counted only more than 0.12 of the cycle from a change of feet, the part of the
stance where a heel-to-toe walk keeps the foot flat (shaping.foot_flat.free_ends 0.12).
"""
import os
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import load_env_config  # noqa: E402
from budget_eval import BUDGETS, LoggedEnv, pad_budget  # noqa: E402
from survive_eval import HEADING_GAIN, HEADING_MAX_WZ, load_policy, sensed_yaw_rate  # noqa: E402

FORCE_N = 0.5
REAL_STANCE_S = 0.06   # a foot loaded for less than this, or lifted for less than this, is chatter and not a step
FLAT_RAD = np.radians(5.0)  # a loaded foot tilted less than this from its standing orientation counts as a full-foot contact
SOLE_GAP_M = 0.002  # a loaded foot whose sole's highest point is within this of its lowest has the whole sole down (ValidLocoEnv.sole_gap)


class FootShape:
    """Each foot's sole tilt from its orientation at the start of the episode, where the robot stands with flat feet, and its lowest mesh point
    above the floor, measured as evaluation_suite_v2 measures clearance."""

    def __init__(self, env):
        m = env.model
        self.env, self.geoms = env, list(env.foot_geom_ids)
        self.verts = [m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]].astype(np.float64) for g in self.geoms]
        self.normal = None

    def __call__(self):
        d = self.env.data
        R = [d.geom_xmat[g].reshape(3, 3) for g in self.geoms]
        if self.normal is None:
            self.normal = [r.T @ np.array([0.0, 0.0, 1.0]) for r in R]
        tilt = np.array([float(np.arccos(np.clip((r @ n)[2], -1.0, 1.0))) for r, n in zip(R, self.normal)])
        clear = np.array([float(np.min(d.geom_xpos[g, 2] + v @ r[2, :])) for g, v, r in zip(self.geoms, self.verts, R)])
        return tilt, clear


def segments(mask):
    """Start and length of each run of True."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def pitch_roll_of(q):
    w, x, y, z = q
    return float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))), float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def com_ahead(env):
    """Whole-robot centre of mass minus the midpoint of the two foot geoms, along the heading, in metres."""
    d = env.data
    total = env.model.body_mass.sum()
    com = (env.model.body_mass[:, None] * d.xipos).sum(axis=0) / total
    feet = d.geom_xpos[list(env.foot_geom_ids)].mean(axis=0)
    yaw = yaw_of(d.qpos[3:7])
    return float((com[0] - feet[0]) * np.cos(yaw) + (com[1] - feet[1]) * np.sin(yaw))


def episode(env, policy, command, seed, steps):
    obs, _ = env.reset(seed=seed, options={"desired_goal": command})
    loads, base, joints, targets, acts, gyro, tilts, clears = [], [], [], [], [], [], [], []
    gaps, angles, slopes = [], [], []
    lean = []
    feetpos = []
    clock = bool(getattr(env, "clock_hz", 0.0))
    weight = float(env.model.body_mass.sum()) * abs(float(env.model.opt.gravity[2]))
    offclock = []  # per substep: [margin][foot] loaded while the clock swings it
    phases = []
    shape = FootShape(env)
    shape()

    def observe(fn, slip, nonfoot):
        loads.append(fn.copy())
        feetpos.append(env.data.geom_xpos[list(env.foot_geom_ids)][:, :2].copy())
        if clock:
            p = env._phase
            phases.append(p)
            dd = getattr(env, "clock_stance", 0.5) - 0.5  # validity.clock_stance: swing windows shrink by the double support
            offclock.append(np.array([[0.5 + dd + m <= (p + dd / 2) % 1.0 < 1.0 - m, 0.5 + dd + m <= (p - 0.5 + dd / 2) % 1.0 < 1.0 - m] for m in (0.0, 0.05)])
                            & (np.asarray(fn) > 0.05 * weight)[None, :])
        tilt, clear = shape()
        tilts.append(tilt)
        clears.append(clear)
        gaps.append(env.sole_gap())
        angles.append(env.sole_angle())
        slopes.append(env.sole_slope())
    env.substep_observer = observe
    for _ in range(steps):
        a = np.asarray(policy(obs), np.float32)
        obs, _, term, trunc, _ = env.step(a)
        acts.append(a)
        base.append((*env.data.qpos[:3], yaw_of(env.data.qpos[3:7]), *env._get_projected_gravity()))
        joints.append([env.data.qvel[i] for i in env.joint_dof_adrs])
        gyro.append(env.data.qvel[3:6].copy())
        lean.append((*pitch_roll_of(env.data.qpos[3:7]), com_ahead(env)))
        if term:
            return None
    env.substep_observer = None
    start = len(loads) // 10  # skip the first second
    fn = np.array(loads)[start:]
    tilt, clear = np.array(tilts)[start:], np.array(clears)[start:]
    gap, angle, slope = np.array(gaps)[start:], np.array(angles)[start:], np.array(slopes)[start:]  # slope: (time, foot, [along, across])
    dt = env.sim_dt
    out = {}
    if clock:
        oc = np.array(offclock)[start:]  # (substeps, margin, foot)
        T_oc = len(oc) * dt
        n_sub = int(round(env.dt / dt))
        for j, m in enumerate(("0", "005")):
            out[f"taps_per_s_m{m}"] = float(sum(len(segments(oc[:, j, k])[0]) for k in range(2)) / T_oc)
            out[f"offclock_steps_per_s_m{m}"] = float(oc[:, j, :].sum() / n_sub / T_oc)
    else:
        for m in ("0", "005"):
            out[f"taps_per_s_m{m}"] = out[f"offclock_steps_per_s_m{m}"] = None
    real_steps, chatter, stance_s, swing_s, lifts = [], [], [], [], []
    fp = np.array(feetpos)[start:]
    ph = np.array(phases)[start:] if clock else None
    foot_lift, foot_stride, events = [[], []], [[], []], []
    for k in range(2):
        on = fn[:, k] > FORCE_N
        st, ln = segments(on)
        s_off, off = segments(~on)
        foot_lift[k] = [float(clear[a:a + n, k].max()) for a, n in zip(s_off, off) if n * dt >= REAL_STANCE_S]
        lifts += foot_lift[k]
        real = ln * dt >= REAL_STANCE_S
        td = st[real]
        events += [(int(i), k) for i in td]
        foot_stride[k] = list(np.diff(fp[td, k, 0])) if len(td) > 1 else []
        real_steps.append(int(real.sum()))
        chatter.append(int((~real).sum()))
        stance_s.append(ln[real] * dt)
        swing_s.append(off[off * dt >= REAL_STANCE_S] * dt)
    T = len(fn) * dt
    mean_or_none = lambda x: float(np.mean(x)) if len(x) else None
    def sym(r, l):
        if r is None or l is None or (r + l) == 0:
            return None
        return float(abs(r - l) / (0.5 * (r + l)))
    st_r, st_l = mean_or_none(stance_s[0]), mean_or_none(stance_s[1])
    sw_r, sw_l = mean_or_none(swing_s[0]), mean_or_none(swing_s[1])
    li_r, li_l = mean_or_none(foot_lift[0]), mean_or_none(foot_lift[1])
    sr_r, sr_l = mean_or_none(foot_stride[0]), mean_or_none(foot_stride[1])
    events.sort()
    if len(events) > 2:
        times = np.array([e[0] for e in events]) * dt
        iv = np.diff(times)
        alt = float(np.mean([events[i][1] != events[i - 1][1] for i in range(1, len(events))]))
        cv = float(iv.std() / iv.mean()) if iv.mean() > 0 else None
    else:
        alt = cv = None
    lock = None
    if clock and events:
        dd = getattr(env, "clock_stance", 0.5) - 0.5
        expected = np.array([(-dd / 2) % 1.0 if k == 0 else 0.5 - dd / 2 for _, k in events])
        got = ph[[i for i, _ in events]]
        err = np.abs((got - expected + 0.5) % 1.0 - 0.5)
        lock = float((err <= 0.1).mean())
    out.update(sym_stance=sym(st_r, st_l), sym_swing=sym(sw_r, sw_l), sym_lift=sym(li_r, li_l), sym_stride=sym(abs(sr_r) if sr_r is not None else None, abs(sr_l) if sr_l is not None else None),
               step_interval_cv=cv, alternation=alt, clock_lock=lock, lift_r_mm=None if li_r is None else 1000 * li_r, lift_l_mm=None if li_l is None else 1000 * li_l,
               stride_r_m=sr_r, stride_l_m=sr_l)
    b = np.array(base)[len(base) // 10:]
    dist = float(np.hypot(*(b[-1, :2] - b[0, :2])))
    path = float(np.sum(np.hypot(*np.diff(b[:, :2], axis=0).T)))
    q = np.array(joints)[len(joints) // 10:]
    a = np.array(acts)
    ln_ = np.array(lean)
    out.update(pitch0_deg=float(np.degrees(ln_[0, 0])))
    ln_ = ln_[len(ln_) // 10:]
    out.update(pitch_mean_deg=float(np.degrees(ln_[:, 0].mean())), pitch_std_deg=float(np.degrees(ln_[:, 0].std())), roll_mean_deg=float(np.degrees(ln_[:, 1].mean())),
               com_ahead_mm=1000 * float(ln_[:, 2].mean()), com_ahead_std_mm=1000 * float(ln_[:, 2].std()))
    loaded = fn > FORCE_N
    out.update(flat_stance_share=float((tilt[loaded] < FLAT_RAD).mean()) if loaded.any() else None,
               stance_tilt_deg_median=float(np.degrees(np.median(tilt[loaded]))) if loaded.any() else None,
               swing_lift_mm_median=1000 * float(np.median(lifts)) if lifts else None,
               flight_share=float((~loaded.any(axis=1)).mean()), single_support_share=float((loaded.sum(axis=1) == 1).mean()))
    if loaded.any():
        part = gap[loaded] > SOLE_GAP_M
        along, across = slope[loaded][:, 0], slope[loaded][:, 1]
        by_length = part & (np.abs(along) >= np.abs(across))
        out.update(whole_sole_share=float((~part).mean()), heel_share=float((by_length & (along < 0)).mean()), toe_share=float((by_length & (along > 0)).mean()),
                   edge_share=float((part & (np.abs(along) < np.abs(across))).mean()),
                   whole_sole_mid_share=(lambda mid: float((~part[mid[loaded.nonzero()[0]]]).mean()) if mid[loaded.nonzero()[0]].any() else None)(
                       np.array([min(q % 0.5, 0.5 - q % 0.5) >= 0.12 for q in np.array(phases)[start:]])) if clock else None,
                   sole_gap_mm_median=1000 * float(np.median(gap[loaded])), sole_angle_deg_median=float(np.degrees(np.median(angle[loaded]))))
    out.update(real_steps_per_s=sum(real_steps) / T, right_steps=real_steps[0], left_steps=real_steps[1], chatter_per_s=sum(chatter) / T,
               stance_s=float(np.mean(np.concatenate(stance_s))) if sum(map(len, stance_s)) else None, swing_s=float(np.mean(np.concatenate(swing_s))) if sum(map(len, swing_s)) else None,
               duty_right=float((fn[:, 0] > FORCE_N).mean()), duty_left=float((fn[:, 1] > FORCE_N).mean()),
               both_down=float(((fn[:, 0] > FORCE_N) & (fn[:, 1] > FORCE_N)).mean()), step_length_m=dist / max(sum(real_steps), 1), straight_m_s=dist / (len(b) * env.dt), path_m_s=path / (len(b) * env.dt),
               turn_rad_s=float((b[-1, 3] - b[0, 3]) / (len(b) * env.dt)), sideways_m=float(b[-1, 1] - b[0, 1]), height_std_mm=1000 * float(b[:, 2].std()), height_mean_m=float(b[:, 2].mean()),
               tilt_rms_deg=float(np.degrees(np.sqrt(np.mean(b[:, 4] ** 2 + b[:, 5] ** 2)))), joint_speed_p95=float(np.percentile(np.abs(q), 95)),
               action_change_rms=float(np.sqrt(np.mean(np.diff(a, axis=0) ** 2))), gyro_rms=float(np.sqrt(np.mean(np.array(gyro) ** 2))))
    return out


def main():
    label, ckpt, rc, out, realism, seconds, seeds, budget = sys.argv[1:9]
    speed = float(sys.argv[9]) if len(sys.argv) > 9 else 0.2
    a, b = (int(x) for x in seeds.split(":"))
    torch.set_num_threads(1)
    steps = int(round(float(seconds) * 50))
    cfg = copy.deepcopy(load_env_config(rc))
    cfg["realism"], cfg["push_enabled"], cfg["max_episode_steps"] = realism, False, steps
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    duty = os.environ.get("GAIT_DUTY")
    if duty:
        cfg.setdefault("validity", {})["duty_range"] = [float(duty), float(duty)]
    curve = os.environ.get("GAIT_CURVE")
    if curve:
        st, nl, cp = (float(x) for x in curve.split(","))
        cfg.setdefault("validity", {})["servo_curve"] = {"stall_nm": st, "noload_rad_s": nl, "cap_nm": cp}
    com = os.environ.get("GAIT_COM")
    if com:
        cfg.setdefault("validity", {})["com_offset_m"] = [float(x) for x in com.split(",")]
    raw_policy, _ = load_policy(ckpt)
    env = LoggedEnv(config=cfg, render_mode=None)
    alpha = os.environ.get("GAIT_ALPHA")
    if alpha:
        env.action_filter_alpha = float(alpha)
    delay = os.environ.get("GAIT_DELAY")
    if delay:
        env.enable_delay = True
        env.delay_range = (float(delay), float(delay))
        env.nominal_delay = float(delay)
    hold = os.environ.get("GAIT_HOLD") == "1"
    gyro_yaw = [0.0]

    def policy(obs):
        if not hold:
            return raw_policy(obs)
        if env._step_count == 0:
            gyro_yaw[0] = 0.0
        seen = {k: v.copy() for k, v in obs.items()}
        seen["desired_goal"][2] = float(np.clip(-HEADING_GAIN * gyro_yaw[0], -HEADING_MAX_WZ, HEADING_MAX_WZ))
        gyro_yaw[0] += sensed_yaw_rate(env, obs) * env.dt
        return raw_policy(seen)
    env.reward_mode = "track"
    env.push_enabled = False
    env.log = []
    command = np.array([speed, 0.0, 0.0, *pad_budget(BUDGETS[budget], env)], np.float32)
    rows = [r for r in (episode(env, policy, command, s, steps) for s in range(a, b)) if r]
    env.close()
    keys = rows[0].keys() if rows else []
    summary = {k: (float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None) for k in keys}
    summary["episodes_up"] = len(rows)
    summary["episodes"] = b - a
    summary["turn_abs_rad_s"] = float(np.mean([abs(r["turn_rad_s"]) for r in rows])) if rows else None
    summary["sideways_abs_m"] = float(np.mean([abs(r["sideways_m"]) for r in rows])) if rows else None
    summary["survival"] = len(rows) / (b - a)
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "budget": budget, "speed": speed, "duty": float(duty) if duty else 1.0,
                                     "alpha": float(alpha) if alpha else None, "hold": hold, "delay": float(delay) if delay else None, "com_offset_m": com, "servo_curve": curve, "summary": summary, "episodes": rows}))
    print(label, budget, f"cmd {speed}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
