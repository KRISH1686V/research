"""Looks for gait habits that the pass lines do not catch: how wide and how turned the feet land, scuffing in mid-swing, when the swing peaks, the
swing foot swinging out sideways, torso wobble and twist, bobbing, joints near their speed or angle limits, torque near the cap, jittery commands,
feet sliding in stance, a locked knee in stance, and hip twisting. Written 7 Nov 2026.

Usage: gait_anomaly.py LABEL CHECKPOINT RESOLVED_CONFIG OUT_JSON SEEDS(start:stop) COMMAND [BUDGET]
Medium realism, no pushes, 10 s episodes, the first second skipped. Only episodes that stay up count. Heuristic flags, not pass lines.
"""
import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import load_env_config  # noqa: E402
from budget_eval import BUDGETS, LoggedEnv, pad_budget  # noqa: E402
from gait_report import FORCE_N, REAL_STANCE_S, FootShape, segments, yaw_of, pitch_roll_of  # noqa: E402
from survive_eval import load_policy  # noqa: E402

NOLOAD = 5.3          # rad/s, the servo's no-load speed
CAP = 2.94            # N m, the torque cap
NAMES = ["r_hip_z", "r_hip_x", "r_hip_y", "r_knee", "r_ankle_y", "r_ankle_x", "l_hip_z", "l_hip_x", "l_hip_y", "l_knee", "l_ankle_y", "l_ankle_x"]

# heuristic lines for a flag (what a careful eye would call odd on a 1.66 kg, 30 cm biped)
FLAGS = {
    "step_width_min_mm": ("<", 40.0, "feet land closer than 4 cm side to side at least once (risk of crossing or tripping)"),
    "foot_yaw_abs_deg": (">", 8.0, "feet land turned in or out by more than 8 deg on average"),
    "scuff_share": (">", 0.10, "the swing foot is within 3 mm of the floor for over 10% of mid-swing (scuffing)"),
    "swing_peak_offcentre": (">", 0.20, "the swing foot reaches its highest point far from mid-swing"),
    "circumduction_mm": (">", 15.0, "the swing foot swings out sideways by over 15 mm"),
    "yaw_ptp_deg": (">", 6.0, "the torso twists by over 6 deg each step"),
    "roll_ptp_deg": (">", 8.0, "the torso rocks side to side by over 8 deg each step"),
    "pitch_ptp_deg": (">", 6.0, "the torso nods by over 6 deg each step"),
    "bob_ptp_mm": (">", 10.0, "the body bobs up and down by over 10 mm each step"),
    "speed_over_80_share": (">", 0.02, "a joint runs above 80% of the servo's no-load speed over 2% of the time"),
    "near_limit_share": (">", 0.02, "a joint sits within 3 deg of its range limit over 2% of the time"),
    "torque_over_80_share": (">", 0.02, "a joint uses over 80% of the torque cap over 2% of the time"),
    "action_hf_share": (">", 0.15, "over 15% of the filtered servo command's movement is faster than 5 Hz (jitter the servos cannot follow)"),
    "stance_slip_mm": (">", 5.0, "a planted foot slides by over 5 mm per stance"),
    "knee_stance_min_deg": (">", -5.0, "the knee is almost straight in stance (locked knee)"),
    "hip_yaw_ptp_deg": (">", 8.0, "the hip twists (yaw) by over 8 deg each step"),
}


def foot_yaw(R):
    """Heading of a foot from its geom frame's x axis projected on the floor."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def episode(env, policy, command, seed, steps=500):
    obs, _ = env.reset(seed=seed, options={"desired_goal": command})
    shape = FootShape(env)
    shape()
    sub = {"fn": [], "pos": [], "yaw": [], "clear": []}
    ctl = {"q": [], "qd": [], "tau": [], "act": [], "pitch": [], "roll": [], "yaw": [], "z": [], "y": [], "gap": [], "tilt": []}
    ref = {}
    feet = list(env.foot_geom_ids)

    def observe(fn, slip, nonfoot):
        sub["fn"].append(fn.copy())
        d = env.data
        sub["pos"].append(d.geom_xpos[feet][:, :2].copy())
        fy = np.array([foot_yaw(d.geom_xmat[g].reshape(3, 3)) for g in feet])
        ty = yaw_of(d.qpos[3:7])
        if not ref:
            ref.update(feet=fy.copy(), torso=ty)
        # foot turn relative to its standing pose, minus the torso's own turn since then: toe-in / toe-out
        sub["yaw"].append(list((fy - ref["feet"]) - (ty - ref["torso"])))
        sub["clear"].append(shape()[1])
    env.substep_observer = observe
    for _ in range(steps):
        a = np.asarray(policy(obs), np.float32)
        obs, _, term, _, _ = env.step(a)
        d = env.data
        ctl["q"].append(d.qpos[env.joint_qpos_adrs].copy())
        ctl["qd"].append(d.qvel[env.joint_dof_adrs].copy())
        ctl["tau"].append(d.actuator_force[env.actuator_ids].copy())
        ctl["act"].append(env._action_filter_state.copy() if env.action_filter_enabled else a)  # what the servos are sent
        p, r = pitch_roll_of(d.qpos[3:7])
        ctl["pitch"].append(p); ctl["roll"].append(r); ctl["yaw"].append(yaw_of(d.qpos[3:7]))
        ctl["gap"].append(env.stance_width_now()); ctl["tilt"].append(float(np.arccos(np.clip(d.xmat[env.torso_body_id][8], -1.0, 1.0))))
        ctl["z"].append(float(d.qpos[2])); ctl["y"].append(float(d.qpos[1]))
        if term:
            env.substep_observer = None
            return None
    env.substep_observer = None
    s0 = len(sub["fn"]) // 10
    c0 = len(ctl["q"]) // 10
    fn = np.array(sub["fn"])[s0:]; pos = np.array(sub["pos"])[s0:]; fyaw = np.array(sub["yaw"])[s0:]; clear = np.array(sub["clear"])[s0:]
    q, qd, tau, act = (np.array(ctl[k])[c0:] for k in ("q", "qd", "tau", "act"))
    pitch, roll, yaw, z = (np.array(ctl[k])[c0:] for k in ("pitch", "roll", "yaw", "z"))
    dt_s, dt_c = env.sim_dt, env.dt
    heading = float(np.mean(yaw))
    left = np.array([-np.sin(heading), np.cos(heading)])
    widths, yaws, scuffs, peaks, circ, slips = [], [], [], [], [], []
    for k in range(2):
        on = fn[:, k] > FORCE_N
        st, ln = segments(on)
        so, off = segments(~on)
        for a0, n in zip(st, ln):
            if n * dt_s < REAL_STANCE_S:
                continue
            other = 1 - k
            widths.append(abs(float((pos[a0, k] - pos[a0, other]) @ left)))
            dyaw = (fyaw[a0, k] + np.pi) % (2 * np.pi) - np.pi
            yaws.append(abs(float(dyaw)))
            slips.append(float(np.linalg.norm(pos[a0 + n - 1, k] - pos[a0, k])))
        for a0, n in zip(so, off):
            if n * dt_s < REAL_STANCE_S or a0 == 0 or a0 + n >= len(fn):
                continue
            c = clear[a0:a0 + n, k]
            mid = c[int(0.15 * n):int(0.85 * n)]
            if len(mid):
                scuffs.append(float((mid < 0.003).mean()))
            peaks.append(abs(float(np.argmax(c)) / max(n - 1, 1) - 0.5))
            path = pos[a0:a0 + n, k]
            line = path[-1] - path[0]
            if np.linalg.norm(line) > 1e-4:
                nrm = np.array([-line[1], line[0]]) / np.linalg.norm(line)
                circ.append(float(np.abs((path - path[0]) @ nrm).max()))
            else:
                circ.append(float(np.abs((path - path[0]) @ left).max()))
    period = 1.0 / (env.clock_hz or 1.0)
    win = max(1, int(round(period / dt_c)))
    def ptp_per_cycle(x):
        return float(np.mean([np.ptp(x[i:i + win]) for i in range(0, len(x) - win + 1, win)])) if len(x) >= win else float(np.ptp(x))
    lo, hi = env.jnt_min, env.jnt_max
    near = (q - lo < np.radians(3)) | (hi - q < np.radians(3))
    spec = np.abs(np.fft.rfft(act - act.mean(axis=0), axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(act), dt_c)
    hf = float(spec[freqs > 5.0].sum() / max(spec.sum(), 1e-12))
    knee = q[:, [3, 9]]
    stance_ctl = np.array([fn[min(int(i * dt_c / dt_s), len(fn) - 1)] > FORCE_N for i in range(len(q))])
    knee_st = np.concatenate([knee[stance_ctl[:, k], k] for k in range(2)]) if stance_ctl.any() else knee.ravel()
    hipz = q[:, [0, 6]]
    return dict(
        gap_p01_mm=1000 * float(np.percentile(np.array(ctl["gap"])[c0:], 1)), gap_p99_mm=1000 * float(np.percentile(np.array(ctl["gap"])[c0:], 99)), gap_min_mm=1000 * float(np.min(np.array(ctl["gap"])[c0:])), gap_max_mm=1000 * float(np.max(np.array(ctl["gap"])[c0:])),
        gap_in_band_share=float(np.mean((np.array(ctl["gap"])[c0:] >= 0.08) & (np.array(ctl["gap"])[c0:] <= 0.12))),
        tilt_p99_deg=float(np.degrees(np.percentile(np.array(ctl["tilt"])[c0:], 99))), tilt_max_deg=float(np.degrees(np.max(np.array(ctl["tilt"])[c0:]))),
        step_width_mm=1000 * float(np.mean(widths)) if widths else None, step_width_min_mm=1000 * float(np.min(widths)) if widths else None,
        foot_yaw_abs_deg=float(np.degrees(np.mean(yaws))) if yaws else None, scuff_share=float(np.mean(scuffs)) if scuffs else None,
        swing_peak_offcentre=float(np.mean(peaks)) if peaks else None, circumduction_mm=1000 * float(np.mean(circ)) if circ else None,
        yaw_ptp_deg=float(np.degrees(ptp_per_cycle(np.unwrap(yaw)))), roll_ptp_deg=float(np.degrees(ptp_per_cycle(roll))),
        pitch_ptp_deg=float(np.degrees(ptp_per_cycle(pitch))), bob_ptp_mm=1000 * ptp_per_cycle(z),
        speed_over_80_share=float((np.abs(qd) > 0.8 * NOLOAD).mean(axis=0).max()),
        speed_over_80_joint=NAMES[int((np.abs(qd) > 0.8 * NOLOAD).mean(axis=0).argmax())],
        near_limit_share=float(near.mean(axis=0).max()), near_limit_joint=NAMES[int(near.mean(axis=0).argmax())],
        torque_over_80_share=float((np.abs(tau) > 0.8 * CAP).mean(axis=0).max()), torque_over_80_joint=NAMES[int((np.abs(tau) > 0.8 * CAP).mean(axis=0).argmax())],
        action_hf_share=hf, stance_slip_mm=1000 * float(np.mean(slips)) if slips else None,
        knee_stance_min_deg=float(np.degrees(np.percentile(knee_st, 95))) if len(knee_st) else None,
        hip_yaw_ptp_deg=float(np.degrees(np.mean([ptp_per_cycle(hipz[:, k]) for k in range(2)]))),
        peak_joint_speed=float(np.abs(qd).max()), peak_joint=NAMES[int(np.abs(qd).max(axis=0).argmax())])


def main():
    label, ckpt, rc, out, seeds, cmd = sys.argv[1:7]
    budget = sys.argv[7] if len(sys.argv) > 7 else "all_mid"
    a, b = (int(x) for x in seeds.split(":"))
    torch.set_num_threads(1)
    cfg = copy.deepcopy(load_env_config(rc))
    cfg["realism"], cfg["push_enabled"], cfg["max_episode_steps"] = "medium", False, 500
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    policy, _ = load_policy(ckpt)
    env = LoggedEnv(config=cfg, render_mode=None)
    env.reward_mode = "track"; env.push_enabled = False; env.log = []
    command = np.array([float(cmd), 0.0, 0.0, *pad_budget(BUDGETS[budget], env)], np.float32)
    rows = [r for r in (episode(env, policy, command, s) for s in range(a, b)) if r]
    env.close()
    num = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if isinstance(rows[0][k], float)} if rows else {}
    if rows:  # the worst episode, not the mean of the episodes' worst
        num["gap_min_mm"], num["gap_max_mm"], num["tilt_max_deg"] = min(r["gap_min_mm"] for r in rows), max(r["gap_max_mm"] for r in rows), max(r["tilt_max_deg"] for r in rows)
    words = {k: max(set(r[k] for r in rows), key=[r[k] for r in rows].count) for k in rows[0] if isinstance(rows[0][k], str)} if rows else {}
    flags = []
    for k, (op, line, text) in FLAGS.items():
        v = num.get(k)
        if v is not None and ((op == ">" and v > line) or (op == "<" and v < line)):
            joint = words.get(k.replace("_share", "_joint"), "")
            flags.append(f"{k} = {v:.3g}{' (' + joint + ')' if joint else ''}: {text}")
    Path(out).write_text(json.dumps({"label": label, "command": float(cmd), "episodes_up": len(rows), "episodes": b - a, "summary": num, "joints": words, "flags": flags}, indent=1))
    print(label, f"cmd {cmd}", f"up {len(rows)}/{b - a}")
    for k, v in num.items():
        print(f"  {k} {v:.3g}")
    print("  FLAGS:" if flags else "  no flags")
    for f in flags:
        print("   -", f)


if __name__ == "__main__":
    main()
