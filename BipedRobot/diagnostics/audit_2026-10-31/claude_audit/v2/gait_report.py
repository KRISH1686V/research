"""Why does a gait look off? Measures it from foot forces at every physics step: real steps against contact chatter, left-right balance, step length, turning, body sway and joint motion.

Usage: gait_report.py LABEL CHECKPOINT RESOLVED_CONFIG OUT_JSON REALISM SECONDS SEEDS(start:stop) BUDGET [COMMAND]
Same env and policy loading as survive_eval.py. No pushes. Only episodes that stay up count.
"""
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
from survive_eval import load_policy  # noqa: E402

FORCE_N = 0.5
REAL_STANCE_S = 0.06   # a foot loaded for less than this, or lifted for less than this, is chatter and not a step
FLAT_RAD = np.radians(5.0)  # a loaded foot tilted less than this from its standing orientation counts as a full-foot contact


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


def episode(env, policy, command, seed, steps):
    obs, _ = env.reset(seed=seed, options={"desired_goal": command})
    loads, base, joints, targets, acts, gyro, tilts, clears = [], [], [], [], [], [], [], []
    shape = FootShape(env)
    shape()

    def observe(fn, slip, nonfoot):
        loads.append(fn.copy())
        tilt, clear = shape()
        tilts.append(tilt)
        clears.append(clear)
    env.substep_observer = observe
    for _ in range(steps):
        a = np.asarray(policy(obs), np.float32)
        obs, _, term, trunc, _ = env.step(a)
        acts.append(a)
        base.append((*env.data.qpos[:3], yaw_of(env.data.qpos[3:7]), *env._get_projected_gravity()))
        joints.append([env.data.qvel[i] for i in env.joint_dof_adrs])
        gyro.append(env.data.qvel[3:6].copy())
        if term:
            return None
    env.substep_observer = None
    start = len(loads) // 10  # skip the first second
    fn = np.array(loads)[start:]
    tilt, clear = np.array(tilts)[start:], np.array(clears)[start:]
    dt = env.sim_dt
    out = {}
    real_steps, chatter, stance_s, swing_s, lifts = [], [], [], [], []
    for k in range(2):
        on = fn[:, k] > FORCE_N
        st, ln = segments(on)
        s_off, off = segments(~on)
        lifts += [float(clear[a:a + n, k].max()) for a, n in zip(s_off, off) if n * dt >= REAL_STANCE_S]
        real = ln * dt >= REAL_STANCE_S
        real_steps.append(int(real.sum()))
        chatter.append(int((~real).sum()))
        stance_s.append(ln[real] * dt)
        swing_s.append(off[off * dt >= REAL_STANCE_S] * dt)
    T = len(fn) * dt
    b = np.array(base)[len(base) // 10:]
    dist = float(np.hypot(*(b[-1, :2] - b[0, :2])))
    path = float(np.sum(np.hypot(*np.diff(b[:, :2], axis=0).T)))
    q = np.array(joints)[len(joints) // 10:]
    a = np.array(acts)
    loaded = fn > FORCE_N
    out.update(flat_stance_share=float((tilt[loaded] < FLAT_RAD).mean()) if loaded.any() else None,
               stance_tilt_deg_median=float(np.degrees(np.median(tilt[loaded]))) if loaded.any() else None,
               swing_lift_mm_median=1000 * float(np.median(lifts)) if lifts else None,
               flight_share=float((~loaded.any(axis=1)).mean()), single_support_share=float((loaded.sum(axis=1) == 1).mean()))
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
    policy, _ = load_policy(ckpt)
    env = LoggedEnv(config=cfg, render_mode=None)
    env.reward_mode = "track"
    env.push_enabled = False
    env.log = []
    command = np.array([speed, 0.0, 0.0, *pad_budget(BUDGETS[budget], env)], np.float32)
    rows = [r for r in (episode(env, policy, command, s, steps) for s in range(a, b)) if r]
    env.close()
    keys = rows[0].keys() if rows else []
    summary = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys}
    summary["episodes_up"] = len(rows)
    summary["episodes"] = b - a
    summary["turn_abs_rad_s"] = float(np.mean([abs(r["turn_rad_s"]) for r in rows])) if rows else None
    summary["sideways_abs_m"] = float(np.mean([abs(r["sideways_m"]) for r in rows])) if rows else None
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "budget": budget, "speed": speed, "summary": summary, "episodes": rows}))
    print(label, budget, f"cmd {speed}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
