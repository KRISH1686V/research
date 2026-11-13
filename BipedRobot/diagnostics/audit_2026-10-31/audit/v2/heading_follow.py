"""Straight-line following with an outer-loop heading controller (evaluation only, no retraining).

Each control step the controller reads the root heading psi and lateral offset y in the start-heading
frame, aims back at the line (psi_ref = -atan(K_Y * y)) and commands wz = clip(K_PSI * (psi_ref - psi),
+-WZ_MAX) through the policy's desired_goal. WZ_MAX defaults to 0.15 rad/s, the trained wz range.
The env and policy are loaded exactly as in evaluation_suite_v2; only the commanded wz changes.

Usage: heading_follow.py LABEL CHECKPOINT RESOLVED_CONFIG REALISM VX OUT_JSON [K_PSI K_Y WZ_MAX [SEEDS start:stop] [--pushes]]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import EvalSettings, _yaw, load_env_config, load_policy, make_eval_env  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if a != "--pushes"]
    label, ckpt, rc, realism, vx, out = args[:6]
    k_psi, k_y, wz_max = (float(x) for x in (args[6:9] if len(args) >= 9 else (1.0, 1.0, 0.15)))
    a, b = (int(x) for x in (args[9] if len(args) > 9 else "5000:5030").split(":"))
    torch.set_num_threads(1)
    settings = EvalSettings(realism=realism, pushes="--pushes" in sys.argv)
    cfg = load_env_config(rc)
    policy = load_policy(ckpt, cfg)
    env = make_eval_env(cfg, settings)
    cmd = np.array([float(vx), 0.0, 0.0], np.float32)
    eps = []
    for seed in range(a, b):
        obs, _ = env.reset(seed=seed, options={"desired_goal": cmd})
        x0, yaw0 = env.data.qpos[:2].copy(), _yaw(env.data.qpos[3:7])
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        pts, wzs, term, trunc, steps = [], [], False, False, 0
        px = py = psi = 0.0
        while not (term or trunc) and steps < settings.max_steps:
            psi_ref = -np.arctan(k_y * py)
            wz = float(np.clip(k_psi * np.angle(np.exp(1j * (psi_ref - psi))), -wz_max, wz_max))
            o = dict(obs)
            o["desired_goal"] = np.array([cmd[0], cmd[1], wz], np.float32)
            obs, _, term, trunc, _ = env.step(np.asarray(policy(o), np.float32))
            dx, dy = env.data.qpos[:2] - x0
            px, py = c * dx - s * dy, s * dx + c * dy
            psi = float(np.angle(np.exp(1j * (_yaw(env.data.qpos[3:7]) - yaw0))))
            pts.append([round(px, 4), round(py, 4), round(psi, 4)])
            wzs.append(round(wz, 4))
            steps += 1
        eps.append({"seed": seed, "survived": bool(steps >= settings.max_steps and not term), "steps": steps,
                    "path": pts, "wz_cmd": wzs})
    env.close()
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "realism": realism, "command": cmd.tolist(),
                                     "controller": {"k_psi": k_psi, "k_y": k_y, "wz_max": wz_max},
                                     "pushes": settings.pushes,
                                     "frame": "start-heading frame: +x = initial forward, +y = initial left",
                                     "episodes": eps}))


if __name__ == "__main__":
    main()
