"""Record top-down root paths (x, y, yaw per control step) with the v2 evaluator's env and policy loaders.

Usage: paths.py LABEL CHECKPOINT RESOLVED_CONFIG REALISM VX OUT_JSON [SEEDS start:stop]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import EvalSettings, _yaw, load_env_config, load_policy, make_eval_env  # noqa: E402


def main():
    label, ckpt, rc, realism, vx, out = sys.argv[1:7]
    a, b = (int(x) for x in (sys.argv[7] if len(sys.argv) > 7 else "5000:5030").split(":"))
    torch.set_num_threads(1)
    settings = EvalSettings(realism=realism)
    cfg = load_env_config(rc)
    policy = load_policy(ckpt, cfg)
    env = make_eval_env(cfg, settings)
    cmd = np.array([float(vx), 0.0, 0.0], np.float32)
    eps = []
    for seed in range(a, b):
        obs, _ = env.reset(seed=seed, options={"desired_goal": cmd})
        x0, yaw0 = env.data.qpos[:2].copy(), _yaw(env.data.qpos[3:7])
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        pts, term, trunc, steps = [], False, False, 0
        while not (term or trunc) and steps < settings.max_steps:
            obs, _, term, trunc, _ = env.step(np.asarray(policy(obs), np.float32))
            dx, dy = env.data.qpos[:2] - x0
            yaw = float(np.angle(np.exp(1j * (_yaw(env.data.qpos[3:7]) - yaw0))))
            pts.append([round(c * dx - s * dy, 4), round(s * dx + c * dy, 4), round(yaw, 4)])
            steps += 1
        eps.append({"seed": seed, "survived": bool(steps >= settings.max_steps and not term), "steps": steps, "path": pts})
    env.close()
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "realism": realism, "command": cmd.tolist(),
                                     "frame": "start-heading frame: +x = initial forward, +y = initial left",
                                     "episodes": eps}))


if __name__ == "__main__":
    main()
