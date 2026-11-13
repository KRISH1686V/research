"""Measure reset-velocity effects for a zero-action stand-still policy."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from BipedRobot.envs.her_loco_env import HERLocoEnv

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows = []
    for config_name in ("A", "A-min"):
        path = ROOT / (
            "BipedRobot/config/pilot_gait_smooth_a.yaml"
            if config_name == "A"
            else "BipedRobot/config/pilot_gait_min.yaml"
        )
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        for vx in (0.2, 0.3, 0.4, 0.6):
            endings = []
            returns = []
            for seed in range(9000, 9100):
                env = HERLocoEnv(config=copy.deepcopy(config), render_mode=None)
                _, reset_info = env.reset(
                    seed=seed, options={"desired_goal": np.array([vx, 0.0, 0.0])}
                )
                total = 0.0
                steps = 0
                terminated = False
                truncated = False
                while not (terminated or truncated) and steps < 500:
                    _, reward, terminated, truncated, _ = env.step(
                        np.zeros(env.num_joints, dtype=np.float32)
                    )
                    total += float(reward)
                    steps += 1
                env.close()
                endings.append(steps <= 50)
                returns.append(total)
            rows.append(
                {
                    "config": config_name,
                    "vx": vx,
                    "episodes": 100,
                    "fraction_ending_within_50_steps": float(np.mean(endings)),
                    "mean_return": float(np.mean(returns)),
                }
            )
    output = {
        "pilot_ranges": {
            "vx": [0.2, 0.6],
            "vy": [-0.2, 0.2],
            "wz": [-0.5, 0.5],
            "p_zero_cmd": 0.20,
            "curriculum_scale": 1.0,
            "reset_velocity_probability": 0.5,
            "reset_velocity_factor_range": [0.5, 1.0],
        },
        "rows": rows,
    }
    destination = ROOT / "BipedRobot/paper_results/reset_velocity_check.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
