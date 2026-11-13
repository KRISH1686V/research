"""Offline search for command-adaptive periodic walking references."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.train_her_biped import build_env_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml"


def rollout(config: dict, command_vx: float, params: tuple[float, float, float, float, float], seed: int) -> dict:
    frequency, hip, knee, ankle, phase_offset = params
    env = HERLocoEnv(config=copy.deepcopy(config))
    command = np.array([command_vx, 0.0, 0.0], dtype=np.float32)
    obs, _ = env.reset(seed=seed, options={"desired_goal": command})
    rewards = []
    velocities = []
    for step in range(env.max_episode_steps):
        phase = 2.0 * np.pi * frequency * step * env.dt + phase_offset
        sine = np.sin(phase)
        action = np.zeros(12, dtype=np.float32)
        action[[2, 8]] = [hip * sine, -hip * sine]
        action[[3, 9]] = [knee * max(0.0, -sine), knee * max(0.0, sine)]
        action[[4, 10]] = [-ankle * sine, ankle * sine]
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        velocities.append(float(obs["achieved_goal"][0]))
        if terminated or truncated:
            break
    env.close()
    survived = len(rewards) == 500 and not terminated
    return {
        "command_vx": command_vx,
        "frequency": frequency,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "phase_offset": phase_offset,
        "steps": len(rewards),
        "survived": survived,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_vx": float(np.mean(velocities)) if velocities else 0.0,
        "rmse_vx": float(np.sqrt(np.mean((np.asarray(velocities) - command_vx) ** 2)))
        if velocities
        else float("inf"),
    }


def main() -> None:
    with CONFIG.open() as handle:
        raw = yaml.safe_load(handle)
    config = build_env_config(raw, "easy", False)
    results = []
    for command_vx in (0.2, 0.4, 0.6):
        candidates = [
            (frequency, hip, knee, ankle, phase_offset)
            for frequency in (1.2, 1.5, 1.8, 2.1, 2.4)
            for hip in (0.16, 0.24, 0.32, 0.40, 0.48)
            for knee in (0.20, 0.30, 0.40, 0.50)
            for ankle in (0.08, 0.16, 0.24, 0.32)
            for phase_offset in (0.0, np.pi / 4.0)
        ]
        command_results = [rollout(config, command_vx, params, 1000) for params in candidates]
        command_results.sort(
            key=lambda row: (
                row["survived"],
                -row["rmse_vx"],
                row["mean_reward"],
            ),
            reverse=True,
        )
        results.extend(command_results[:20])
        print(json.dumps(command_results[:3], indent=2))
    destination = ROOT / "BipedRobot/paper_results/adaptive_walking_reference_search.json"
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
