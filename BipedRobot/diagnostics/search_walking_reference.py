"""Search simple periodic controllers for a valid offline walking reference."""

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


def action(
    step: int,
    dt: float,
    frequency: float,
    hip: float,
    knee: float,
    ankle: float,
    phase_offset: float,
) -> np.ndarray:
    phase = 2.0 * np.pi * frequency * step * dt + phase_offset
    sine = np.sin(phase)
    action = np.zeros(12, dtype=np.float32)
    action[[2, 8]] = [hip * sine, -hip * sine]
    action[[3, 9]] = [knee * max(0.0, -sine), knee * max(0.0, sine)]
    action[[4, 10]] = [-ankle * sine, ankle * sine]
    return action


def evaluate(
    config: dict,
    params: tuple[float, float, float, float, float],
    command_vx: float,
    seed: int = 1000,
) -> dict:
    frequency, hip, knee, ankle, phase_offset = params
    env = HERLocoEnv(config=copy.deepcopy(config))
    command = np.array([command_vx, 0.0, 0.0], dtype=np.float32)
    obs, _ = env.reset(seed=seed, options={"desired_goal": command})
    rewards, velocities = [], []
    fell = False
    for step in range(env.max_episode_steps):
        obs, reward, terminated, truncated, _ = env.step(
            action(step, env.dt, frequency, hip, knee, ankle, phase_offset)
        )
        rewards.append(float(reward))
        velocities.append(float(obs["achieved_goal"][0]))
        if terminated or truncated:
            fell = bool(terminated)
            break
    env.close()
    return {
        "frequency": frequency,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "phase_offset": phase_offset,
        "command_vx": command_vx,
        "steps": len(rewards),
        "survived": not fell and len(rewards) == 500,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_vx": float(np.mean(velocities)) if velocities else 0.0,
    }


def main() -> None:
    with CONFIG.open() as handle:
        raw = yaml.safe_load(handle)
    config = build_env_config(raw, "easy", False)
    candidates = [
        (frequency, hip, knee, ankle, phase_offset)
        for frequency in (1.35, 1.45, 1.50, 1.55, 1.65)
        for hip in (0.16, 0.20, 0.24)
        for knee in (0.21, 0.25, 0.29)
        for ankle in (0.06, 0.10, 0.14)
        for phase_offset in (0.0,)
    ]
    results = [
        evaluate(config, candidate, command_vx)
        for command_vx in (0.2, 0.4, 0.6)
        for candidate in candidates
    ]
    results.sort(
        key=lambda row: (
            row["survived"],
            -abs(row["mean_vx"] - row["command_vx"]),
            row["mean_reward"],
        ),
        reverse=True,
    )
    destination = ROOT / "BipedRobot/paper_results/walking_reference_search.json"
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    main()
