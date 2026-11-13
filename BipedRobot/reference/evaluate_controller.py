"""Evaluate the closed-loop walking reference offline."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.train_her_biped import build_env_config
from BipedRobot.reference.walking_controller import ClosedLoopWalkingController, WalkingParameters

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml"


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def evaluate(config: dict, command_vx: float, seed: int, params: WalkingParameters) -> dict:
    env = HERLocoEnv(config=copy.deepcopy(config))
    command = np.array([command_vx, 0.0, 0.0], dtype=np.float32)
    obs, _ = env.reset(seed=seed, options={"desired_goal": command})
    controller = ClosedLoopWalkingController(env, command_vx, params)
    rewards = []
    velocities = []
    contacts = []
    heights = []
    terminated = False
    for _ in range(env.max_episode_steps):
        action = controller.action()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        velocities.append(float(obs["achieved_goal"][0]))
        contacts.append(env._feet_contact().copy())
        heights.append([float(env.data.xpos[body_id, 2]) for body_id in env.foot_body_ids])
        if terminated or truncated:
            break
    env.close()
    velocity = np.asarray(velocities)
    contact = np.asarray(contacts)
    steady = velocity[100:] if len(velocity) > 100 else velocity
    return {
        "command_vx": command_vx,
        "seed": seed,
        "steps": len(rewards),
        "survived": not terminated and len(rewards) == env.max_episode_steps,
        "mean_vx": float(np.mean(velocity)) if len(velocity) else 0.0,
        "full_rmse_vx": float(np.sqrt(np.mean((velocity - command_vx) ** 2))) if len(velocity) else 0.0,
        "steady_rmse_vx": float(np.sqrt(np.mean((steady - command_vx) ** 2))) if len(steady) else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "double_support_fraction": float(np.mean(np.sum(contact, axis=1) == 2)) if len(contact) else 1.0,
        "flight_fraction": float(np.mean(np.sum(contact, axis=1) == 0)) if len(contact) else 1.0,
        "peak_swing_height_m": float(np.max(heights)) if heights else 0.0,
        "action_saturation_fraction": controller.mean_action_saturation,
    }


def main() -> None:
    with CONFIG_PATH.open() as handle:
        raw = yaml.safe_load(handle)
    config = build_env_config(raw, "easy", False)
    config["domain_randomization"] = {
        **config.get("domain_randomization", {}),
        "enabled": False,
    }
    config["push_enabled"] = False
    params = WalkingParameters()
    output = {
        "git_commit": git_hash(),
        "config_path": str(CONFIG_PATH),
        "config": config,
        "controller_parameters": params.__dict__,
        "results": [
            evaluate(config, command, seed, params)
            for command in (0.2, 0.4, 0.6)
            for seed in range(1000, 1003)
        ],
    }
    destination = ROOT / "BipedRobot/paper_results/closed_loop_reference_dr_off.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
