"""Read-only Phase 4 reward diagnostic; never trains or writes checkpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.train_her_biped import build_env_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml"
CHECKPOINTS = {
    "stageA0_stand_200k": ROOT / "BipedRobot/checkpoints/stageA0_stand/seed_42/sb3_her_model_200000_steps.zip",
    "stageA0_stand_safe_200k": ROOT / "BipedRobot/checkpoints/stageA0_stand_safe/seed_42/sb3_her_model_200000_steps.zip",
    "recovery_seed0_1100k": ROOT / "BipedRobot/checkpoints/recovery_m0_seed0_v2/seed_0/sb3_her_model_1100000_steps.zip",
    "recovery_seed1_1100k": ROOT / "BipedRobot/checkpoints/recovery_m0_seed1_v2/seed_1/sb3_her_model_1100000_steps.zip",
}


def _terms(env: HERLocoEnv, info: dict, achieved: np.ndarray, desired: np.ndarray) -> dict[str, float]:
    progress = (
        env.progress_weight * float(np.clip(achieved[0] / desired[0], 0.0, 1.0))
        if desired[0] > 0.1
        else 0.0
    )
    gait = info["r_gait_move"] if np.linalg.norm(desired[:2]) > 0.1 else info["r_gait_still"]
    return {
        "tracking": float(info["r_track"]),
        "progress": float(progress),
        "smooth": float(info["r_smooth"]),
        "flight": float(info["r_flight"]),
        "gait": float(gait),
        "total": float(env.compute_reward(achieved, desired, info)),
    }


def _run_policy(
    env_config: dict,
    policy_fn,
    command: np.ndarray,
    seeds: range,
) -> dict:
    rows = []
    for seed in seeds:
        env = HERLocoEnv(config=copy.deepcopy(env_config))
        obs, _ = env.reset(seed=seed, options={"desired_goal": command})
        sums = {key: 0.0 for key in ("tracking", "progress", "smooth", "flight", "gait", "total")}
        vx = []
        survived = False
        for _ in range(env.max_episode_steps):
            action = policy_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            step_terms = _terms(env, info, obs["achieved_goal"], obs["desired_goal"])
            for key, value in step_terms.items():
                sums[key] += value
            vx.append(float(obs["achieved_goal"][0]))
            if terminated or truncated:
                survived = truncated and not terminated
                break
        length = len(vx)
        rows.append(
            {
                **{key: value / max(1, length) for key, value in sums.items()},
                "mean_vx": float(np.mean(vx)) if vx else 0.0,
                "survived": float(survived and length == env.max_episode_steps),
                "steps": length,
            }
        )
        env.close()
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("tracking", "progress", "smooth", "flight", "gait", "total", "mean_vx", "survived")
    }


def _scripted_action(step: int, command: np.ndarray) -> np.ndarray:
    phase = 2.0 * np.pi * 1.2 * step * 0.02
    action = np.zeros(12, dtype=np.float32)
    if command[0] > 0.1:
        swing = np.sin(phase)
        action[[2, 8]] = [0.25 * swing, -0.25 * swing]
        action[[3, 9]] = [0.35 * max(0.0, -swing), 0.35 * max(0.0, swing)]
        action[[4, 10]] = [-0.20 * swing, 0.20 * swing]
    return action


def _scripted_policy(command: np.ndarray):
    step = 0

    def policy(_obs):
        nonlocal step
        action = _scripted_action(step, command)
        step += 1
        return action

    return policy


def main() -> None:
    with CONFIG.open() as handle:
        raw_config = yaml.safe_load(handle)
    env_config = build_env_config(raw_config, "easy", False)
    output = {"config": str(CONFIG), "commands": {}}
    for vx in (0.2, 0.4):
        command = np.array([vx, 0.0, 0.0], dtype=np.float32)
        command_output = {"standing_checkpoints": {}, "reference": {}}
        for name, path in CHECKPOINTS.items():
            if not path.exists():
                continue
            model_env = DummyVecEnv(
                [lambda env_config=env_config: HERLocoEnv(config=copy.deepcopy(env_config))]
            )
            model = SAC.load(str(path), env=model_env, device="cpu")
            command_output["standing_checkpoints"][name] = _run_policy(
                env_config,
                lambda obs, model=model: model.predict(obs, deterministic=True)[0],
                command,
                range(1000, 1010),
            )
            model_env.close()
        command_output["reference"]["scripted"] = _run_policy(
            env_config,
            _scripted_policy(command),
            command,
            range(1000, 1010),
        )
        output["commands"][str(vx)] = command_output
    destination = ROOT / "BipedRobot/paper_results/phase4_reward_diagnostic.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
