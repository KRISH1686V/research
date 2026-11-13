"""Evaluate one SB3 checkpoint at the two pilot monitoring commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import SAC

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.evaluation_suite import evaluate_command_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    env = HERLocoEnv(config=config, render_mode=None)
    model = SAC.load(args.checkpoint, env=env, device="cpu")

    def policy(observation):
        return model.predict(observation, deterministic=True)[0]

    rows = [
        evaluate_command_suite(policy, env, (vx, 0.0, 0.0), range(10), max_steps=500)
        for vx in (0.2, 0.4)
    ]
    env.close()
    Path(args.output).write_text(json.dumps({"checkpoint": args.checkpoint, "commands": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
