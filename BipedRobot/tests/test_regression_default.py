"""Default-path regression against a fixed tracking-only checkpoint."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml
from stable_baselines3 import SAC

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.evaluation_suite import evaluate_command_suite
from BipedRobot.train_her_biped import build_env_config


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "BipedRobot" / "paper_results" / "regression_fixed_checkpoint.json"


def test_default_tracking_regression():
    expected = json.loads(FIXTURE.read_text())
    with (ROOT / "BipedRobot" / "config" / "frozen_baseline.yaml").open() as handle:
        config = yaml.safe_load(handle)
    env_config = build_env_config(config, "easy", False)
    env = HERLocoEnv(config=copy.deepcopy(env_config))
    model = SAC.load(expected["checkpoint"], env=env, device="cpu")
    actual = evaluate_command_suite(
        lambda obs: model.predict(obs, deterministic=True)[0],
        env,
        expected["command"],
        expected["seeds"],
        max_steps=expected["max_steps"],
    )
    env.close()
    assert actual == expected["evaluation"]
