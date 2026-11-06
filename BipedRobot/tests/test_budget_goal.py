"""Opt-in budget-goal and budget-HER regression tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.her.budget_her_replay_buffer import BudgetHerReplayBuffer
from BipedRobot.train_her_biped import build_env_config


def budget_config():
    with open("BipedRobot/config/frozen_baseline.yaml") as handle:
        config = yaml.safe_load(handle)
    config["budget_goal"] = {
        "enabled": True,
        "costs": ["torque_rms", "torso_tilt", "foot_slip", "action_rate"],
        "d_max": [1.0, 0.2, 1.0, 0.2],
        "min_budget": [0.1, 0.1, 0.1, 0.1],
        "p_all_unconstrained": 0.2,
        "p_budget_relabel": 0.6,
    }
    env_config = build_env_config(config, "easy", False)
    env_config["budget_goal"] = copy.deepcopy(config["budget_goal"])
    return env_config


def test_budget_goal_shape_and_ranges():
    env = HERLocoEnv(config=budget_config())
    assert env.goal_dim == 7
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        assert obs["achieved_goal"].shape == (7,)
        assert obs["desired_goal"].shape == (7,)
        assert np.all(obs["desired_goal"][3:] >= env.budget_min)
        assert np.all(obs["desired_goal"][3:] <= 1.0)
    env.close()


def test_budget_reward_recomputes_from_goals_and_preserves_fall():
    env = HERLocoEnv(config=budget_config())
    achieved = np.array([0.2, 0.0, 0.0, 0.5, 0.2, 0.1, 0.1], dtype=np.float32)
    desired = np.array([0.2, 0.0, 0.0, 0.5, 0.2, 0.1, 0.1], dtype=np.float32)
    assert env.compute_reward(achieved, desired, {"fell": False}) == pytest.approx(1.0)
    assert env.compute_reward(achieved, desired, {"fell": True}) == pytest.approx(
        -env.fall_penalty
    )
    env.close()


def test_budget_her_buffer_p_zero_keeps_virtual_goal_path():
    config = budget_config()
    vec_env = DummyVecEnv([lambda: HERLocoEnv(config=copy.deepcopy(config))])
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=BudgetHerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 2,
            "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
            "copy_info_dict": True,
            "p_budget_relabel": 0.0,
            "min_budget": config["budget_goal"]["min_budget"],
        },
        buffer_size=1000,
        learning_starts=10000,
        batch_size=32,
        seed=7,
        device="cpu",
    )
    model.learn(total_timesteps=600, progress_bar=False)
    real_before = model.replay_buffer.observations["desired_goal"].copy()
    real_sample_before = model.replay_buffer._get_real_samples(
        np.array([0]), np.array([0]), None
    ).observations["desired_goal"].clone()
    samples = model.replay_buffer.sample(16, env=None)
    assert samples.observations["desired_goal"].shape == (16, 7)
    assert samples.next_observations["achieved_goal"].shape == (16, 7)
    assert np.isfinite(samples.rewards.cpu().numpy()).all()
    assert np.array_equal(real_before, model.replay_buffer.observations["desired_goal"])
    real_sample_after = model.replay_buffer._get_real_samples(
        np.array([0]), np.array([0]), None
    ).observations["desired_goal"]
    assert np.array_equal(real_sample_before.numpy(), real_sample_after.numpy())
    vec_env.close()
