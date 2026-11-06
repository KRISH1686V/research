"""HER-consistency tests for the opt-in walking objective."""

from __future__ import annotations

import copy

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.train_her_biped import build_env_config


def _config(path: str, shaping: bool) -> dict:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if shaping:
        config["shaping"] = {
            "gait_shaping": True,
            "action_filter": True,
            "prev_action_obs": True,
            "smooth_penalties": True,
            "reset_velocity": True,
            "action_filter_alpha": 0.6,
            "progress_weight": 0.3,
            "air_time_weight": 3.0,
            "single_support_weight": 0.05,
            "swing_clearance_weight": 0.1,
            "flight_penalty_weight": -0.5,
            "zero_touchdown_weight": -0.3,
            "action_rate_weight": 0.05,
            "action_second_diff_weight": 0.002,
            "base_motion_weight": 0.05,
            "smoothness_ramp": 1.0,
            "reset_velocity_probability": 0.5,
        }
    return build_env_config(config, "easy", False)


def test_step_reward_matches_compute_reward_for_1000_transitions_with_flags():
    for shaping in (False, True):
        env = HERLocoEnv(
            config=copy.deepcopy(
                _config("BipedRobot/config/frozen_baseline.yaml", shaping)
            )
        )
        env.reset(seed=123, options={"desired_goal": np.array([0.2, 0.0, 0.0])})
        for index in range(1000):
            _, step_reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            recomputed = env.compute_reward(
                info["achieved_goal"], info["desired_goal"], info
            )
            assert abs(step_reward - recomputed) <= 1e-6, (
                f"reward mismatch at transition {index}, shaping={shaping}"
            )
            if terminated or truncated:
                env.reset(
                    seed=1000 + index,
                    options={"desired_goal": np.array([0.2, 0.0, 0.0])},
                )
        env.close()


def test_real_her_replay_samples_match_stored_and_relabelled_rewards():
    for shaping in (False, True):
        config = _config("BipedRobot/config/frozen_baseline.yaml", shaping)
        vec_env = DummyVecEnv([lambda: HERLocoEnv(config=copy.deepcopy(config))])
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": 2,
                "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
                "copy_info_dict": True,
            },
            buffer_size=1000,
            learning_starts=10000,
            batch_size=32,
            seed=7,
            device="cpu",
        )
        model.learn(total_timesteps=600, progress_bar=False)
        buffer = model.replay_buffer
        indices = np.array([0, 1, 2, 3])
        env_indices = np.zeros(4, dtype=np.int64)
        real = buffer._get_real_samples(indices, env_indices, None)
        for row, index in enumerate(indices):
            stored_info = buffer.infos[index, 0]
            expected = vec_env.envs[0].compute_reward(
                buffer.next_observations["achieved_goal"][index, 0],
                buffer.observations["desired_goal"][index, 0],
                stored_info,
            )
            assert abs(float(real.rewards[row].cpu().numpy().item()) - float(expected)) <= 1e-6
        virtual = buffer._get_virtual_samples(indices, env_indices, None)
        relabeled_goals = virtual.observations["desired_goal"].cpu().numpy()
        relabeled_achieved = buffer.next_observations["achieved_goal"][indices, 0]
        for row, info in enumerate(buffer.infos[indices, 0]):
            expected = vec_env.envs[0].compute_reward(
                relabeled_achieved[row],
                relabeled_goals[row],
                info,
            )
            assert abs(float(virtual.rewards[row].cpu().numpy().item()) - float(expected)) <= 1e-6
        sampled = buffer.sample(32, env=None)
        assert np.isfinite(sampled.rewards.cpu().numpy()).all()
        vec_env.close()
