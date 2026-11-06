"""Stage A gates for shared rewards, standing, HER, evaluation, and scaling."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.train_her_biped import build_env_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "BipedRobot" / "config" / "her_loco_config.yaml"


def stage_config(realism: str = "easy", stand_only: bool = False):
    with CONFIG_PATH.open() as handle:
        config = yaml.safe_load(handle)
    return build_env_config(config, realism, stand_only)


def make_env(config):
    return HERLocoEnv(config=copy.deepcopy(config), render_mode=None)


def test_t1_step_reward_equals_compute_reward_for_random_transitions():
    env = make_env(stage_config("full"))
    observation, _ = env.reset(seed=42)
    for step_index in range(1000):
        next_observation, step_reward, terminated, truncated, info = env.step(env.action_space.sample())
        recomputed = env.compute_reward(
            next_observation["achieved_goal"], next_observation["desired_goal"], info
        )
        assert abs(step_reward - recomputed) <= 1e-6, f"mismatch at transition {step_index}"
        if terminated or truncated:
            observation, _ = env.reset(seed=step_index + 1000)
        else:
            observation = next_observation
    env.close()


def test_t2_track_reward_bounds_and_fall_penalty():
    config = stage_config("easy")
    config["reward"]["fall_penalty"] = 5.0
    env = make_env(config)
    achieved = np.zeros(env.goal_dim, dtype=np.float32)
    desired = np.zeros(env.goal_dim, dtype=np.float32)
    normal = env.compute_reward(achieved, desired, {"fell": False})
    fallen = env.compute_reward(achieved, desired, {"fell": True})
    assert 0.0 <= normal <= 1.0
    assert fallen == pytest.approx(-5.0)
    env.close()


@pytest.mark.parametrize("realism", ["easy", "full"])
def test_t3_zero_action_stands(realism):
    env_config = stage_config(realism, stand_only=True)
    for seed in range(5):
        env = make_env(env_config)
        observation, _ = env.reset(seed=seed, options={"desired_goal": np.zeros(env.goal_dim)})
        start_height = float(env.data.qpos[2])
        minimum_height = start_height
        minimum_upright = 1.0
        maximum_saturation = 0.0
        for _ in range(500):
            observation, _, terminated, truncated, info = env.step(np.zeros(env.num_joints, dtype=np.float32))
            minimum_height = min(minimum_height, float(env.data.qpos[2]))
            minimum_upright = min(minimum_upright, env._upright_cosine())
            maximum_saturation = max(maximum_saturation, float(info["torque_saturation_ratio"]))
            assert not terminated
            assert not truncated or env._step_count == 500
        assert abs(start_height - minimum_height) <= 0.03
        assert minimum_upright > 0.99
        assert maximum_saturation < 0.01
        env.close()


def test_t4_action_mapping_reaches_configured_fraction():
    env = make_env(stage_config("easy", stand_only=True))
    assert np.allclose(env._action_to_target(np.zeros(env.num_joints)), env.default_qpos)
    positive = env._action_to_target(np.ones(env.num_joints))
    negative = env._action_to_target(-np.ones(env.num_joints))
    expected_positive = np.clip(
        env.default_qpos + env.action_range_fraction * (env.jnt_max - env.default_qpos),
        env.ctrl_min,
        env.ctrl_max,
    )
    expected_negative = np.clip(
        env.default_qpos - env.action_range_fraction * (env.default_qpos - env.jnt_min),
        env.ctrl_min,
        env.ctrl_max,
    )
    assert np.allclose(positive, expected_positive)
    assert np.allclose(negative, expected_negative)
    env.close()


def test_t5_her_replay_recomputes_reward_and_preserves_fell():
    env_config = stage_config("easy", stand_only=True)
    vec_env = DummyVecEnv([lambda: make_env(env_config)])
    model = SAC(
        "MultiInputPolicy",
        vec_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": GoalSelectionStrategy.FUTURE,
            "copy_info_dict": True,
        },
        buffer_size=2_000,
        learning_starts=10_000,
        batch_size=32,
        seed=42,
        device="cpu",
    )
    model.learn(total_timesteps=600, progress_bar=False)
    buffer = model.replay_buffer
    assert buffer.pos > 0 or buffer.full
    assert any("fell" in info for row in buffer.infos for info in row if isinstance(info, dict))
    samples = buffer.sample(32, env=None)
    relabeled_rewards = samples.rewards.cpu().numpy().reshape(-1)
    assert np.all(np.isfinite(relabeled_rewards))
    vec_env.close()


def test_t6_evaluate_uses_actual_achieved_goal():
    source = (ROOT / "evaluate.py").read_text()
    assert '"achieved_goal": achieved[None, :]' in source
    assert '"achieved_goal": goal[None, :]' not in source


def test_t7_reset_restores_nominal_randomized_parameters():
    config = stage_config("full")
    env = make_env(config)
    nominal_friction = env._nominal_geom_friction.copy()
    nominal_mass = env._nominal_torso_mass
    env.reset(seed=1)
    env.dr_enabled = False
    env.reset(seed=2)
    assert np.allclose(env.model.geom_friction, nominal_friction)
    assert env.model.body_mass[env.torso_body_id] == pytest.approx(nominal_mass)
    env.close()


def test_t8_twenty_k_step_smoke_training():
    exp_name = "stageA_test_t8_smoke"
    run_dir = ROOT / "BipedRobot" / "runs" / exp_name / "seed_42"
    checkpoint_dir = ROOT / "BipedRobot" / "checkpoints" / exp_name / "seed_42"
    if run_dir.exists() or checkpoint_dir.exists():
        pytest.skip("T8 experiment directory already exists; refusing to overwrite it")
    command = [
        sys.executable,
        str(ROOT / "BipedRobot" / "train_her_biped.py"),
        "--exp_name", exp_name,
        "--realism", "easy",
        "--stand_only",
        "--total_steps", "20000",
        "--seed", "42",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    assert (checkpoint_dir / "sb3_her_final.zip").exists()


def test_t9_observation_scaling():
    env = make_env(stage_config("easy", stand_only=True))
    env.reset(seed=0)
    raw_joint_pos = np.full(12, 0.2, dtype=np.float32)
    raw_joint_vel = np.full(12, 2.0, dtype=np.float32)
    raw_ang_vel = np.full(3, 4.0, dtype=np.float32)
    env.data.qvel[3:6] = raw_ang_vel
    for index, address in enumerate(env.joint_qpos_adrs):
        env.data.qpos[address] = env.default_qpos[index] + raw_joint_pos[index]
    for index, address in enumerate(env.joint_dof_adrs):
        env.data.qvel[address] = raw_joint_vel[index]
    env._prev_action[:] = 0.3
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    observation = env._get_obs_dict()["observation"]
    assert np.allclose(observation[:12], raw_joint_pos * env.obs_scale["joint_pos"])
    assert np.allclose(observation[12:24], raw_joint_vel * env.obs_scale["joint_vel"])
    assert np.allclose(observation[24:27], raw_ang_vel * env.obs_scale["base_ang_vel"])
    assert np.allclose(observation[36:48], 0.3 * env.obs_scale["prev_action"])
    env.close()
