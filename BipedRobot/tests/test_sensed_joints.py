"""Joint sensing as on the real robot (validity.encoder_rad, validity.joint_vel_from_position). Run by path from the workspace root."""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import BudgetLocoEnv  # noqa: E402
from BipedRobot.fast.valid_env import ValidLocoEnv  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STEP = 0.00153398


def config(realism="easy", **validity):
    c = yaml.safe_load(open(REPO / "config/v2_shaped.yaml"))
    ec = build_env_config(c, realism, False)
    ec["validity"] = validity
    ec.setdefault("shaping", {})["reset_velocity"] = False
    return ec


def make(c):
    env = ValidLocoEnv(config=copy.deepcopy(c))
    env.reward_mode = "track"
    return env


def test_off_by_default_nothing_changes():
    a = make(config(history=2))
    assert a.encoder_rad == 0.0 and a.vel_from_position is False
    o, _ = a.reset(seed=1)
    frame = o["observation"][42:]
    truth = (np.array([a.data.qvel[x] for x in a.joint_dof_adrs]) * a.obs_scale["joint_vel"]).astype(np.float32)
    assert np.allclose(frame[12:24], truth, atol=1e-6), "speed stays the simulator's true speed"
    a.close()


def test_angles_are_rounded_to_encoder_steps():
    env = make(config(encoder_rad=STEP))
    o, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for _ in range(30):
        q = o["observation"][:12]
        assert np.allclose(q / STEP, np.round(q / STEP), atol=1e-3)
        o, *_ = env.step(rng.uniform(-0.2, 0.2, 12).astype(np.float32))
    env.close()


def test_speed_is_the_difference_of_observed_angles():
    env = make(config(encoder_rad=STEP, joint_vel_from_position=True))
    scale = env.obs_scale["joint_vel"]
    o, _ = env.reset(seed=1)
    assert np.array_equal(o["observation"][12:24], np.zeros(12, np.float32)), "the first reading has no previous angle, as on the robot"
    rng = np.random.default_rng(0)
    for _ in range(40):
        before = o["observation"][:12].copy()
        o, _, term, _, _ = env.step(rng.uniform(-0.2, 0.2, 12).astype(np.float32))
        want = (o["observation"][:12] - before) / env.dt * scale
        assert np.allclose(o["observation"][12:24], want, atol=1e-5)
        if term:
            break
    env.close()


def test_noise_and_bias_reach_the_speed_only_through_the_angles():
    env = make(config("full", joint_vel_from_position=True, obs_noise={"joint_pos": 0.002, "joint_vel": 0.0}, obs_bias={"joint_pos": 0.0175}))
    o, _ = env.reset(seed=3)
    bias_seen = o["observation"][:12].copy()
    o, *_ = env.step(np.zeros(12, np.float32))
    v = o["observation"][12:24] / env.obs_scale["joint_vel"]
    assert 0.01 < float(np.std(v)) < 0.6, "angle noise of 0.002 rad over 20 ms gives about 0.14 rad/s, plus real motion"
    assert np.max(np.abs(v)) < 5.0, "a constant bias must not look like speed"
    assert not np.array_equal(bias_seen, np.zeros(12))
    env.close()


def test_a_new_episode_forgets_the_last_angle():
    env = make(config(joint_vel_from_position=True))
    env.reset(seed=1)
    env.step(np.ones(12, np.float32) * 0.5)
    o, _ = env.reset(seed=2)
    assert np.array_equal(o["observation"][12:24], np.zeros(12, np.float32))
    env.close()


def test_history_holds_the_sensed_values_and_critic_keeps_the_truth():
    env = make(config(encoder_rad=STEP, joint_vel_from_position=True, history=3))
    env.reset(seed=1)
    for _ in range(3):
        o, *_ = env.step(np.ones(12, np.float32) * 0.2)
    newest = o["observation"][84:]
    prior = o["observation"][42:84]
    assert np.allclose(newest[12:24], (newest[:12] - prior[:12]) / env.dt * env.obs_scale["joint_vel"], atol=1e-5)
    truth = np.array([env.data.qvel[x] for x in env.joint_dof_adrs]) * env.obs_scale["joint_vel"]
    assert np.allclose(env.privileged()[12:24], truth, atol=1e-6), "the critic still sees the true speed"
    assert not np.allclose(newest[12:24], truth, atol=1e-3)
    env.close()


def test_deploy_configs_load_and_match_each_other():
    for name, gait, budget in (("v2_deploy_budget", False, True), ("v2_deploy_budget_gait", True, True), ("v2_deploy_cmd", False, False), ("v2_deploy_cmd_gait", True, False)):
        c = yaml.safe_load(open(REPO / f"config/{name}.yaml"))
        v = c["validity"]
        assert v["joint_vel_from_position"] is True and v["encoder_rad"] == STEP and v["obs_noise"]["joint_vel"] == 0.0
        assert v["obs_noise"]["joint_pos"] == 0.002 and v["history"] == 5
        assert c["shaping"]["gait_shaping"] is gait
        assert ("window_steps" in c["token_budget"]) is budget
        ec = build_env_config(c, "full", False)
        ec["validity"], ec["token_budget"] = c["validity"], c["token_budget"]
        env = BudgetLocoEnv(config=ec)
        env.reward_mode = "track"
        o, _ = env.reset(seed=1)
        assert o["observation"].shape == (213,)
        o, *_ = env.step(np.zeros(12, np.float32))
        env.close()
