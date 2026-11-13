"""validity.servo_play, added 10 Nov 2026: gear play inside the servo's position loop. Off unless named. CPU only."""
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BipedRobot.fast.collector import env_class  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "config/v2_deploy_budget_alpha30_r11v_fit.yaml"


def make(play=None, realism="medium"):
    c = yaml.safe_load(BASE.read_text())
    if play is not None:
        c["validity"]["servo_play"] = play
    c["actuator"]["enable_backlash"] = False
    ec = build_env_config(c, realism, False)
    for k in ("validity", "token_budget"):
        if k in c:
            ec[k] = c[k]
    return env_class(ec)(config=ec, render_mode=None)


def test_off_by_default_and_in_evaluation():
    env = make()
    env.reset(seed=1)
    assert not env._play_on and env._play_half is None
    env.close()
    env = make(play={"deg": [2.0, 2.0]}, realism="medium")
    env.reset(seed=1)
    assert not env._play_on  # evaluation realism keeps the old servo
    env.close()


def test_training_draws_the_play_and_moves_damping_to_the_motor():
    env = make(play={"deg": [2.0, 2.0], "link_armature": 0.005}, realism="full")
    env.set_difficulty(1.0)
    env.reset(seed=1)
    assert env._play_on and np.allclose(env._play_half, np.radians(1.0))
    assert np.allclose(env.model.dof_armature[env.joint_dof_adrs], 0.005)
    assert np.allclose(env.model.actuator_biasprm[env.actuator_ids, 2], 0.0)
    env.close()


def test_play_off_again_restores_the_model():
    env = make(play={"deg": [2.0, 2.0]}, realism="full")
    arm0 = env._arm0.copy()
    env.reset(seed=1)
    env._play_on = False
    env.reset(seed=2)
    assert np.allclose(env.model.dof_armature[env.joint_dof_adrs], arm0)
    assert np.allclose(-env.model.actuator_biasprm[env.actuator_ids, 2], env._kv0)
    env.close()


def test_holds_the_pose_without_blowing_up_and_target_is_kept():
    env = make(play={"deg": [2.0, 2.0]}, realism="full")
    env.set_difficulty(1.0)
    env.reset(seed=3)
    for _ in range(100):
        o, _, term, trunc, _ = env.step(np.zeros(12, np.float32))
        assert np.all(np.isfinite(env.data.qpos))
    assert np.allclose(env.data.ctrl[env.actuator_ids], env._play_target)  # the logged target is the command, not the trick value
    env.close()


def test_mesh_is_slack_inside_the_play_and_pushes_outside():
    env = make(play={"deg": [2.0, 2.0], "stiffness": 80.0, "damping": 0.0}, realism="full")
    env.set_difficulty(1.0)
    env.reset(seed=4)
    m, d = env.model, env.data
    q = d.qpos[env.joint_qpos_adrs].copy()
    env._play_target = q.copy()
    env._play_new = False
    env._play_w[:] = 0.0
    env._play_m = q + 0.5 * env._play_half  # inside the slack: no torque on the joint
    env._servo_play(m, d)
    assert np.allclose(d.ctrl[env.actuator_ids], q, atol=1e-9)
    env._play_m = q + 2.0 * env._play_half  # one half-play past contact: stiffness x that
    env._play_w[:] = 0.0
    env._servo_play(m, d)
    tau = m.actuator_gainprm[env.actuator_ids, 0] * (d.ctrl[env.actuator_ids] - q)
    assert np.allclose(tau, 80.0 * env._play_half, rtol=1e-3)
    env.close()


def test_motor_torque_respects_its_limits():
    env = make(play={"deg": [1.0, 1.0]}, realism="full")
    env.set_difficulty(1.0)
    env.reset(seed=5)
    lo, hi = env._play_limits(np.zeros(12))
    assert np.all(hi > 0) and np.all(lo < 0) and np.allclose(hi, env._cap * env._torque_scale)  # the same cap as the servo curve, with motor-strength randomisation
    lo, hi = env._play_limits(np.full(12, 100.0))  # far past no-load speed: no more drive that way
    assert np.allclose(hi, 0.0) and np.all(lo < 0)
    env.close()
