"""validity.servo_model and validity.delay_mode: zoh (fast/valid_env.py, 13 Nov 2026; paper/v3_work/servo_fit/servo_fit.md, Section 6). CPU only.

Off by default; the fitted values are set at every realism and survive reset and motor strength; armature_range draws inside its range at full
realism with randomisation on and is put back at every reset; the zoh queue is emptied at reset; a zoh delay of whole control steps equals the
interpolated one exactly; and the env reproduces the servo-fit replay (paper/v3_work/servo_fit/scripts/replay.py) on a logged robot run to 1e-9.
"""
import copy
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BipedRobot.fast.valid_env import ValidLocoEnv  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

CFG = ROOT / "BipedRobot/config/v2_deploy_budget_alpha30_r17wS_fit.yaml"
SM = {"kp_scale": 1.8, "kv": 1.23, "armature": 0.010, "frictionloss": 0.12}
SCRIPTS = ROOT / "paper/v3_work/servo_fit/scripts"
LOG = Path("/home/rpuser1/biped_budget_her/robot_runs_10nov/policy_run_20261110_094553/log.csv")


def make(realism="full", validity=None, actuator=None, dr=False):
    config = yaml.safe_load(CFG.read_text())
    if actuator:
        config["actuator"].update(actuator)
    env_config = build_env_config(config, realism, False)
    env_config["validity"] = copy.deepcopy(config["validity"])
    env_config["validity"].update(validity or {})
    env_config["push_enabled"] = False
    env_config["domain_randomization"]["enabled"] = dr
    return ValidLocoEnv(config=env_config, render_mode=None)


def joints(env):
    return env.data.qpos[env.joint_qpos_adrs].copy()


def actions(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(np.cumsum(rng.normal(0.0, 0.08, (n, 12)), axis=0), -0.6, 0.6).astype(np.float32)


def rollout(env, A, delay=None, seed=3):
    env.reset(seed=seed)
    if delay is not None:
        env._active_delay = delay
    out = []
    for a in A:
        env.step(a)
        out.append(joints(env))
    return np.array(out)


def test_off_by_default():
    env = make()
    xml = mujoco.MjModel.from_xml_path(env.xml_path)
    ids, dofs = env.actuator_ids, env.joint_dof_adrs
    assert not env.delay_zoh and env._servo_model is None and env._arm_range is None
    np.testing.assert_array_equal(env.model.actuator_gainprm[ids, 0], xml.actuator_gainprm[ids, 0])
    np.testing.assert_array_equal(env.model.actuator_biasprm[ids, :3], xml.actuator_biasprm[ids, :3])
    np.testing.assert_array_equal(env.model.dof_armature[dofs], xml.dof_armature[dofs])
    np.testing.assert_array_equal(env.model.dof_frictionloss[dofs], xml.dof_frictionloss[dofs])
    A = actions(40)
    explicit = make(validity={"delay_mode": "interp"})
    np.testing.assert_array_equal(rollout(env, A), rollout(explicit, A))


@pytest.mark.parametrize("realism", ["medium", "full"])
def test_servo_model_values_kept(realism):
    env = make(realism, validity={"servo_model": SM}, dr=True)
    xml = mujoco.MjModel.from_xml_path(env.xml_path)
    ids, dofs = env.actuator_ids, env.joint_dof_adrs
    for seed in (1, 2):
        env.reset(seed=seed)
        np.testing.assert_allclose(env._kp0, xml.actuator_gainprm[ids, 0] * 1.8, rtol=1e-12)
        scale = env._torque_scale if env.motor_torque else np.ones(12)
        np.testing.assert_allclose(env.model.actuator_gainprm[ids, 0], xml.actuator_gainprm[ids, 0] * 1.8 * scale, rtol=1e-12)
        np.testing.assert_allclose(env.model.actuator_biasprm[ids, 2], -1.23)
        np.testing.assert_allclose(env.model.dof_armature[dofs], 0.010)
        np.testing.assert_allclose(env.model.dof_frictionloss[dofs], 0.12)


def test_bad_keys_raise():
    with pytest.raises(ValueError):
        make(validity={"servo_model": {"kp": 2.0}})
    with pytest.raises(ValueError):
        make(validity={"delay_mode": "linear"})


def test_armature_range_drawn_and_restored():
    sm = dict(SM, armature_range=[0.007, 0.014])
    env = make("full", validity={"servo_model": sm}, dr=True)
    dofs = env.joint_dof_adrs
    draws = []
    for seed in range(6):
        env.model.dof_armature[dofs] = 0.5  # anything left from before is replaced at reset
        env.reset(seed=seed)
        a = env.model.dof_armature[dofs].copy()
        assert np.all(a >= 0.007 - 1e-12) and np.all(a <= 0.014 + 1e-12)
        draws.append(a)
    assert np.std(np.array(draws)) > 1e-4
    env.set_difficulty(0.0)
    env.reset(seed=9)
    np.testing.assert_allclose(env.model.dof_armature[dofs], 0.010)  # difficulty 0: the fitted value
    env.set_difficulty(0.5)
    env.reset(seed=10)
    a = env.model.dof_armature[dofs]
    assert np.all(a >= 0.0085 - 1e-12) and np.all(a <= 0.012 + 1e-12)
    for realism, dr in (("medium", True), ("full", False)):
        e = make(realism, validity={"servo_model": sm}, dr=dr)
        e.model.dof_armature[e.joint_dof_adrs] = 0.5
        e.reset(seed=1)
        np.testing.assert_allclose(e.model.dof_armature[e.joint_dof_adrs], 0.010)  # evaluation and fixed runs: the fitted value


def test_zoh_queue_and_reset():
    env = make(validity={"delay_mode": "zoh"})
    env.reset(seed=1)
    np.testing.assert_array_equal(env._zoh_prev, np.clip(env.default_qpos, env.ctrl_min, env.ctrl_max).astype(np.float64))
    env._active_delay = 0.03  # 6 substeps: each step's target is still queued when the step ends
    A = actions(5)
    for k, a in enumerate(A):
        env.step(a)
        assert len(env._zq) == 1 and env._zq[0][0] == k * env.frame_skip + 6
    env.reset(seed=2)
    assert env._zq == [] and env._zoh_s == 0 and not env._zoh_new
    np.testing.assert_array_equal(env._zoh_prev, np.clip(env.default_qpos, env.ctrl_min, env.ctrl_max).astype(np.float64))
    assert env.enable_delay  # restored after every step


@pytest.mark.parametrize("realism", ["medium", "full"])
def test_zoh_whole_steps_equal_interpolated(realism):
    A = actions(60, seed=5)
    for d in (0.0, 0.02, 0.04):
        a = rollout(make(realism, validity={"delay_mode": "zoh", "servo_model": SM}), A, delay=d)
        b = rollout(make(realism, validity={"servo_model": SM}), A, delay=d)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    a = rollout(make(realism, validity={"delay_mode": "zoh", "servo_model": SM}), A, delay=0.01)
    b = rollout(make(realism, validity={"servo_model": SM}), A, delay=0.01)
    assert np.abs(a - b).max() > 1e-4  # half a step: interpolation averages two targets, zoh does not


@pytest.mark.parametrize("realism", ["medium", "full"])
def test_runs_with_randomisation(realism):
    sm = dict(SM, armature_range=[0.007, 0.014])
    env = make(realism, validity={"servo_model": sm, "delay_mode": "zoh"}, dr=True)
    for seed in (4, 5):
        env.reset(seed=seed)
        if realism == "medium":
            assert env._active_delay == pytest.approx(0.01)
        for a in actions(40, seed=seed):
            env.step(a)
        assert np.all(np.isfinite(env.data.qpos))


def test_zoh_with_servo_play_runs():
    env = make("full", validity={"delay_mode": "zoh", "servo_play": {"deg": [1.0, 1.0]}}, dr=True)
    assert env._play_on
    env.reset(seed=1)
    for a in actions(40):
        env.step(a)
    assert np.all(np.isfinite(env.data.qpos))


@pytest.mark.skipif(not (SCRIPTS / "replay.py").exists() or not LOG.exists(), reason="servo-fit scripts or robot log not on this machine")
def test_matches_servo_fit_replay_on_robot_log():
    sys.path.insert(0, str(SCRIPTS))
    import replay as R

    X = np.genfromtxt(LOG, delimiter=",", names=True)
    T = np.column_stack([X[f"target_{j}"] for j in range(12)])[100:400]
    Q = np.column_stack([X[f"q_{j}"] for j in range(12)])[100:400]
    act = {"enable_lag": False, "backlash_deadband": math.radians(0.5), "enable_backlash": True}
    ref_env = R.make_env(actuator=act)
    R.set_servo(ref_env, **SM)
    ref = R.replay(ref_env, T, Q[0], 0.018, duty=0.5, zoh_delay=True)
    env = R.make_env(actuator=act, validity={"servo_model": SM, "delay_mode": "zoh"})
    R.hang_reset(env, Q[0], T[0], 0.018, 0.5)
    out = []
    for a in R.target_to_action(env, T):
        env.step(a)
        out.append(joints(env))
    assert np.abs(np.array(out) - ref).max() <= 1e-9
