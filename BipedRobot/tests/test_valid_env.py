"""Each ValidLocoEnv correction must do what it says, and nothing when off. Run by path from the workspace root."""
import copy
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.envs import her_loco_env as hle  # noqa: E402
from BipedRobot.envs.her_loco_env import HERLocoEnv  # noqa: E402
from BipedRobot.fast.valid_env import ValidLocoEnv  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def cfg(realism="full", **validity):
    c = build_env_config(yaml.safe_load(open(REPO / "config/pilot_gait_min_v2.yaml")), realism, False)
    c["validity"] = validity
    return c


def settle(env, n=40):
    for _ in range(n):
        env.step(np.zeros(12, np.float32))


def floor_contact_friction(env):
    d = env.data
    return [float(d.contact[i].friction[0]) for i in range(d.ncon) if env.floor_geom_id in (d.contact[i].geom1, d.contact[i].geom2)]


def test_unchanged_when_off():
    a, b = HERLocoEnv(config=cfg()), ValidLocoEnv(config=cfg())
    a.reward_mode = b.reward_mode = "track"
    a.push_enabled = b.push_enabled = True
    rng = np.random.default_rng(1)
    oa, _ = a.reset(seed=5)
    ob, _ = b.reset(seed=5)
    for _ in range(400):
        act = rng.uniform(-0.3, 0.3, 12).astype(np.float32)
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
        assert ra == rb and ta == tb and tra == trb
        assert all(np.array_equal(oa[k], ob[k]) for k in oa)
        if ta or tra:
            oa, _ = a.reset(seed=6)
            ob, _ = b.reset(seed=6)
    a.close()
    b.close()


def test_friction_correction_makes_low_friction_real():
    for fix, want_low in ((False, 1.0), (True, 0.6)):
        env = ValidLocoEnv(config=cfg("medium", friction=fix))
        env.dr_friction_range = (0.6, 0.6)
        env.reset(seed=0)
        settle(env)
        got = floor_contact_friction(env)
        assert got, "robot must be on the floor"
        assert np.allclose(got, want_low, atol=1e-6), (fix, got)
        env.close()


def _count_push_substeps(env):
    seen = []
    class Hook:
        def __getattr__(self, n):
            return getattr(mujoco, n)
        def mj_step(self, m, d):
            mujoco.mj_step(m, d)
            seen.append(float(np.linalg.norm(d.xfrc_applied[env.torso_body_id, :3])))
    env.push_enabled = True
    env.reset(seed=1)
    a = np.zeros(12, np.float32)
    original = hle.mujoco
    hle.mujoco = Hook()
    try:
        for _ in range(env.dr_push_interval + 8):
            env.step(a)
    finally:
        hle.mujoco = original
    return sum(x > 0 for x in seen)


def test_push_duration():
    assert _count_push_substeps(ValidLocoEnv(config=cfg())) == 5
    assert _count_push_substeps(ValidLocoEnv(config=cfg(push_s=0.1))) == 20


def test_motor_torque_scales_force_not_target():
    env = ValidLocoEnv(config=cfg("full", motor_torque=True))
    env.dr_motor_strength_range = (0.8, 0.8)
    env.reset(seed=2)
    assert np.allclose(env.model.actuator_forcerange[env.actuator_ids, 1], 2.94 * 0.8)
    assert np.allclose(env.model.actuator_gainprm[env.actuator_ids, 0], 21.1 * 0.8)
    assert np.allclose(env._motor_strength_scale, 1.0)
    env.close()
    old = ValidLocoEnv(config=cfg("full"))
    old.dr_motor_strength_range = (0.8, 0.8)
    old.reset(seed=2)
    assert np.allclose(old.model.actuator_forcerange[old.actuator_ids, 1], 2.94)
    assert np.allclose(old._motor_strength_scale, 0.8)
    old.close()


def test_servo_curve_limits_torque_in_direction_of_motion():
    w0 = 4.72
    env = ValidLocoEnv(config=cfg("easy", servo_curve={"stall_nm": 2.94, "noload_rad_s": w0}))
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    checked = over = 0
    for _ in range(300):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 12).astype(np.float32))
        qd = np.array([env.data.qvel[a] for a in env.joint_dof_adrs])
        tau = env.data.actuator_force[env.actuator_ids]
        same = np.sign(qd) == np.sign(tau)
        # the limit was computed from the velocity one substep earlier, so allow a margin
        assert (np.abs(tau) <= 2.94 + 1e-9).all()
        fast = same & (np.abs(qd) > 1.0)
        checked += int(fast.sum())
        over += int((np.abs(tau[fast]) > 2.94 * np.clip(1 - np.abs(qd[fast]) / w0, 0, 1) + 0.6).sum())
        if term or trunc:
            env.reset()
    assert checked > 50
    assert over / checked < 0.05, (over, checked)
    assert mujoco.get_mjcb_control() is None
    env.close()


def test_observation_noise_is_seeded_and_only_on_observation():
    c = cfg("easy", obs_noise={"joint_pos": 0.01, "joint_vel": 0.1, "gyro": 0.05, "gravity": 0.02})
    a, b = ValidLocoEnv(config=copy.deepcopy(c)), ValidLocoEnv(config=copy.deepcopy(c))
    clean = HERLocoEnv(config=build_env_config(yaml.safe_load(open(REPO / "config/pilot_gait_min_v2.yaml")), "easy", False))
    oa, _ = a.reset(seed=9)
    ob, _ = b.reset(seed=9)
    oc, _ = clean.reset(seed=9)
    assert np.array_equal(oa["observation"], ob["observation"])
    assert not np.array_equal(oa["observation"], oc["observation"])
    assert np.array_equal(oa["achieved_goal"], oc["achieved_goal"])
    assert np.allclose(oa["observation"][30:], oc["observation"][30:]), "previous action must stay noise-free"
    for e in (a, b, clean):
        e.close()


def test_training_uses_the_corrected_env_only_when_asked():
    from BipedRobot.fast.collector import env_class
    assert env_class({"validity": {"friction": True}}) is ValidLocoEnv
    assert env_class({}) is HERLocoEnv
    assert env_class({"validity": {}}) is HERLocoEnv
    c = yaml.safe_load(open(REPO / "config/valid_push100.yaml"))
    env_config = build_env_config(c, "full", False)
    env_config["validity"] = c["validity"]
    env = ValidLocoEnv(config=env_config)
    assert env.delay_range == (0.01, 0.05)
    assert env._push_substeps == 20 and env.motor_torque and env._noise["gyro"] == 0.05
    assert env.model.geom_priority[env.floor_geom_id] == 1
    env.close()


def test_servo_curve_cap_and_constant_bias():
    env = ValidLocoEnv(config=cfg("easy", servo_curve={"stall_nm": 6.53, "noload_rad_s": 5.31, "cap_nm": 2.94}, obs_bias={"joint_pos": 0.02, "gyro": 0.02, "gravity": 0.05}))
    env.reset(seed=1)
    class D:
        qvel = env.data.qvel.copy()
    qd = np.array([0.0, 1.0, 3.0, 5.0, 6.0, -1.0, -3.0, -5.0, -6.0, 0.0, 2.0, -2.0])
    D.qvel[env.joint_dof_adrs] = qd
    env._servo_curve(env.model, D)
    hi, lo = env.model.actuator_forcerange[env.actuator_ids, 1], env.model.actuator_forcerange[env.actuator_ids, 0]
    head = np.minimum(2.94, 6.53 * np.clip(1 - np.abs(qd) / 5.31, 0, 1))
    assert np.allclose(hi, np.where(qd > 0, head, 2.94)) and np.allclose(lo, -np.where(qd < 0, head, 2.94))
    b0 = copy.deepcopy(env._bias)
    assert (np.abs(b0["joint_pos"]) <= 0.02).all() and np.abs(b0["joint_pos"]).max() > 0.001
    a = np.zeros(12, np.float32)
    env.step(a)
    assert all(np.array_equal(b0[k], env._bias[k]) for k in b0), "bias must stay constant within an episode"
    env.reset()
    assert not np.array_equal(b0["joint_pos"], env._bias["joint_pos"]), "bias must change between episodes"
    env.close()
