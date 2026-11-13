"""The hand-written walking pattern (fast/gait_ref.py) and the policy-on-top option (validity.gait_ref). Run by path from the workspace root."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
from BipedRobot.fast import gait_ref as gr  # noqa: E402
from BipedRobot.fast.budget_env import BudgetLocoEnv  # noqa: E402
from BipedRobot.fast.mirror import action_perm  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

CFG = "v2_deploy_budget_alpha30_human_ref_sole"
GOAL = np.array([0.2, 0, 0, 1, 1, 1, 1, 1, 1], np.float32)


def env_for(name=CFG, realism="easy", **validity):
    c = yaml.safe_load(open(REPO / f"config/{name}.yaml"))
    ec = build_env_config(c, realism, False)
    ec["validity"] = {**c["validity"], **validity}
    ec["token_budget"] = c["token_budget"]
    ec.setdefault("shaping", {})["reset_velocity"] = False
    env = BudgetLocoEnv(config=ec)
    env.reward_mode = "track"
    return env


def test_saved_table_is_what_the_code_builds_and_fits_the_robot():
    z = np.load(REPO / "config/gait_ref_v2_2hz.npz")
    env = env_for(gait_ref=None)
    env.reset(seed=0)
    params = {k: float(z[k]) for k in gr.DEFAULTS if k in z.files}
    r = gr.build_table(env, **params)
    assert np.allclose(r["table"], z["table"], atol=1e-6), "config/gait_ref_v2_2hz.npz must be rebuilt from fast/gait_ref.py"
    assert r["ik_pos_err_max_m"] < 1e-3 and r["stance_rot_err_max_deg"] < 1.0 and r["stance_level_spread_mm"] < 0.1
    assert r["swing_clearance_max_mm"] > 20.0
    t = z["table"]
    assert np.all(t >= env.jnt_min - 1e-6) and np.all(t <= env.jnt_max + 1e-6)
    assert r["joint_speed_max_rad_s"] < 5.31, "within the servo's no-load speed"
    env.close()


def test_left_leg_is_the_right_leg_half_a_cycle_later_so_mirroring_is_exact():
    t = np.load(REPO / "config/gait_ref_v2_2hz.npz")["table"]
    n = len(t)
    perm = action_perm()
    for k in range(n):
        assert np.array_equal(t[k][perm], t[(k + n // 2) % n]), k


def test_target_is_pattern_plus_correction_at_the_delayed_time():
    env = env_for()
    env.reset(seed=3, options={"desired_goal": GOAL})
    z = np.load(REPO / "config/gait_ref_v2_2hz.npz")["table"]
    half = (env.jnt_max - env.jnt_min) / 2.0
    rng = np.random.default_rng(0)
    for n in range(120):
        a = rng.uniform(-1, 1, 12).astype(np.float32)
        env.step(a)
        want_t = n * env.dt - (env._active_delay if env.enable_delay else 0.0)
        assert abs(env._ref_t - want_t) < 1e-9, n
        ramp = np.clip(want_t / 1.0, 0.0, 1.0)
        pose = env.default_qpos + ramp * (gr.lookup(z, want_t * 1.0) - env.default_qpos)
        assert np.allclose(env.gait_ref_pose(want_t), pose, atol=1e-9)
        b = rng.uniform(-1, 1, 12)
        got = env._action_to_target(b)
        want = np.clip(pose + b * 0.25 * half, np.maximum(env.jnt_min, env.ctrl_min), np.minimum(env.jnt_max, env.ctrl_max))
        assert np.allclose(got, want, atol=1e-5)
    assert np.allclose(env.gait_ref_pose(-0.5), env.default_qpos) and np.allclose(env.gait_ref_pose(3.0), gr.lookup(z, 3.0))
    env.close()


def test_needs_the_clock_at_the_table_period_and_is_off_by_default():
    with pytest.raises(ValueError, match="gait_ref"):
        env_for(clock_hz=2.0)
    plain = env_for(gait_ref=None)
    assert plain._ref_table is None
    plain.reset(seed=1, options={"desired_goal": GOAL})
    a = np.full(12, 0.3)
    from BipedRobot.envs.her_loco_env import HERLocoEnv
    assert np.array_equal(plain._action_to_target(a), HERLocoEnv._action_to_target(plain, a))
    plain.close()


def test_pattern_alone_steps_in_rhythm():
    env = env_for()
    env.reset(seed=0, options={"desired_goal": GOAL})
    right = []
    for t in range(250):
        _, _, term, _, _ = env.step(np.zeros(12, np.float32))
        right.append(env._feet_contact()[0] > 0.5)
        if term:
            break
    assert t > 200, "the pattern alone stays up for a few seconds on the easy floor"
    r = np.array(right[50:])
    assert 0.2 < r.mean() < 0.9, "the right foot is down for part of each cycle, not all of it"
    env.close()


def test_heel_to_toe_table_rolls_over_heel_and_toe_and_stays_exact():
    z = np.load(REPO / "config/gait_ref_v2_heeltoe.npz")
    env = env_for(gait_ref=None)
    env.reset(seed=0)
    r = gr.build_table(env, **{k: float(z[k]) for k in gr.DEFAULTS if k in z.files})
    assert np.allclose(r["table"], z["table"], atol=1e-6), "config/gait_ref_v2_heeltoe.npz must be rebuilt from fast/gait_ref.py"
    assert float(z["speed_m_s"]) == 0.2 and float(z["heel_deg"]) > 0 and float(z["toe_deg"]) > 0
    assert r["ik_pos_err_max_m"] < 1e-3 and r["stance_rot_err_max_deg"] < 1.0
    assert r["stance_level_spread_mm"] < 0.1, "through stance some point of the sole stays on the floor: heel, then sole, then toe"
    assert r["joint_speed_max_rad_s"] < 5.31
    pitch, n = r["swing_pitch_deg"], len(z["table"])
    D = float(z["duty"])
    first = int(np.ceil(((0.25 - D / 2) % 1.0) * n)) % n
    last = int(np.floor((0.25 + D / 2) * n)) % n
    assert pitch[first] < -5.0, "lands heel first, toe up (swing_pitch_deg is positive heel up)"
    assert pitch[last] > 10.0, "leaves on the toe, heel up"
    mid = int(0.25 * n)
    assert abs(pitch[mid]) < 0.5, "flat in mid-stance"
    perm = action_perm()
    for k in range(n):
        assert np.array_equal(z["table"][k][perm], z["table"][(k + n // 2) % n])
    env.close()
