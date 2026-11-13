"""validity.servo_planner (fast/valid_env.py): the real servo's own move planner, fitted on 6 Nov 2026 to the robot's step and sine tests."""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HW = ROOT / "BipedRobot/diagnostics/audit_2026-10-31/audit/v2/hardware_tests"
sys.path.insert(0, str(HW))
CONFIG = ROOT / "BipedRobot/config/v2_deploy_budget_alpha30_heeltoe_ref.yaml"
FIT = {"accel_rad_s2": 9.0, "accel_range": [7.5, 10.5], "dead_s": [0.0, 0.02]}


def make(planner, realism="full", dr=False):
    from BipedRobot.fast.collector import env_class
    from BipedRobot.train_her_biped import build_env_config
    c = yaml.safe_load(CONFIG.read_text())
    c["validity"].pop("gait_ref", None)
    c["validity"].pop("clock_hz", None)
    if planner:
        c["validity"]["servo_planner"] = planner
    ec = build_env_config(c, realism, False)
    ec["validity"] = copy.deepcopy(c["validity"])
    ec["push_enabled"] = False
    ec["domain_randomization"]["enabled"] = dr
    return env_class(ec)(config=ec, render_mode=None)


def test_off_unless_named():
    env = make(None)
    assert not env._planner_on
    assert env.delay_range == (0.0, 0.05)


def test_dead_time_replaces_delay_range_and_easy_turns_it_off():
    assert make(FIT, "full").delay_range == (0.0, 0.02)
    med = make(FIT, "medium")
    assert med._planner_on and med.delay_range == (0.01, 0.01)
    assert not make(FIT, "easy")._planner_on


def test_setpoint_obeys_accel_and_stops_at_target():
    """Held in the air, one 10 deg command on the knee: the setpoint's acceleration never exceeds the cap and it settles on the target."""
    env = make(FIT)
    env.action_filter_enabled, env.enable_lag = False, False  # one clean step into the planner, as the robot test sends it
    env.reset(seed=0)
    env.model.dof_armature[:6] = 1e6
    a = np.zeros(env.num_joints, np.float32)
    k = 3
    span = env.jnt_max[k] - env.default_qpos[k]
    a[k] = math.radians(10) / (env.action_range_fraction * span)
    ps, vs = [], []
    import mujoco
    real = mujoco.mj_step

    def logged(m, d):
        real(m, d)
        ps.append(env._plan_p[k])
        vs.append(env._plan_v[k])
    mujoco.mj_step = logged
    try:
        for _ in range(40):
            env.step(a)
    finally:
        mujoco.mj_step = real
    target = env._plan_target[k]  # the command after the config's backlash band, which comes before the planner
    assert abs(target - (env.default_qpos[k] + math.radians(10) - env.backlash_deadband)) < 1e-6
    acc = np.abs(np.diff(vs)) / env.sim_dt
    # the cap holds everywhere except the one substep where it lands on the target (speed at most 3 a h before it) and stops
    assert np.sum(acc > env._plan_a * (1 + 1e-9)) <= 1 and acc.max() <= 3 * env._plan_a * (1 + 1e-9)
    assert abs(ps[-1] - target) < 1e-9 and vs[-1] == 0.0
    assert max(ps) <= target + 1e-9, "the planner brakes to stop at the target, so it never passes it"
    # a 10 deg move at 9 rad/s^2 takes 2 sqrt(d / a) = 0.24 s
    reach = next(i for i, p in enumerate(ps) if abs(p - target) < 1e-6) * env.sim_dt
    assert 0.22 < reach < 0.30


def test_accel_drawn_per_episode_in_range():
    env = make(FIT, "full", dr=True)
    seen = set()
    for s in range(20):
        env.reset(seed=s)
        assert 7.5 <= env._plan_a <= 10.5
        seen.add(round(env._plan_a, 6))
    assert len(seen) > 10
    med = make(FIT, "medium", dr=True)
    med.reset(seed=1)
    assert med._plan_a == 9.0


def test_fitted_model_matches_the_robot():
    """With the fitted servo the simulated step and sine tests land near the robot's 6 Nov numbers (5 deg, hanging, fast settings)."""
    from sim_servo import REAL, make_env, run_all
    env, delay = make_env(str(CONFIG), {"accel_rad_s2": 9.0, "dead_s": [0.0, 0.0]}, 0.0, 0.0, 0.2)
    summary, _ = run_all(env, delay, joints=["r_knee", "r_hip_x"])
    assert abs(summary["step"]["t63_ms"] - REAL["step"]["t63_ms"]) < 20
    assert abs(summary["step"]["t90_ms"] - REAL["step"]["t90_ms"]) < 20
    for f in ("1.0", "2.0", "3.0"):
        assert abs(summary["track"][f]["gain"] - REAL["track"][f]["gain"]) < 0.08, f
        assert abs(summary["track"][f]["lag_ms"] - REAL["track"][f]["lag_ms"]) < 25, f
