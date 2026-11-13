"""The low-step budget (token_budget.costs with "lowsteps"), five budgets at once, and named extra budgets in the test tools. Run by path from the workspace root."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics/audit_2026-10-31/audit/v2"))
from BipedRobot.fast.collector import NO_COST, _pad_cost  # noqa: E402
from BipedRobot.tests.test_budget_her import make_env  # noqa: E402
from budget_eval import BUDGETS, pad_budget  # noqa: E402

LIFT = dict(costs=["flight", "slip", "torque", "lowsteps"], rate_min=[0.002, 0.02, 0.05, 0.002], rate_max=[0.2, 0.2, 0.8, 0.2], low_step_mm=20)
BOTH = dict(costs=["flight", "slip", "torque", "steps", "lowsteps"], rate_min=[0.002, 0.02, 0.05, 0.01, 0.002], rate_max=[0.2, 0.2, 0.8, 0.2, 0.2])


def test_low_steps_match_an_independent_count():
    env = make_env(**LIFT)
    goal = np.array([0.2, 0, 0, 1, 1, 1, 1], np.float32)
    env.reset(seed=4, options={"desired_goal": goal})
    rng = np.random.default_rng(3)
    peak, low_total, high_total = np.zeros(2), 0, 0
    for _ in range(600):
        _, _, term, trunc, info = env.step(rng.uniform(-0.7, 0.7, 12).astype(np.float32))
        clear, contact, td = env.foot_clearance(), env._feet_contact() > 0.5, info["touchdown_events"] > 0.5
        want = 0
        for k in range(2):
            if td[k]:
                want += int(peak[k] < 0.020)
                high_total += int(peak[k] >= 0.020)
                peak[k] = 0.0
            elif not contact[k]:
                peak[k] = max(peak[k], clear[k])
        assert info["budget_cost"][3] == np.float32(want)
        low_total += want
        if term or trunc:
            env.reset(options={"desired_goal": goal})
            peak[:] = 0.0
    assert low_total > 5, "the test must see low steps"
    env.close()


def test_clearance_is_zero_ish_when_standing():
    env = make_env(**LIFT)
    env.reset(seed=1, options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1, 1], np.float32)})
    for _ in range(40):
        env.step(np.zeros(12, np.float32))
    clear = env.foot_clearance()
    assert np.all(clear < 0.003) and np.all(clear > -0.005), clear
    env.close()


def test_five_budgets_together():
    env = make_env(**BOTH)
    o, _ = env.reset(seed=2, options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1, 0.3, 0.0], np.float32)})
    assert env.cost_names == tuple(BOTH["costs"]) and o["desired_goal"].shape == (8,) and o["observation"].shape == (42 + 5,)
    _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert info["budget_cost"].shape == (5,)
    assert len(NO_COST) == 6 and _pad_cost(info["budget_cost"]) == tuple(float(x) for x in info["budget_cost"]) + (0.0,)
    env.close()


def test_named_budgets_follow_each_env_s_cost_order():
    three, lift, both = make_env(), make_env(**LIFT), make_env(**BOTH)
    assert pad_budget(BUDGETS["all_mid"], three) == (0.5, 0.5, 0.5)
    assert pad_budget(BUDGETS["all_mid"], lift) == (0.5, 0.5, 0.5, 1.0)
    assert pad_budget(BUDGETS["mid_lift"], lift) == (0.5, 0.5, 0.5, 0.0)
    assert pad_budget(BUDGETS["mid_steps2"], lift) is None, "lift-only env has no step budget"
    assert pad_budget(BUDGETS["mid_steps2_lift"], both) == (0.5, 0.5, 0.5, 0.2314, 0.0)
    assert pad_budget(BUDGETS["mid_lift"], three) is None
    for e in (three, lift, both):
        e.close()


SIX = dict(costs=["flight", "slip", "torque", "steps", "lowsteps", "tiltstance"], rate_min=[0.002, 0.02, 0.05, 0.01, 0.002, 0.02], rate_max=[0.2, 0.2, 0.8, 0.2, 0.2, 2.0])


def test_tilted_stance_counts_tilted_feet_on_the_floor():
    env = make_env(**SIX)
    goal = np.array([0.2, 0, 0, 1, 1, 1, 1, 1, 1], np.float32)
    env.reset(seed=6, options={"desired_goal": goal})
    for _ in range(30):
        _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert info["budget_cost"][5] == 0.0, "standing still on flat feet is not tilted"
    rng = np.random.default_rng(7)
    seen = 0
    for _ in range(400):
        _, _, term, trunc, info = env.step(rng.uniform(-0.7, 0.7, 12).astype(np.float32))
        want = float(np.sum((env._feet_contact() > 0.5) & (env.sole_tilt() > np.radians(8.0))))
        assert info["budget_cost"][5] == np.float32(want)
        seen += int(want > 0)
        if term or trunc:
            env.reset(options={"desired_goal": goal})
    assert seen > 5
    assert len(NO_COST) == 6 and pad_budget(BUDGETS["mid_human"], env) == (0.5, 0.5, 0.5, 0.2314, 0.0, 0.4375)
    env.close()
