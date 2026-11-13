"""token_budget cost "shake", added 10 Nov 2026: the action's mean square second difference, counted against its own budget. CPU only."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BipedRobot.fast.collector import env_class  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NEW = REPO / "config/v2_deploy_budget_alpha30_r15s_fit.yaml"
OLD = REPO / "config/v2_deploy_budget_alpha30_r11v_fit.yaml"


def make(path=NEW, realism="medium"):
    c = yaml.safe_load(path.read_text())
    ec = build_env_config(c, realism, False)
    for k in ("validity", "token_budget"):
        if k in c:
            ec[k] = c[k]
    return env_class(ec)(config=ec, render_mode=None)


def test_config_names_shake_and_keeps_three_costs():
    env = make()
    assert env.cost_names == ("flight", "slip", "shake") and env.n_cost == 3
    assert env.observation_space["desired_goal"].shape == (6,)  # same goal size as before: the robot runner's three --budget numbers still fit
    env.close()


def test_shake_cost_is_the_action_second_difference():
    env = make()
    env.reset(seed=1, options={"desired_goal": np.array([0.1, 0, 0, 1, 1, 1], np.float32)})
    acts = [np.zeros(12), np.full(12, 0.2), np.full(12, -0.1), np.full(12, 0.3)]
    alpha = env.action_filter_alpha if env.action_filter_enabled else 1.0  # the cost is on the filtered action, the command the servos get
    f = env._action_filter_state.astype(np.float64).copy()
    hist = [env._prev_prev_action.astype(np.float64).copy(), env._prev_action.astype(np.float64).copy()]
    for a in acts:
        _, _, term, trunc, info = env.step(a.astype(np.float32))
        f = alpha * a + (1.0 - alpha) * f
        want = float(np.mean((f - 2 * hist[-1] + hist[-2]) ** 2))
        assert info["budget_cost"][2] == pytest.approx(want, rel=1e-4, abs=1e-8)
        hist.append(f.copy())
    env.close()


def test_constant_action_costs_nothing_and_jitter_overdraws_a_tight_budget():
    env = make()
    env.reset(seed=2, options={"desired_goal": np.array([0.1, 0, 0, 1, 1, 0], np.float32)})
    for _ in range(3):
        _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert info["budget_cost"][2] == 0.0
    over = False
    for k in range(60):
        _, _, term, trunc, info = env.step(np.full(12, 0.3 if k % 2 else -0.3, np.float32))
        over = over or info["budget_overdrawn"]
        if term or trunc:
            break
    assert over  # filtered, still about 0.03 a step against 0.0008: the bucket empties fast
    env.close()


def test_old_configs_unchanged():
    env = make(OLD)
    assert env.cost_names == ("flight", "slip", "torque")
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.zeros(12, np.float32))
    assert info["budget_cost"][2] == pytest.approx(info["cost_torque_sat"])
    env.close()
