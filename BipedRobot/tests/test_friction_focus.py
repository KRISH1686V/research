"""Extra training episodes at the lab floor's friction, and that no test can pick them up by accident. Run by path from the workspace root."""
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import BudgetLocoEnv  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GOAL = np.array([0.2, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)
FOCUS = {"share": 0.5, "range": [0.3, 0.4]}


def make(focus=None, on=False, difficulty=1.0):
    c = yaml.safe_load(open(REPO / "config/v2_deploy_budget_alpha30.yaml"))
    ec = build_env_config(c, "full", False)
    ec["validity"], ec["token_budget"] = copy.deepcopy(c["validity"]), c["token_budget"]
    if focus:
        ec["validity"]["friction_focus"] = focus
    env = BudgetLocoEnv(config=ec)
    env.reward_mode = "track"
    env.use_friction_focus = on
    env.set_difficulty(difficulty)
    return env


def draws(env, n=300, seed=11):
    rows = []
    env.reset(seed=seed, options={"desired_goal": GOAL})
    for _ in range(n):
        env.reset(options={"desired_goal": GOAL})
        rows.append((float(env.model.geom_friction[env.floor_geom_id, 0]), float(env.model.body_mass[env.torso_body_id]), float(env._active_delay)))
    env.close()
    return np.array(rows)


def test_set_in_config_but_not_switched_on_changes_nothing():
    """What every test tool does: it loads the run's config, which names the option, and builds its own env."""
    assert np.array_equal(draws(make()), draws(make(FOCUS)))


def test_share_of_episodes_lands_in_range_and_other_draws_stay():
    plain, focused = draws(make()), draws(make(FOCUS, on=True))
    assert np.array_equal(plain[:, 1:], focused[:, 1:]), "mass and delay must not change"
    changed = plain[:, 0] != focused[:, 0]
    assert 0.40 < changed.mean() < 0.60
    assert ((focused[changed, 0] >= 0.3) & (focused[changed, 0] <= 0.4)).all()
    assert plain[:, 0].min() >= 0.25 and plain[:, 0].max() <= 0.8


def test_scales_with_difficulty():
    plain, focused = draws(make(difficulty=0.0)), draws(make(FOCUS, on=True, difficulty=0.0))
    assert np.array_equal(plain, focused), "no extra episodes while training is still at its easiest"
    plain, focused = draws(make(difficulty=0.5)), draws(make(FOCUS, on=True, difficulty=0.5))
    assert 0.15 < (plain[:, 0] != focused[:, 0]).mean() < 0.35


def test_critic_sees_the_friction_actually_used():
    env = make(FOCUS, on=True)
    env.reset(seed=3, options={"desired_goal": GOAL})
    for _ in range(40):
        env.reset(options={"desired_goal": GOAL})
        assert abs(env.privileged()[-4] - env.model.geom_friction[env.floor_geom_id, 0]) < 1e-6
    env.close()
