"""The slow yaw-rate input. Run by path from the workspace root."""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import BudgetLocoEnv  # noqa: E402
from BipedRobot.fast.mirror import obs_maps  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GOAL = np.array([0.2, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)


def make(name, realism="full"):
    c = yaml.safe_load(open(REPO / f"config/{name}.yaml"))
    ec = build_env_config(c, realism, False)
    ec["validity"], ec["token_budget"] = c["validity"], c["token_budget"]
    ec.setdefault("shaping", {})["reset_velocity"] = False
    env = BudgetLocoEnv(config=ec)
    env.reward_mode = "track"
    return env


def test_input_is_the_mean_of_the_gyro_the_policy_senses():
    env = make("v2_deploy_budget_alpha30_prog10_slowyaw")
    o, _ = env.reset(seed=1, options={"desired_goal": GOAL})
    assert o["observation"].shape == (214,), "5 frames of 42, slow yaw, 3 balances"
    i = 2 * 12 + 2
    scale = env.obs_scale["base_ang_vel"]
    sensed = [float(o["observation"][4 * 42 + i]) / scale]
    assert abs(o["observation"][210] - 5 * sensed[0]) < 1e-5
    rng = np.random.default_rng(0)
    for _ in range(60):
        o, _, term, _, _ = env.step(rng.uniform(-0.1, 0.1, 12).astype(np.float32))
        sensed.append(float(o["observation"][4 * 42 + i]) / scale)
        assert abs(o["observation"][210] - 5 * np.mean(sensed[-25:])) < 1e-4, "0.5 s is 25 control steps"
        if term:
            break
    o, _ = env.reset(seed=2, options={"desired_goal": GOAL})
    assert abs(o["observation"][210] - 5 * o["observation"][4 * 42 + i] / scale) < 1e-5, "a new episode forgets the old turning"
    env.close()


def test_off_by_default():
    env = make("v2_deploy_budget_alpha30")
    o, _ = env.reset(seed=1, options={"desired_goal": GOAL})
    assert o["observation"].shape == (213,) and env.slow_yaw_s == 0.0 and env._slow_yaw is None
    env.close()


def test_mirror_flips_slow_yaw_and_keeps_balances_last():
    perm, sign = obs_maps(5, 3, False, True)
    dim = 3 + 6 + 210 + 1 + 3
    assert len(perm) == dim and sorted(perm) == list(range(dim))
    assert perm[219] == 219 and sign[219] == -1.0
    assert list(perm[220:]) == [220, 221, 222] and list(sign[220:]) == [1.0, 1.0, 1.0]
    both, both_sign = obs_maps(5, 3, True, True)
    assert list(both_sign[219:222]) == [-1.0, -1.0, -1.0] and list(both[222:]) == [222, 223, 224]
