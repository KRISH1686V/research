"""GaitCostEnv must measure what evaluation_suite_v2 measures and must not change the dynamics. Run by path from the workspace root."""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.envs import her_loco_env as hle  # noqa: E402
from BipedRobot.envs.her_loco_env import HERLocoEnv  # noqa: E402
from BipedRobot.evaluation_suite_v2 import EvalSettings, Thresholds, _foot_vertices, _substep_state, load_policy, make_eval_env, run_episode  # noqa: E402
from BipedRobot.fast.gait_costs import GaitCostEnv  # noqa: E402
from BipedRobot.fast.replay import RewardConfig, relabel_reward  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "checkpoints/fast_min_regress_20261101_seed4/seed_4/sb3_her_final.zip"
RESOLVED = REPO / "runs/fast_min_regress_20261101_seed4/seed_4/resolved_config.yaml"


def env_config(**shaping):
    config = yaml.safe_load(open(REPO / "config/pilot_gait_min_v2.yaml"))
    config["shaping"].update(shaping)
    return build_env_config(config, "easy", False)


def test_substep_measurements_equal_the_evaluators():
    env = GaitCostEnv(config=env_config())
    verts = _foot_vertices(env)
    got = []
    def observer(fn, slip, nonfoot):
        s = _substep_state(env, verts)
        got.append((fn.copy(), slip.copy(), nonfoot, s["fn"], s["slip"], s["nonfoot"]))
    env.substep_observer = observer
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    for _ in range(600):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 12).astype(np.float32))
        if term or trunc:
            env.reset()
    assert len(got) == 2400
    assert sum(g[0].sum() > 0.5 for g in got) > 500 and sum(g[1].sum() > 1e-3 for g in got) > 100
    for fn, slip, nonfoot, fn2, slip2, nonfoot2 in got:
        assert np.allclose(fn, fn2, atol=1e-12)
        assert np.allclose(slip, slip2, atol=1e-9)
        assert abs(nonfoot - nonfoot2) < 1e-12
    env.close()


@pytest.mark.skipif(not CKPT.exists(), reason="needs the fast regression checkpoint")
def test_episode_costs_match_evaluator_metrics_on_a_walking_policy():
    from BipedRobot.evaluation_suite_v2 import load_env_config
    cfg = load_env_config(RESOLVED)
    policy = load_policy(str(CKPT), cfg)
    settings = EvalSettings(realism="easy")
    command = np.array([0.2, 0.0, 0.0], np.float32)
    plain = make_eval_env(cfg, settings)
    want = run_episode(policy, plain, command, 5003, settings, Thresholds())
    plain.close()
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["realism"] = "easy"
    eval_cfg["push_enabled"] = False
    eval_cfg.setdefault("shaping", {})["reset_velocity"] = False
    env = GaitCostEnv(config=eval_cfg)
    env.push_enabled = False
    obs, _ = env.reset(seed=5003, options={"desired_goal": command})
    rows, term, trunc, steps = [], False, False, 0
    while not (term or trunc) and steps < settings.max_steps:
        obs, _, term, trunc, info = env.step(np.asarray(policy(obs), np.float32))
        rows.append((info["cost_flight"], info["cost_slip"], info["cost_slip_n"], info["cost_torque_sat"]))
        steps += 1
    env.close()
    assert steps == want["steps"] and want["survived"]
    r = np.array(rows)
    steady = r[50:]  # steady_start_s 1.0 at 50 Hz
    assert abs(steady[:, 0].mean() - want["flight_frac"]) < 1e-9
    slip = (steady[:, 1] * steady[:, 2]).sum() / steady[:, 2].sum()
    assert abs(slip - want["stance_slip_mps_mean"]) < 1e-9
    assert abs(r[:, 3].mean() - want["torque_sat_frac"]) < 1e-9
    assert want["flight_frac"] > 0 and want["stance_slip_mps_mean"] > 0.01, "the check must cover real flight and slip"


def test_dynamics_and_rewards_unchanged_at_zero_penalty():
    cfg = env_config()
    a, b = HERLocoEnv(config=copy.deepcopy(cfg)), GaitCostEnv(config=copy.deepcopy(cfg))
    a.reward_mode = b.reward_mode = "track"
    original = hle.mujoco
    rng = np.random.default_rng(3)
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    for _ in range(400):
        act = rng.uniform(-0.3, 0.3, 12).astype(np.float32)
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
        assert hle.mujoco is original
        assert ra == rb and ta == tb and tra == trb
        assert all(np.array_equal(oa[k], ob[k]) for k in oa)
        if ta or tra:
            oa, _ = a.reset(seed=12)
            ob, _ = b.reset(seed=12)
    a.close()
    b.close()


def test_torque_penalty_is_exact_and_relabelling_stays_consistent():
    w = 0.2
    a = GaitCostEnv(config=env_config())
    b = GaitCostEnv(config=env_config(torque_penalty_weight=w))
    for e in (a, b):
        e.reward_mode = "track"
        e.set_smoothness_ramp(0.5)
    rc = RewardConfig.from_env(b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(4)
    a.reset(seed=1)
    b.reset(seed=1)
    seen = 0
    for _ in range(300):
        act = rng.uniform(-0.3, 0.3, 12).astype(np.float32)
        oa, ra, ta, tra, ia = a.step(act)
        ob, rb, tb, trb, ib = b.step(act)
        assert abs(ib["r_smooth"] - (ia["r_smooth"] - 0.5 * w * ib["torque_rms"] ** 2)) < 1e-9
        if not ib["fell"]:
            assert abs((ra - rb) - 0.5 * w * ib["torque_rms"] ** 2) < 1e-5
            seen += ib["torque_rms"] > 0.05
        t = lambda x: torch.as_tensor(np.asarray([x], dtype=np.float32), device=device)
        got = relabel_reward(rc, torch.as_tensor(ob["achieved_goal"][None], device=device), torch.as_tensor(ob["desired_goal"][None], device=device),
                             t(ib["r_smooth"]), t(ib["r_gait_move"]), t(ib["r_gait_still"]), t(ib["r_flight"]), t(ib["fell"])).item()
        assert abs(got - rb) < 1e-5
        if ta or tra:
            a.reset(seed=2)
            b.reset(seed=2)
    assert seen > 50
    a.close()
    b.close()


def test_overhead_is_reported():
    cfg = env_config()
    rates = {}
    for cls in (HERLocoEnv, GaitCostEnv):
        env = cls(config=copy.deepcopy(cfg))
        env.reset(seed=0)
        act = np.zeros(12, np.float32)
        t0 = time.process_time()
        for _ in range(400):
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                env.reset(seed=0)
        rates[cls.__name__] = 400 / (time.process_time() - t0)
        env.close()
    print("control steps per CPU second:", {k: round(v) for k, v in rates.items()})
    assert rates["GaitCostEnv"] > 0.3 * rates["HERLocoEnv"]
