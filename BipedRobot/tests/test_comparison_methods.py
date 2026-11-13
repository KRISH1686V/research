"""The E-B comparison methods: episode budgets, dual ascent, plain SAC, budgets switched off. Run by path from the workspace root."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import decode_rate  # noqa: E402
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.learner import SacLearner  # noqa: E402
from BipedRobot.fast.replay import RewardConfig  # noqa: E402
from BipedRobot.tests.test_budget_her import BUDGET, DEVICE, env_obs_dim, fill, make_env, relabeled_truth, rollout  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EPISODE = {**BUDGET, "mode": "episode", "alive_bonus": 0.2}
LAGRANGE = {**BUDGET, "rate_min": [10.0, 10.0, 10.0], "rate_max": [20.0, 20.0, 20.0], "p_keep": 1.0, "p_achieved": 0.0, "p_unconstrained": 1.0, "alive_bonus": 0.2,
            "lagrange": {"limit": [0.03, 0.06, 0.05], "lr": 0.1, "init": 0.0, "max": 100.0}}


def test_episode_budget_in_the_env_is_one_allowance_with_no_refill():
    env = make_env(mode="episode", alive_bonus=0.2)
    assert env.episode_budget
    data = rollout(env, 600, amp=0.6)
    ep, over_seen = [], 0
    for d in data:
        ep.append(d)
        if d["term"] or d["trunc"]:
            u = ep[0]["obs"][6:9].astype(np.float64)
            rate = decode_rate(u, env.rate_min, env.rate_max)
            cap = rate * env.max_episode_steps
            spent = np.vstack((np.zeros(3), np.cumsum([e["info"]["budget_cost"] for e in ep], axis=0)))
            for t, e in enumerate(ep):
                assert np.allclose(e["obs"][-3:], np.clip(1 - spent[t] / cap, -1, 1), atol=1e-5)
                over = bool((spent[t + 1] > cap).any())
                assert over == e["info"]["budget_overdrawn"]
                over_seen += int(over)
                if over and not e["info"]["fell"]:
                    assert e["r"] == env.alive_bonus
            ep = []
    env.close()


def test_episode_mode_refuses_a_window():
    with pytest.raises(ValueError, match="no window"):
        make_env(mode="episode", window_steps=100)


def test_episode_budget_relabels_exactly():
    env = make_env(mode="episode", alive_bonus=0.2)
    data = rollout(env, 1500, amp=0.6)
    rb = fill(env, data, budget=EPISODE)
    index = relabeled_truth(env, rb, data)
    total = 0
    for _ in range(3):
        obs, _, rew, nxt, _, _ = [x.cpu().numpy() for x in rb.sample()]
        for j in range(rb.nb_virtual):
            ep, t = index[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
            goal = obs[j, 3:9].astype(np.float64)
            rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
            cap = rate * env.max_episode_steps
            spent = np.vstack((np.zeros(3), np.cumsum([e["info"]["budget_cost"] for e in ep], axis=0)))
            assert np.allclose(obs[j, -3:], np.clip(1 - spent[t] / cap, -1, 1), atol=2e-5)
            assert np.allclose(nxt[j, -3:], np.clip(1 - spent[t + 1] / cap, -1, 1), atol=2e-5)
            e = ep[t]
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            want = base if e["info"]["fell"] else (0.0 if (spent[t + 1] > cap).any() else base) + env.alive_bonus
            assert abs(want - rew[j]) < 1e-5
            total += 1
    assert total > 500
    env.close()


def test_dual_step_rises_above_the_limit_falls_below_and_stays_in_range():
    env = make_env()
    rb = BudgetHerReplay(1000, 1, 64, 2, env_obs_dim(env), 12, RewardConfig.from_env(env), DEVICE, LAGRANGE)
    lim = np.array(LAGRANGE["lagrange"]["limit"])
    assert np.allclose(rb.dual_step(lim * 2.0), 0.1), "twice the limit raises lambda by the step size, for every cost alike"
    assert np.allclose(rb.dual_step(lim), 0.1), "at the limit nothing changes"
    assert np.allclose(rb.dual_step(lim * 0.5), 0.05)
    assert np.allclose(rb.lam.cpu().numpy(), 0.05)
    for _ in range(5):
        lam = rb.dual_step(np.zeros(3))
    assert (lam == 0).all(), "lambda never goes negative"
    for _ in range(100):
        lam = rb.dual_step(lim * 1001.0)
    assert (lam == 100.0).all(), "and stays below the cap"
    env.close()


def test_dual_penalty_is_exact_on_every_sample_and_budgets_are_off():
    env = make_env(rate_min=[10.0, 10.0, 10.0], rate_max=[20.0, 20.0, 20.0], p_unconstrained=1.0, alive_bonus=0.2)
    data = rollout(env, 1200, amp=0.6)
    data = [d for d in data]
    rb = fill(env, data, budget=LAGRANGE)
    rb.dual_step(np.array(LAGRANGE["lagrange"]["limit"]) * 1.5)
    lam = rb._lam_np
    assert (lam > 0).all()
    rb_off = fill(env, data, budget={k: v for k, v in LAGRANGE.items() if k != "lagrange"})
    torch.manual_seed(5)
    obs, _, rew, _, _, _ = [x.cpu().numpy() for x in rb.sample()]
    torch.manual_seed(5)
    obs_off, _, rew_off, _, _, _ = [x.cpu().numpy() for x in rb_off.sample()]
    assert np.array_equal(obs, obs_off), "only the reward differs"
    stored = {np.concatenate((d["obs"][:3], d["obs"][9:-3])).tobytes(): d for d in data}
    for j in range(len(rew)):
        d = stored[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
        assert abs((rew_off[j] - float(lam @ d["info"]["budget_cost"])) - rew[j]) < 1e-4
        assert np.array_equal(obs[j, 6:9], d["obs"][6:9]), "budgets are never relabeled"
    assert np.allclose(obs[:, -3:], 1.0), "and the buckets stay full"
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_capture_sees_lambda_change_between_replays():
    env = make_env(rate_min=[10.0, 10.0, 10.0], rate_max=[20.0, 20.0, 20.0], p_unconstrained=1.0, alive_bonus=0.2)
    data = rollout(env, 1200, amp=0.6)
    rb = fill(env, data, batch=256, budget=LAGRANGE)
    B = rb.batch_size
    o_s, r_s = torch.zeros((B, rb.obs_dim), device=DEVICE), torch.zeros(B, device=DEVICE)
    rb.sample()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o, _, r, _, _, _ = rb.sample()
        o_s.copy_(o)
        r_s.copy_(r)
    costs = {np.concatenate((d["obs"][:3], d["obs"][9:-3])).tobytes(): d for d in data}
    for level in (0.0, 5.0):
        rb.lam.fill_(level)
        rb._lam_np[:] = level
        g.replay()
        torch.cuda.synchronize()
        obs, rew = o_s.cpu().numpy(), r_s.cpu().numpy()
        for j in range(rb.nb_virtual, B):  # real transitions keep their stored reward, then pay the penalty
            d = costs[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
            assert abs(rew[j] - (d["r"] - level * float(np.sum(d["info"]["budget_cost"])))) < 1e-4
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_plain_sac_never_relabels_and_trains_in_a_graph():
    env = make_env(rate_min=[10.0, 10.0, 10.0], rate_max=[20.0, 20.0, 20.0], p_unconstrained=1.0, alive_bonus=0.2)
    data = rollout(env, 1000, amp=0.6)
    dim = env_obs_dim(env)
    budget = {**BUDGET, "rate_min": [10.0] * 3, "rate_max": [20.0] * 3, "p_keep": 1.0, "p_achieved": 0.0, "alive_bonus": 0.2}
    rb = BudgetHerReplay(4000, 1, 256, 0, dim, 12, RewardConfig.from_env(env), DEVICE, budget)
    assert rb.nb_virtual == 0
    i = rb.i_rew
    for d in data:
        row = np.zeros((1, rb.width), dtype=np.float32)
        info = d["info"]
        row[0, :rb.obs_dim], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = d["obs"], d["nxt"], d["a"]
        row[0, i:] = (d["r"], d["term"], info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"], *info["budget_cost"])
        rb.add(torch.as_tensor(row, device=DEVICE), np.array([d["term"] or d["trunc"]]))
    obs, _, rew, nxt, done, _ = [x.cpu().numpy() for x in rb.sample()]
    stored = {d["obs"].tobytes(): d for d in data}
    for j in range(len(rew)):
        d = stored[obs[j].tobytes()]
        assert np.array_equal(nxt[j], d["nxt"]) and abs(rew[j] - d["r"]) < 1e-6
    torch.manual_seed(1)
    learner = SacLearner(rb, dim, 12, DEVICE, mode="graph", layer_norm=True, avg_q=True)
    for _ in range(20):
        learner.update()
    torch.cuda.synchronize()
    assert np.isfinite(list(learner.read_stats().values())).all()
    env.close()


def test_comparison_configs_differ_only_where_they_should():
    def load(name):
        return yaml.safe_load(open(REPO / f"config/{name}.yaml"))
    cmd, plain, lag = load("v2_deploy_cmd"), load("v2_deploy_plain"), load("v2_deploy_lagrange")
    budget, episode = load("v2_deploy_budget"), load("v2_deploy_episode")
    assert plain["her"]["n_sampled_goal"] == 0 and cmd["her"]["n_sampled_goal"] == 2
    assert {k: v for k, v in plain.items() if k != "her"} == {k: v for k, v in cmd.items() if k != "her"}
    assert lag["token_budget"].pop("lagrange")["limit"] == [0.0316, 0.0632, 0.0447]
    assert lag == cmd
    assert episode["token_budget"].pop("mode") == "episode" and budget["token_budget"].pop("window_steps") == 100
    assert episode == budget
    r = decode_rate(np.full(3, 0.5), np.array(budget["token_budget"]["rate_min"]), np.array(budget["token_budget"]["rate_max"]))
    assert np.allclose(r, [0.0316, 0.0632, 0.0447], atol=1e-4), "the dual-ascent limits are the budget at u = 0.5"


def test_normalized_penalty_pays_per_unit_of_each_limit():
    """lagrange.normalize (11 Nov 2026): the penalty is lambda * cost / limit, so a tiny cost such as joint shake is not ignored."""
    env = make_env(rate_min=[10.0, 10.0, 10.0], rate_max=[20.0, 20.0, 20.0], p_unconstrained=1.0, alive_bonus=0.2)
    data = rollout(env, 1200, amp=0.6)
    norm = {**LAGRANGE, "lagrange": {**LAGRANGE["lagrange"], "normalize": True, "lr": 0.0, "init": 0.3}}
    rb = fill(env, data, budget=norm)
    rb.dual_step(np.array(LAGRANGE["lagrange"]["limit"]) * 1.5)  # lr 0: a fixed penalty, lambda stays at init
    assert np.allclose(rb._lam_np, 0.3)
    rb_off = fill(env, data, budget={k: v for k, v in LAGRANGE.items() if k != "lagrange"})
    torch.manual_seed(7)
    obs, _, rew, _, _, _ = [x.cpu().numpy() for x in rb.sample()]
    torch.manual_seed(7)
    _, _, rew_off, _, _, _ = [x.cpu().numpy() for x in rb_off.sample()]
    lim = np.array(LAGRANGE["lagrange"]["limit"])
    stored = {np.concatenate((d["obs"][:3], d["obs"][9:-3])).tobytes(): d for d in data}
    for j in range(len(rew)):
        d = stored[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
        assert abs((rew_off[j] - float((0.3 / lim) @ d["info"]["budget_cost"])) - rew[j]) < 1e-4
    rb_raw = fill(env, data, budget=LAGRANGE)
    assert torch.allclose(rb_raw.lag_scale, torch.ones(3, device=rb_raw.lag_scale.device)), "default unchanged: raw costs"
    env.close()
