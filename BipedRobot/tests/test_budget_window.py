"""Bucket with bounded memory (token_budget.window_steps): the guarantee, recovery, and exact relabeling against brute force. Run by path from the workspace root."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import decode_rate  # noqa: E402
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.replay import RewardConfig  # noqa: E402
from BipedRobot.tests.test_budget_her import BUDGET, DEVICE, env_obs_dim, fill, make_env, relabeled_truth, rollout  # noqa: E402

W = 20


def brute_window(costs, rate, window):
    S = np.vstack((np.zeros(np.shape(costs)[1]), np.cumsum(costs, axis=0)))
    F = S - rate * np.arange(len(S))[:, None]
    return np.array([F[t] - F[max(0, t - window):t + 1].min(0) for t in range(len(F))])


def test_no_overdraft_means_every_short_window_is_within_budget():
    rng = np.random.default_rng(0)
    r, B, kept = np.full(3, 0.1), 2.0, 0
    for _ in range(300):
        c = rng.uniform(0, 1, (300, 3)) * rng.integers(0, 2, (300, 1)) * rng.uniform(0, 0.6)
        if (brute_window(c, r, W) > B).any():
            continue
        kept += 1
        for a in range(0, 300, 7):
            for T in range(1, W + 1, 3):
                assert c[a:a + T].sum(0).max() <= r[0] * min(T, 300 - a) + B + 1e-9
    assert kept > 10, "the test must cover sequences that stay within budget"


def test_long_window_equals_the_plain_bucket():
    rng = np.random.default_rng(1)
    c = rng.uniform(0, 0.5, (200, 3))
    r = np.array([0.05, 0.2, 0.4])
    plain, d = [np.zeros(3)], np.zeros(3)
    for x in c:
        d = np.maximum(0.0, d + x - r)
        plain.append(d.copy())
    assert np.allclose(brute_window(c, r, 1000), np.array(plain))


def run_costs(env, costs):
    """Drive the env's bucket directly, so the recovery test does not depend on what the robot does."""
    env.set_budget(np.zeros(3, np.float32))
    out = []
    for c in costs:
        env._deficit = env._window_deficit(np.asarray(c, float))
        out.append(env._deficit.copy())
    return np.array(out)


def test_overdraft_recovers_after_the_window_and_old_bucket_does_not():
    env = make_env(window_steps=W)
    event = np.zeros((1, 3)) + 8.0
    quiet = np.zeros((W + 2, 3))
    d = run_costs(env, np.vstack((event, quiet)))
    cap = env._capacity
    assert (d[0] > cap).all(), "the event must overdraw every bucket"
    assert (d[W + 1] <= cap).all() and (d[W + 1] == 0).all(), "after W quiet steps the event has left the window"
    plain = np.zeros(3)
    for c in np.vstack((event, quiet)):
        plain = np.maximum(0.0, plain + c - env._rate)
    assert (plain > cap).all(), "the plain bucket is still overdrawn at the same step"
    env.close()


def test_small_overspend_recovers_sooner_than_a_big_one():
    env = make_env(window_steps=100)
    r = decode_rate(np.zeros(3), env.rate_min, env.rate_max)
    cap = r * env.burst_steps
    def recovery(extra):
        d = run_costs(env, np.vstack((np.full((1, 3), 0.0) + cap + extra * r + r, np.zeros((120, 3)))))
        return int(np.argmax((d <= cap).all(axis=1)))
    assert 0 < recovery(5) < recovery(50) < recovery(95)
    env.close()


def test_env_levels_and_reward_gate_match_brute_force():
    env = make_env(window_steps=W, alive_bonus=0.2)
    data = rollout(env, 900, amp=0.6)
    ep, over_seen, recovered = [], 0, 0
    for d in data:
        ep.append(d)
        if d["term"] or d["trunc"]:
            u = ep[0]["obs"][6:9].astype(np.float64)
            rate = decode_rate(u, env.rate_min, env.rate_max)
            cap = rate * env.burst_steps
            deficits = brute_window(np.array([e["info"]["budget_cost"] for e in ep], dtype=np.float64), rate, W)
            was_over = False
            for t, e in enumerate(ep):
                assert np.allclose(e["obs"][-3:], np.clip(1 - deficits[t] / cap, -1, 1), atol=1e-5)
                assert np.allclose(e["nxt"][-3:], np.clip(1 - deficits[t + 1] / cap, -1, 1), atol=1e-5)
                over = bool((deficits[t + 1] > cap).any())
                assert over == e["info"]["budget_overdrawn"]
                over_seen += int(over)
                recovered += int(was_over and not over)
                was_over = over
                if over and not e["info"]["fell"]:
                    assert e["r"] == env.alive_bonus
            ep = []
    assert over_seen > 0, "the test must cover overdrawn buckets"
    env.close()


def check_batch(env, rb, obs, rew, nxt, index, window):
    checked = over = 0
    for j in range(rb.nb_virtual):
        ep, t = index[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
        goal = obs[j, 3:9].astype(np.float64)
        rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
        cap = rate * env.burst_steps
        deficits = brute_window(np.array([e["info"]["budget_cost"] for e in ep], dtype=np.float64), rate, window)
        assert np.allclose(obs[j, -3:], np.clip(1 - deficits[t] / cap, -1, 1), atol=2e-5), (obs[j, -3:], 1 - deficits[t] / cap)
        assert np.allclose(nxt[j, -3:], np.clip(1 - deficits[t + 1] / cap, -1, 1), atol=2e-5)
        e = ep[t]
        base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
        is_over = bool((deficits[t + 1] > cap).any())
        want = base if e["info"]["fell"] else (0.0 if is_over else base) + env.alive_bonus
        assert abs(want - rew[j]) < 1e-5, (want, rew[j], is_over)
        checked += 1
        over += int(is_over)
    return checked, over


def test_relabeling_matches_brute_force_exactly():
    budget = {**BUDGET, "window_steps": W, "alive_bonus": 0.2}
    env = make_env(window_steps=W, alive_bonus=0.2)
    data = rollout(env, 1500, amp=0.6)
    rb = fill(env, data, budget=budget)
    index = relabeled_truth(env, rb, data)
    total = over_total = 0
    for _ in range(4):
        obs, _, rew, nxt, _, _ = rb.sample()
        c, o = check_batch(env, rb, *[x.cpu().numpy() for x in (obs, rew, nxt)], index, W)
        total += c
        over_total += o
    assert total > 1000 and over_total > 10, "the test must cover overdrawn relabels"
    env.close()


def test_wide_window_relabels_like_the_plain_bucket():
    """Same data and random numbers: with a window longer than any episode, both code paths give the same batch."""
    env = make_env()
    data = rollout(env, 1200, amp=0.6)
    plain = fill(env, data)
    wide = fill(env, data, budget={**BUDGET, "window_steps": 600})
    torch.manual_seed(3)
    a = plain.sample()
    torch.manual_seed(3)
    b = wide.sample()
    for x, y in zip(a[:5], b[:5]):
        assert torch.allclose(x, y, atol=2e-5)
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_capture_resamples_and_stays_exact():
    budget = {**BUDGET, "window_steps": W, "alive_bonus": 0.2}
    env = make_env(window_steps=W, alive_bonus=0.2)
    data = rollout(env, 1500, amp=0.6)
    rb = fill(env, data, batch=256, budget=budget)
    index = relabeled_truth(env, rb, data)
    B = rb.batch_size
    o_s, r_s, n_s = torch.zeros((B, rb.obs_dim), device=DEVICE), torch.zeros(B, device=DEVICE), torch.zeros((B, rb.obs_dim), device=DEVICE)
    rb.sample()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o, _, r, n, _, _ = rb.sample()
        o_s.copy_(o); r_s.copy_(r); n_s.copy_(n)
    g.replay()
    first = o_s.clone()
    g.replay()
    assert not torch.equal(first, o_s)
    for _ in range(3):
        g.replay()
        c, _ = check_batch(env, rb, *[x.cpu().numpy() for x in (o_s, r_s, n_s)], index, W)
        assert c == rb.nb_virtual
    env.close()
