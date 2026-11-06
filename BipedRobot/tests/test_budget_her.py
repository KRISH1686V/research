"""Budget-as-goal HER: bucket math, exact relabeling against brute force, and env consistency. Run by path from the workspace root."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import BudgetLocoEnv, decode_rate  # noqa: E402
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.replay import RewardConfig  # noqa: E402
from BipedRobot.fast.valid_env import ValidLocoEnv  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUDGET = dict(rate_min=[0.002, 0.02, 0.05], rate_max=[0.2, 0.2, 0.8], burst_steps=100, p_unconstrained=0.2, p_keep=0.25, p_achieved=0.5)


def env_config(**budget):
    c = yaml.safe_load(open(REPO / "config/valid_nogait.yaml"))
    ec = build_env_config(c, "full", False)
    ec["validity"] = c["validity"]
    ec["token_budget"] = {**BUDGET, **budget}
    return ec


def brute_deficit(costs, rate):
    d = np.zeros(3)
    out = [d.copy()]
    for c in costs:
        d = np.maximum(0.0, d + c - rate)
        out.append(d.copy())
    return np.array(out)


def test_window_guarantee_holds_for_any_cost_sequence():
    rng = np.random.default_rng(0)
    r, B = 0.1, 2.0
    for _ in range(200):
        c = rng.uniform(0, 1, 300) * rng.integers(0, 2, 300)
        b, ok = B, True
        for i in range(300):
            nb = min(B, b + r - c[i])
            if nb < 0:
                ok = False
                break
            b = nb
        if not ok:
            continue
        for a in range(0, 300, 13):
            for e in range(a + 1, 301, 17):
                assert c[a:e].sum() <= r * (e - a) + B + 1e-9


def rollout(env, steps, seed=0, amp=1.0):
    rng = np.random.default_rng(seed)
    goal = lambda: {"desired_goal": np.concatenate(([0.25, 0.0, 0.0], rng.uniform(0.0, 0.5, 3))).astype(np.float32)}  # tight, so buckets overdraw
    o, _ = env.reset(seed=seed, options=goal())
    data, eps = [], []
    for _ in range(steps):
        a = (rng.uniform(-amp, amp, 12)).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        data.append(dict(obs=flatten_obs(o), nxt=flatten_obs(o2), a=a, r=r, term=term, trunc=trunc, info=copy.deepcopy(info)))
        o = o2
        if term or trunc:
            eps.append(len(data))
            o, _ = env.reset(options=goal())
    return data


def fill(env, data, capacity=4000, batch=512, budget=BUDGET):
    rb = BudgetHerReplay(capacity, 1, batch, 2, env_obs_dim(env), 12, RewardConfig.from_env(env), DEVICE, budget)
    i = rb.i_rew
    for d in data:
        row = np.zeros((1, rb.width), dtype=np.float32)
        info = d["info"]
        row[0, :rb.obs_dim], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = d["obs"], d["nxt"], d["a"]
        row[0, i:] = (d["r"], d["term"], info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"], *info["budget_cost"])
        rb.add(torch.as_tensor(row, device=DEVICE), np.array([d["term"] or d["trunc"]]))
    return rb


def env_obs_dim(env):
    return sum(int(np.prod(s.shape)) for s in env.observation_space.spaces.values())


def make_env(**budget):
    env = BudgetLocoEnv(config=env_config(**budget))
    env.reward_mode = "track"
    return env


def test_env_physics_unchanged_and_shapes():
    c = env_config()
    plain = copy.deepcopy(c)
    plain.pop("token_budget")
    a, b = ValidLocoEnv(config=plain), make_env()
    a.reward_mode = "track"
    oa, _ = a.reset(seed=5, options={"desired_goal": np.array([0.2, 0, 0], np.float32)})
    ob, _ = b.reset(seed=5, options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1], np.float32)})
    assert ob["desired_goal"].shape == (6,) and ob["observation"].shape == (45,) and ob["achieved_goal"].shape == (3,)
    assert np.allclose(ob["observation"][-3:], 1.0), "buckets start full"
    rng = np.random.default_rng(1)
    for _ in range(150):
        act = rng.uniform(-0.3, 0.3, 12).astype(np.float32)
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, ib = b.step(act)
        assert np.array_equal(a.data.qpos, b.data.qpos)
        assert ta == tb and tra == trb
        assert np.array_equal(oa["observation"], ob["observation"][:-3]) and np.array_equal(oa["achieved_goal"], ob["achieved_goal"])
        if not ib["budget_overdrawn"]:
            assert ra == rb
        if ta or tra:
            break
    a.close()
    b.close()


def test_env_levels_match_brute_force_and_reward_gate():
    env = make_env()
    data = rollout(env, 700, amp=0.6)
    ep, seen_overdrawn = [], 0
    for d in data:
        ep.append(d)
        if d["term"] or d["trunc"]:
            u = ep[0]["obs"][6:9].astype(np.float64)
            rate = decode_rate(u, env.rate_min, env.rate_max)
            cap = rate * env.burst_steps
            deficits = brute_deficit([e["info"]["budget_cost"].astype(np.float64) for e in ep], rate)
            for t, e in enumerate(ep):
                assert np.allclose(e["obs"][-3:], np.clip(1 - deficits[t] / cap, -1, 1), atol=1e-5)
                assert np.allclose(e["nxt"][-3:], np.clip(1 - deficits[t + 1] / cap, -1, 1), atol=1e-5)
                over = (deficits[t + 1] > cap).any()
                assert over == e["info"]["budget_overdrawn"]
                seen_overdrawn += int(over)
                if over and not e["info"]["fell"]:
                    assert e["r"] == env.alive_bonus
            ep = []
    assert seen_overdrawn > 0, "the test must cover overdrawn buckets"
    env.close()


def relabeled_truth(env, rb, data):
    """Index every transition by its noise-free fingerprint so a sampled row can be traced back to its episode and step."""
    index, ep = {}, []
    for d in data:
        ep.append(d)
        if d["term"] or d["trunc"]:
            for t, e in enumerate(ep):
                index[np.concatenate((e["obs"][:3], e["obs"][9:-3])).tobytes()] = (ep, t)
            ep = []
    return index


def check_batch(env, rb, data, obs, rew, nxt, done, index):
    V = rb.nb_virtual
    checked = over = 0
    for j in range(V):
        ep, t = index[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
        goal = obs[j, 3:9].astype(np.float64)
        rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
        cap = rate * env.burst_steps
        costs = [e["info"]["budget_cost"].astype(np.float64) for e in ep]
        deficits = brute_deficit(costs, rate)
        assert np.allclose(obs[j, -3:], np.clip(1 - deficits[t] / cap, -1, 1), atol=2e-5), (obs[j, -3:], 1 - deficits[t] / cap)
        assert np.allclose(nxt[j, -3:], np.clip(1 - deficits[t + 1] / cap, -1, 1), atol=2e-5)
        assert np.array_equal(obs[j, 3:9], nxt[j, 3:9])
        e = ep[t]
        base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
        is_over = (deficits[t + 1] > cap).any()
        want = base if e["info"]["fell"] else (0.0 if is_over else base) + env.alive_bonus
        assert abs(want - rew[j]) < 1e-5, (want, rew[j], is_over)
        assert any(np.array_equal(f["nxt"][:3], goal[:3].astype(np.float32)) for f in ep[t:]), "velocity goal must come from the future of the episode"
        checked += 1
        over += int(is_over)
    return checked, over


def test_relabeling_matches_brute_force_exactly():
    env = make_env()
    data = rollout(env, 1500, amp=0.6)
    rb = fill(env, data)
    index = relabeled_truth(env, rb, data)
    total = over_total = 0
    for _ in range(4):
        obs, _, rew, nxt, done, _ = rb.sample()
        c, o = check_batch(env, rb, data, *[x.cpu().numpy() for x in (obs, rew, nxt, done)], index)
        total += c
        over_total += o
    assert total > 1000 and over_total > 10, "the test must cover overdrawn relabels"
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_capture_resamples_and_stays_exact():
    env = make_env()
    data = rollout(env, 1500, amp=0.6)
    rb = fill(env, data, batch=256)
    index = relabeled_truth(env, rb, data)
    B = rb.batch_size
    o_s, r_s, n_s, d_s = torch.zeros((B, rb.obs_dim), device=DEVICE), torch.zeros(B, device=DEVICE), torch.zeros((B, rb.obs_dim), device=DEVICE), torch.zeros(B, device=DEVICE)
    rb.sample()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o, _, r, n, dn, _ = rb.sample()
        o_s.copy_(o); r_s.copy_(r); n_s.copy_(n); d_s.copy_(dn)
    g.replay()
    first = o_s.clone()
    g.replay()
    assert not torch.equal(first, o_s)
    for _ in range(3):
        g.replay()
        c, _ = check_batch(env, rb, data, *[x.cpu().numpy() for x in (o_s, r_s, n_s, d_s)], index)
        assert c == rb.nb_virtual
    env.close()


def test_real_samples_keep_stored_budget_and_reward():
    env = make_env()
    data = rollout(env, 1200, amp=0.6)
    rb = fill(env, data)
    obs, _, rew, nxt, done, _ = [x if not torch.is_tensor(x) else x.cpu().numpy() for x in rb.sample()]
    V = rb.nb_virtual
    stored = {d["obs"].tobytes(): d for d in data}
    for j in range(V, rb.batch_size):
        d = stored[obs[j].tobytes()]
        assert np.array_equal(nxt[j], d["nxt"]) and abs(rew[j] - d["r"]) < 1e-6 and done[j] == float(d["term"])
    env.close()


def test_alive_bonus_is_exact_under_relabeling():
    env = make_env(alive_bonus=0.2)
    data = rollout(env, 1500, amp=0.6)
    rb = fill(env, data, budget={**BUDGET, "alive_bonus": 0.2})
    index = relabeled_truth(env, rb, data)
    total = over = 0
    for _ in range(3):
        obs, _, rew, nxt, done, _ = rb.sample()
        c, o = check_batch(env, rb, data, *[x.cpu().numpy() for x in (obs, rew, nxt, done)], index)
        total += c
        over += o
    assert total > 500 and over > 10
    env.close()


def test_budget_curriculum_starts_loose():
    env = make_env(curriculum=True, p_unconstrained=0.0)
    env.set_difficulty(0.0)
    for s in range(5):
        o, _ = env.reset(seed=s)
        assert np.allclose(o["desired_goal"][3:], 1.0)
    env.set_difficulty(0.3)
    us = np.array([env.reset(seed=s)[0]["desired_goal"][3:] for s in range(30)])
    assert us.min() >= 0.7 - 1e-6 and us.max() <= 1.0 and us.std() > 0.01
    data = rollout(env, 800, amp=0.6)
    rb = fill(env, data, budget={**BUDGET, "p_keep": 0.0, "p_achieved": 0.0})
    rb.set_budget_floor(0.8)
    obs = rb.sample()[0].cpu().numpy()
    assert obs[:rb.nb_virtual, 6:9].min() >= 0.8 - 1e-5
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_budget_with_history_mirror_and_clean_critic_state():
    from BipedRobot.fast.learner import SacLearner
    from BipedRobot.fast.mirror import action_perm, obs_maps, priv_maps
    c = yaml.safe_load(open(REPO / "config/v2_budget_survive.yaml"))
    ec = build_env_config(c, "full", False)
    ec["validity"], ec["token_budget"] = c["validity"], c["token_budget"]
    env = BudgetLocoEnv(config=ec)
    env.reward_mode = "track"
    assert env.xml_path.endswith("robot_v2_train.xml")
    o, _ = env.reset(seed=1)
    assert o["observation"].shape == (213,) and o["desired_goal"].shape == (6,)
    assert np.allclose(o["observation"][-3:], 1.0)
    o2, *_ = env.step(np.zeros(12, np.float32))
    assert o2["observation"].shape == (213,) and np.array_equal(o2["observation"][:168], o["observation"][42:210])
    dim, pd = 222, 46
    op, os_ = obs_maps(5, 3)
    pp, ps = priv_maps()
    rb = BudgetHerReplay(3000, 1, 256, 2, dim, 12, RewardConfig.from_env(env), DEVICE, c["token_budget"], priv_dim=pd,
                         mirror={"obs_perm": op, "obs_sign": os_, "act_perm": action_perm(), "priv_perm": pp, "priv_sign": ps})
    rng = np.random.default_rng(0)
    o, _ = env.reset(seed=2)
    pv = env.privileged()
    i = rb.i_rew
    for _ in range(1200):
        a = rng.uniform(-0.5, 0.5, 12).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        row = np.zeros((1, rb.width), dtype=np.float32)
        row[0, :dim], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = flatten_obs(o), flatten_obs(o2), a
        row[0, i:i + 10] = (r, term, info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"], *info["budget_cost"])
        row[0, rb.i_priv:rb.i_next_priv], row[0, rb.i_next_priv:] = pv, env.privileged()
        rb.add(torch.as_tensor(row, device=DEVICE), np.array([term or trunc]))
        o, pv = o2, env.privileged()
        if term or trunc:
            o, _ = env.reset()
            pv = env.privileged()
    out = rb.sample()
    assert len(out) == 8 and out[6].shape == (256, pd)
    torch.manual_seed(2)
    learner = SacLearner(rb, dim, 12, DEVICE, mode="graph", layer_norm=True, avg_q=True, priv_dim=pd, ema_tau=0.002)
    for _ in range(50):
        learner.update()
    torch.cuda.synchronize()
    assert np.isfinite(list(learner.read_stats().values())).all()
    env.close()
