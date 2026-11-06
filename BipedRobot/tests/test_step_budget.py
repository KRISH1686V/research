"""A fourth budget on steps (token_budget.costs ending in "steps"): its cost, shapes, physics, relabeling and mirroring. Run by path from the workspace root."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import decode_rate  # noqa: E402
from BipedRobot.fast.collector import NO_COST, _pad_cost, flatten_obs  # noqa: E402
from BipedRobot.fast.mirror import obs_maps  # noqa: E402
from BipedRobot.tests.test_budget_her import fill, make_env  # noqa: E402
from BipedRobot.tests.test_budget_window import brute_window  # noqa: E402

W = 20
STEPS = dict(costs=["flight", "slip", "torque", "steps"], rate_min=[0.002, 0.02, 0.05, 0.01], rate_max=[0.2, 0.2, 0.8, 0.2])
N = 4


def rollout(env, steps, n_u, seed=0, amp=0.6):
    rng = np.random.default_rng(seed)
    goal = lambda: {"desired_goal": np.concatenate(([0.25, 0.0, 0.0], rng.uniform(0.0, 0.5, n_u))).astype(np.float32)}
    o, _ = env.reset(seed=seed, options=goal())
    data = []
    for _ in range(steps):
        a = rng.uniform(-amp, amp, 12).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        data.append(dict(obs=flatten_obs(o), nxt=flatten_obs(o2), a=a, r=r, term=term, trunc=trunc, info=copy.deepcopy(info)))
        o = o2
        if term or trunc:
            o, _ = env.reset(options=goal())
    return data


def test_shapes_and_the_step_cost_counts_touchdowns():
    env = make_env(**STEPS)
    assert env.cost_names == ("flight", "slip", "torque", "steps") and env.n_cost == N
    o, _ = env.reset(seed=2, options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1, 0.3], np.float32)})
    assert o["desired_goal"].shape == (7,) and o["observation"].shape == (42 + N,)
    with pytest.raises(ValueError):
        env.reset(seed=2, options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1], np.float32)})
    rng = np.random.default_rng(0)
    seen = 0
    for _ in range(200):
        _, _, term, trunc, info = env.step(rng.uniform(-0.6, 0.6, 12).astype(np.float32))
        assert info["budget_cost"].shape == (N,)
        assert info["budget_cost"][3] == np.float32(np.sum(info["touchdown_events"]))
        seen += int(info["budget_cost"][3] > 0)
        if term or trunc:
            env.reset(options={"desired_goal": np.array([0.2, 0, 0, 1, 1, 1, 0.3], np.float32)})
    assert seen > 5, "the test must see touchdowns"
    env.close()


def test_three_costs_are_unchanged_by_a_fourth():
    a, b = make_env(), make_env(**STEPS)
    oa, _ = a.reset(seed=5, options={"desired_goal": np.array([0.2, 0, 0, 0.4, 0.4, 0.4], np.float32)})
    ob, _ = b.reset(seed=5, options={"desired_goal": np.array([0.2, 0, 0, 0.4, 0.4, 0.4, 1.0], np.float32)})
    rng = np.random.default_rng(1)
    for _ in range(150):
        act = rng.uniform(-0.4, 0.4, 12).astype(np.float32)
        oa, ra, ta, tra, ia = a.step(act)
        ob, rb, tb, trb, ib = b.step(act)
        assert np.array_equal(a.data.qpos, b.data.qpos)
        assert np.array_equal(ia["budget_cost"], ib["budget_cost"][:3])
        assert np.array_equal(oa["observation"][:-3], ob["observation"][:-N])
        if ta or tra:
            break
    a.close()
    b.close()


def test_relabeling_with_four_costs_matches_brute_force_exactly():
    budget = {**make_env.__globals__["BUDGET"], **STEPS, "window_steps": W, "alive_bonus": 0.2}
    env = make_env(**STEPS, window_steps=W, alive_bonus=0.2)
    data = rollout(env, 1500, N)
    rb = fill(env, data, budget=budget)
    assert rb.n_cost == N and rb.i_level == rb.obs_dim - N
    index, ep = {}, []
    for d in data:
        ep.append(d)
        if d["term"] or d["trunc"]:
            for t, e in enumerate(ep):
                index[np.concatenate((e["obs"][:3], e["obs"][6 + N:-N])).tobytes()] = (ep, t)
            ep = []
    checked = over = steps_binding = 0
    for _ in range(4):
        obs, _, rew, nxt, _, _ = [x.cpu().numpy() for x in rb.sample()]
        for j in range(rb.nb_virtual):
            ep, t = index[np.concatenate((obs[j, :3], obs[j, 6 + N:-N])).tobytes()]
            goal = obs[j, 3:6 + N].astype(np.float64)
            rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
            cap = rate * env.burst_steps
            deficits = brute_window(np.array([e["info"]["budget_cost"] for e in ep], dtype=np.float64), rate, W)
            assert np.allclose(obs[j, -N:], np.clip(1 - deficits[t] / cap, -1, 1), atol=2e-5)
            assert np.allclose(nxt[j, -N:], np.clip(1 - deficits[t + 1] / cap, -1, 1), atol=2e-5)
            e = ep[t]
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            is_over = bool((deficits[t + 1] > cap).any())
            want = base if e["info"]["fell"] else (0.0 if is_over else base) + env.alive_bonus
            assert abs(want - rew[j]) < 1e-5
            checked += 1
            over += int(is_over)
            steps_binding += int(deficits[t + 1][3] > cap[3])
    assert checked > 1000 and over > 10 and steps_binding > 0, (checked, over, steps_binding)
    env.close()


def test_mirror_keeps_all_four_budgets_in_place():
    perm, sign = obs_maps(5, N)
    dim = 3 + 3 + N + 5 * 42 + N
    assert sorted(perm.tolist()) == list(range(dim))
    for k in list(range(6, 6 + N)) + list(range(dim - N, dim)):
        assert perm[k] == k and sign[k] == 1.0


def test_collector_pads_up_to_six_costs():
    assert _pad_cost((0.5, 0.25, 0.125)) == (0.5, 0.25, 0.125, 0.0, 0.0, 0.0)
    assert _pad_cost((1.0, 2.0, 3.0, 4.0)) == (1.0, 2.0, 3.0, 4.0, 0.0, 0.0)
    assert len(NO_COST) == 6
    with pytest.raises(ValueError):
        _pad_cost((0.0,) * 7)
