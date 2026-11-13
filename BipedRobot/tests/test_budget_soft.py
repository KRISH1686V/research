"""token_budget.overdraft "soft" and cost "jshake", added 11 Nov 2026. Soft: while overdrawn the task reward is min(r, r * exp(-sum(excess / capacity)))
plus the alive bonus, in the env and in relabeled replay samples alike. jshake: mean over joints of the squared second difference of the true joint angles.
Uses the brute-force helpers of test_budget_her.py. Run by path from the workspace root."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_budget_her as H  # noqa: E402
from BipedRobot.fast.budget_env import decode_rate  # noqa: E402
from BipedRobot.fast.collector import env_class  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

SOFT = {**H.BUDGET, "overdraft": "soft", "alive_bonus": 0.2}


def soft_want(base, fell, deficit, cap, alive):
    if fell:
        return base
    f = float(np.exp(-(np.maximum(0.0, deficit - cap) / cap).sum()))
    return min(base, base * f) + alive


def episodes(data):
    ep = []
    for d in data:
        ep.append(d)
        if d["term"] or d["trunc"]:
            yield ep
            ep = []


def test_soft_env_reward_matches_brute_force():
    env = H.make_env(overdraft="soft", alive_bonus=0.2)
    data = H.rollout(env, 900, amp=0.6)
    over = scaled = 0
    for ep in episodes(data):
        goal = ep[0]["obs"][3:9].astype(np.float64)
        rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
        cap = rate * env.burst_steps
        deficits = H.brute_deficit([e["info"]["budget_cost"].astype(np.float64) for e in ep], rate)
        for t, e in enumerate(ep):
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            want = soft_want(base, e["info"]["fell"], deficits[t + 1], cap, env.alive_bonus)
            assert abs(want - e["r"]) < 1e-5, (want, e["r"])
            is_over = bool((deficits[t + 1] > cap).any())
            assert is_over == e["info"]["budget_overdrawn"]
            over += int(is_over)
            scaled += int(is_over and not e["info"]["fell"] and base > 0 and abs(e["r"] - env.alive_bonus - base) > 1e-6 and e["r"] > env.alive_bonus)
    assert over > 20 and scaled > 5, "the test must cover overdrawn steps that are scaled, not zeroed"
    env.close()


def test_soft_relabeling_matches_brute_force():
    env = H.make_env(overdraft="soft", alive_bonus=0.2)
    data = H.rollout(env, 1500, amp=0.6)
    rb = H.fill(env, data, budget=SOFT)
    index = H.relabeled_truth(env, rb, data)
    checked = over = 0
    for _ in range(3):
        obs, _, rew, nxt, done, _ = [x.cpu().numpy() if hasattr(x, "cpu") else x for x in rb.sample()]
        for j in range(rb.nb_virtual):
            ep, t = index[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
            goal = obs[j, 3:9].astype(np.float64)
            rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
            cap = rate * env.burst_steps
            deficits = H.brute_deficit([e["info"]["budget_cost"].astype(np.float64) for e in ep], rate)
            e = ep[t]
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            want = soft_want(base, e["info"]["fell"], deficits[t + 1], cap, env.alive_bonus)
            assert abs(want - rew[j]) < 2e-5, (want, rew[j])
            checked += 1
            over += int((deficits[t + 1] > cap).any())
    assert checked > 500 and over > 10
    env.close()


def test_soft_never_pays_for_overdraft_and_hard_is_the_default():
    for base in (-1.0, -0.1, 0.0, 0.3, 2.0):
        for excess in (0.0, 0.5, 3.0):
            f = np.exp(-excess)
            assert min(base, base * f) <= base + 1e-12  # overdrawn is never better than within budget
    env = H.make_env()
    assert not env.soft_overdraft
    env.close()
    with pytest.raises(ValueError):
        H.make_env(overdraft="maybe")


def test_jshake_is_the_joint_second_difference():
    env = H.make_env(costs=["flight", "slip", "jshake"])
    env.reset(seed=4, options={"desired_goal": np.array([0.1, 0, 0, 1, 1, 1], np.float32)})
    q = [env.data.qpos[env.joint_qpos_adrs].astype(np.float64).copy()] * 2
    rng = np.random.default_rng(1)
    for _ in range(20):
        _, _, term, trunc, info = env.step(rng.uniform(-0.5, 0.5, 12).astype(np.float32))
        q.append(env.data.qpos[env.joint_qpos_adrs].astype(np.float64).copy())
        want = float(np.mean((q[-1] - 2 * q[-2] + q[-3]) ** 2))
        assert info["budget_cost"][2] == pytest.approx(want, rel=1e-5, abs=1e-12)
        if term or trunc:
            break
    assert max(float(np.mean((q[k] - 2 * q[k - 1] + q[k - 2]) ** 2)) for k in range(2, len(q))) > 0
    env.close()


def test_round17_config_loads():
    repo = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((repo / "config/v2_deploy_budget_alpha30_r17j_fit.yaml").read_text())
    ec = build_env_config(c, "medium", False)
    for k in ("validity", "token_budget"):
        ec[k] = c[k]
    env = env_class(ec)(config=ec, render_mode=None)
    assert env.cost_names == ("flight", "slip", "jshake") and env.soft_overdraft
    env.close()


@pytest.mark.skipif(not H.torch.cuda.is_available(), reason="needs CUDA")
def test_soft_graph_capture_stays_exact():
    torch = H.torch
    env = H.make_env(overdraft="soft", alive_bonus=0.2)
    data = H.rollout(env, 1500, amp=0.6)
    rb = H.fill(env, data, batch=256, budget=SOFT)
    B = rb.batch_size
    o_s, r_s = torch.zeros((B, rb.obs_dim), device=H.DEVICE), torch.zeros(B, device=H.DEVICE)
    rb.sample()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o, _, r, n, dn, _ = rb.sample()
        o_s.copy_(o); r_s.copy_(r)
    index = H.relabeled_truth(env, rb, data)
    for _ in range(3):
        g.replay()
        obs, rew = o_s.cpu().numpy(), r_s.cpu().numpy()
        for j in range(rb.nb_virtual):
            ep, t = index[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
            goal = obs[j, 3:9].astype(np.float64)
            rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
            cap = rate * env.burst_steps
            deficits = H.brute_deficit([e["info"]["budget_cost"].astype(np.float64) for e in ep], rate)
            e = ep[t]
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            assert abs(soft_want(base, e["info"]["fell"], deficits[t + 1], cap, env.alive_bonus) - rew[j]) < 2e-5
    env.close()


def test_u_low_draws_tighter_budgets_in_the_env():
    env = H.make_env(u_low=-0.3)
    us = []
    for s in range(60):
        o, _ = env.reset(seed=s)
        us.append(env._u.copy())
    us = np.array(us)
    assert us.min() < -0.05 and us.min() >= -0.3 - 1e-6
    assert np.all(decode_rate(us.astype(np.float64), env.rate_min, env.rate_max) > 0)
    env.close()
    env = H.make_env(u_low=-0.3, curriculum=True)
    env.set_difficulty(0.0)
    env.reset(seed=1)
    assert np.all((env._u >= 1.0 - 1e-6) | (env._u == 1.0))  # curriculum at 0: loose only
    env.set_difficulty(1.0)
    lows = [env.reset(seed=s)[0] is not None and env._u.min() for s in range(60)]
    assert min(lows) < -0.05
    env.close()
    with pytest.raises(ValueError):
        H.make_env(u_low=0.5)


def test_u_low_relabeling_stays_exact_below_zero():
    budget = {**H.BUDGET, "u_low": -0.3}
    env = H.make_env(u_low=-0.3)
    data = H.rollout(env, 1500, amp=0.6)
    rb = H.fill(env, data, budget=budget)
    index = H.relabeled_truth(env, rb, data)
    neg = total = 0
    for _ in range(3):
        obs, _, rew, nxt, done, _ = rb.sample()
        obs_np = obs.cpu().numpy()
        neg += int((obs_np[:rb.nb_virtual, 6:9] < 0).sum())
        c, _ = H.check_batch(env, rb, data, *[x.cpu().numpy() for x in (obs, rew, nxt, done)], index)
        total += c
    assert total > 500 and neg > 50, "prior draws must reach below u = 0"
    env.close()


def test_target_entropy_from_config_and_default():
    from BipedRobot.fast.learner import SacLearner
    env = H.make_env()
    data = H.rollout(env, 300, amp=0.6)
    rb = H.fill(env, data)
    dim = H.env_obs_dim(env)
    assert SacLearner(rb, dim, 12, H.DEVICE).target_entropy == -12.0
    assert SacLearner(rb, dim, 12, H.DEVICE, target_entropy=-24.0).target_entropy == -24.0
    env.close()


def test_capacity_floor_in_env_and_relabeling_match_brute_force():
    """capacity_floor (12 Nov 2026): B = max(rate x burst, floor), in the env and in relabeled samples."""
    floor = [0.5, 0.0, 0.0]
    env = H.make_env(capacity_floor=floor, alive_bonus=0.2)
    assert np.allclose(env.capacity_floor, floor)
    data = H.rollout(env, 1500, amp=0.6)
    for ep in episodes(data):
        goal = ep[0]["obs"][3:9].astype(np.float64)
        rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
        cap = np.maximum(rate * env.burst_steps, floor)
        deficits = H.brute_deficit([e["info"]["budget_cost"].astype(np.float64) for e in ep], rate)
        for t, e in enumerate(ep):
            assert bool((deficits[t + 1] > cap).any()) == e["info"]["budget_overdrawn"]
            assert np.allclose(e["nxt"][-3:], np.clip(1 - deficits[t + 1] / cap, -1, 1), atol=1e-5)
    rb = H.fill(env, data, budget={**H.BUDGET, "capacity_floor": floor, "alive_bonus": 0.2})
    index = H.relabeled_truth(env, rb, data)
    checked = 0
    for _ in range(2):
        obs, _, rew, nxt, done, _ = [x.cpu().numpy() for x in rb.sample()]
        for j in range(rb.nb_virtual):
            ep, t = index[np.concatenate((obs[j, :3], obs[j, 9:-3])).tobytes()]
            goal = obs[j, 3:9].astype(np.float64)
            rate = decode_rate(goal[3:], env.rate_min, env.rate_max)
            cap = np.maximum(rate * env.burst_steps, floor)
            deficits = H.brute_deficit([e["info"]["budget_cost"].astype(np.float64) for e in ep], rate)
            assert np.allclose(obs[j, -3:], np.clip(1 - deficits[t] / cap, -1, 1), atol=2e-5)
            e = ep[t]
            base = float(env.compute_reward(e["nxt"][:3], goal[:3].astype(np.float32), e["info"]))
            over = (deficits[t + 1] > cap).any()
            want = base if e["info"]["fell"] else (0.0 if over else base) + env.alive_bonus
            assert abs(want - rew[j]) < 2e-5
            checked += 1
    assert checked > 300
    env.close()
    with pytest.raises(ValueError):
        H.make_env(capacity_floor=[0.5, 0.0])


def test_settle_steps_zero_the_first_costs_only():
    env = H.make_env(settle_steps=3)
    env.reset(seed=2, options={"desired_goal": np.array([0.1, 0, 0, 1, 1, 1], np.float32)})
    rng = np.random.default_rng(3)
    costs = []
    for _ in range(12):
        _, _, term, trunc, info = env.step(rng.uniform(-0.8, 0.8, 12).astype(np.float32))
        costs.append(info["budget_cost"].copy())
        if term or trunc:
            break
    assert np.all(np.array(costs[:3]) == 0.0)
    assert np.array(costs[3:]).sum() > 0.0
    env.close()


def test_p_corner_draws_one_tight_cost_in_env_and_replay():
    env = H.make_env(u_low=-0.3, p_corner=1.0, p_unconstrained=0.0)
    for s in range(30):
        env.reset(seed=s)
        u = env._u
        assert (u < 1.0).sum() == 1 and u.min() <= 0.25 + 1e-6 and u.min() >= -0.3 - 1e-6
    env.close()
    budget = {**H.BUDGET, "u_low": -0.3, "p_corner": 0.5}
    env = H.make_env(u_low=-0.3)
    data = H.rollout(env, 1500, amp=0.6)
    rb = H.fill(env, data, budget=budget)
    index = H.relabeled_truth(env, rb, data)
    corners = total = 0
    for _ in range(3):
        obs, _, rew, nxt, done, _ = rb.sample()
        u = obs.cpu().numpy()[:rb.nb_virtual, 6:9]
        corners += int(((u < 1.0 - 1e-6).sum(axis=1) == 1).sum())
        c, _ = H.check_batch(env, rb, data, *[x.cpu().numpy() for x in (obs, rew, nxt, done)], index)
        total += c
    assert total > 500 and corners > 10, "corner priors must appear (prior draws are 25% of relabels) and relabeling must stay exact"
    env.close()
    with pytest.raises(ValueError):
        H.make_env(p_corner=1.5)
