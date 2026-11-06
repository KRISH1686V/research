"""Checks that the fast learner computes what SB3's SAC+HER computes. Run by path from the workspace root."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer  # noqa: E402

from BipedRobot.envs.her_loco_env import HERLocoEnv  # noqa: E402
from BipedRobot.evaluation_suite_v2 import load_policy  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.export_sb3 import Sb3Exporter  # noqa: E402
from BipedRobot.fast.learner import SacLearner  # noqa: E402
from BipedRobot.fast.nets import Actor  # noqa: E402
from BipedRobot.fast.replay import GpuHerReplay, RewardConfig, relabel_reward  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_config(name="pilot_gait_min_v2.yaml", **shaping):
    config = yaml.safe_load(open(REPO / "config" / name))
    config["shaping"].update(shaping)
    return build_env_config(config, "easy", False), config


def rollout(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    o, _ = env.reset(seed=seed)
    data = []
    for _ in range(steps):
        a = rng.uniform(-1, 1, 12).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        data.append(dict(ach=o2["achieved_goal"].copy(), goal=o2["desired_goal"].copy(), r=r, info=copy.deepcopy(info), term=term, trunc=trunc,
                         obs=flatten_obs(o), nxt=flatten_obs(o2), a=a))
        o = o2
        if term or trunc:
            o, _ = env.reset()
    return data


@pytest.mark.parametrize("shaping", [{"gait_shaping": True}, {"gait_shaping": False}, {"gait_shaping": True, "action_rate_weight": 0.3, "base_motion_weight": 0.05}])
def test_reward_matches_compute_reward(shaping):
    cfg, _ = env_config(**shaping)
    env = HERLocoEnv(config=cfg)
    env.reward_mode = "track"
    env.set_smoothness_ramp(0.5)
    data = rollout(env, 1500)
    rc = RewardConfig.from_env(env)
    ach = np.array([d["ach"] for d in data])
    infos = [d["info"] for d in data]
    rng = np.random.default_rng(1)
    # stored goals, goals from other steps (HER), zero and small commands
    goals = [np.array([d["goal"] for d in data]), ach[rng.integers(0, len(ach), len(ach))], np.zeros_like(ach), rng.uniform(-0.15, 0.15, ach.shape).astype(np.float32),
             np.column_stack([np.full(len(ach), 0.1), np.zeros(len(ach)), np.zeros(len(ach))]).astype(np.float32)]
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=DEVICE)
    scal = {k: t([float(i[k]) for i in infos]) for k in ("r_smooth", "r_gait_move", "r_gait_still", "r_flight", "fell")}
    assert any(i["fell"] for i in infos), "rollout needs a fall"
    for g in goals:
        want = env.compute_reward(ach, g, infos)
        got = relabel_reward(rc, t(ach), t(g), scal["r_smooth"], scal["r_gait_move"], scal["r_gait_still"], scal["r_flight"], scal["fell"]).cpu().numpy()
        assert np.max(np.abs(want - got)) < 1e-6
    stored = np.array([d["r"] for d in data], dtype=np.float32)
    got = relabel_reward(rc, t(ach), goals[0] if False else t(goals[0]), scal["r_smooth"], scal["r_gait_move"], scal["r_gait_still"], scal["r_flight"], scal["fell"]).cpu().numpy()
    assert np.max(np.abs(stored - got)) < 1e-6
    env.close()


def filled_replay(steps=2200, n_envs=2, capacity=5000, batch=512):
    cfg, _ = env_config()
    envs = [HERLocoEnv(config=copy.deepcopy(cfg)) for _ in range(n_envs)]
    for e in envs:
        e.reward_mode = "track"
    rc = RewardConfig.from_env(envs[0])
    rb = GpuHerReplay(capacity, n_envs, batch, 2, 48, 12, rc, DEVICE)
    all_data = [rollout(e, steps, seed=i) for i, e in enumerate(envs)]
    for t in range(steps):
        row = np.zeros((n_envs, rb.width), dtype=np.float32)
        ended = np.zeros(n_envs, dtype=bool)
        for e in range(n_envs):
            d = all_data[e][t]
            i = rb.i_rew
            row[e, :48], row[e, rb.i_next:rb.i_act], row[e, rb.i_act:i] = d["obs"], d["nxt"], d["a"]
            info = d["info"]
            row[e, i:] = (d["r"], d["term"], info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"])
            ended[e] = d["term"] or d["trunc"]
        rb.add(torch.as_tensor(row, device=DEVICE), ended)
    return rb, all_data, envs


def test_her_goals_come_from_the_future_of_the_same_episode():
    rb, all_data, envs = filled_replay()
    for _ in range(20):
        obs, act, rew, nxt, done, _ = rb.sample()
        V = rb.nb_virtual
        real_obs, goals = obs[V:].cpu().numpy(), obs[:V, 3:6].cpu().numpy()
        assert np.array_equal(goals, nxt[:V, 3:6].cpu().numpy())
        # each virtual transition: find it in the raw data and check the goal is an achieved goal at t' >= t in its episode
        for o, g in zip(obs[:V].cpu().numpy()[:60], goals[:60]):
            found = False
            for data in all_data:
                for t, d in enumerate(data):
                    if np.array_equal(d["obs"][6:], o[6:]) and np.array_equal(d["obs"][:3], o[:3]):
                        k = t
                        while True:
                            if np.array_equal(data[k]["ach"], g):
                                found = True
                                break
                            if data[k]["term"] or data[k]["trunc"]:
                                break
                            k += 1
                    if found:
                        break
                if found:
                    break
            assert found
    for e in envs:
        e.close()


def test_incomplete_episodes_are_never_sampled_and_rewards_recompute():
    rb, all_data, envs = filled_replay()
    last_end = [max(i for i, d in enumerate(data) if d["term"] or d["trunc"]) for data in all_data]
    assert rb._n_valid == sum(x + 1 for x in last_end)
    flat = rb.valid[:rb._n_valid].cpu().numpy()
    for e, end in enumerate(last_end):
        rows = flat[flat % rb.n_envs == e] // rb.n_envs
        assert rows.max() == end
    # virtual rewards equal the env's reward for that transition with the new goal
    env = envs[0]
    obs, act, rew, nxt, done, _ = rb.sample()
    V = rb.nb_virtual
    lookup = {d["nxt"].tobytes()[12:]: d["info"] for data in all_data for d in data}
    for i in range(V):
        n = nxt[i].cpu().numpy()
        info = lookup[n.tobytes()[12:]] if n.tobytes()[12:] in lookup else None
        key = np.concatenate((n[:3], n[6:]))
        match = [d for data in all_data for d in data if np.array_equal(np.concatenate((d["nxt"][:3], d["nxt"][6:])), key)]
        assert match
        want = env.compute_reward(n[:3], n[3:6], match[0]["info"])
        assert abs(want - rew[i].item()) < 1e-6
        if i > 40:
            break
    for e in envs:
        e.close()


def test_replay_wraps_like_sb3():
    rb, all_data, envs = filled_replay(steps=1600, n_envs=2, capacity=2000)
    assert rb.rows == 1000
    # nothing valid may point at a row overwritten by a later episode: every valid slot's episode must be intact
    flat = rb.valid[:rb._n_valid].cpu().numpy()
    ep_len = rb._ep_len_np.ravel()
    assert (ep_len[flat] > 0).all()
    assert rb._n_valid == int((ep_len > 0).sum())
    rb.sample()
    for e in envs:
        e.close()


def test_log_prob_matches_sb3():
    cfg, _ = env_config()
    env = HERLocoEnv(config=cfg)
    model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs={"n_sampled_goal": 2, "copy_info_dict": True}, buffer_size=1000, learning_starts=0, device="cpu", verbose=0)
    mine = Actor(48, 12)
    mine.load_state_dict(model.actor.state_dict(), strict=True)
    o, _ = env.reset(seed=0)
    d = {k: torch.as_tensor(v)[None].repeat(32, 1) for k, v in o.items()}
    x = torch.cat((d["achieved_goal"], d["desired_goal"], d["observation"]), dim=1)
    torch.manual_seed(5)
    a_sb3, lp_sb3 = model.actor.action_log_prob(d)
    torch.manual_seed(5)
    a, lp = mine.sample(x)
    assert torch.allclose(a, a_sb3, atol=1e-6)
    assert torch.allclose(lp, lp_sb3.reshape(-1), atol=1e-4)
    assert torch.allclose(mine.deterministic(x), model.actor(d, deterministic=True), atol=1e-6)
    env.close()


def test_export_loads_in_the_independent_evaluator(tmp_path):
    cfg, _ = env_config()
    env = HERLocoEnv(config=copy.deepcopy(cfg))
    env.reward_mode = "track"
    rc = RewardConfig.from_env(env)
    rb = GpuHerReplay(1000, 1, 8, 2, 48, 12, rc, DEVICE)
    learner = SacLearner(rb, 48, 12, DEVICE, mode="eager")
    exporter = Sb3Exporter(cfg, 2, 8, 0.99, 0.005, 3e-4)
    path = tmp_path / "exported"
    exporter.save(learner, str(path), 1234)
    policy = load_policy(str(path) + ".zip", cfg)
    o, _ = env.reset(seed=3)
    a = policy(o)
    x = torch.as_tensor(flatten_obs(o), device=DEVICE)[None]
    b = learner.actor.deterministic(x)[0].detach().cpu().numpy()
    assert np.allclose(a, b, atol=1e-5)
    env.close()


def _captured_sampler(rb):
    B = rb.batch_size
    obs_s = torch.zeros((B, rb.obs_dim), device=DEVICE)
    nxt_s = torch.zeros((B, rb.obs_dim), device=DEVICE)
    rew_s = torch.zeros(B, device=DEVICE)
    rb.sample()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o, _, r, n, _, _ = rb.sample()
        obs_s.copy_(o)
        nxt_s.copy_(n)
        rew_s.copy_(r)
    return g, obs_s, nxt_s, rew_s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_replay_resamples_relabels_correctly_and_sees_new_data():
    rb, all_data, envs = filled_replay(batch=512)
    g, obs_s, nxt_s, rew_s = _captured_sampler(rb)
    g.replay()
    a = obs_s.clone()
    g.replay()
    assert not torch.equal(a, obs_s), "replay reused the same random numbers"

    # relabelled rewards from the replayed graph match the env's compute_reward
    V = rb.nb_virtual
    env = envs[0]
    index = {np.concatenate((d["nxt"][:3], d["nxt"][6:])).tobytes(): d["info"] for data in all_data for d in data}
    n, r = nxt_s.cpu().numpy(), rew_s.cpu().numpy()
    checked = 0
    for i in range(V):
        info = index[np.concatenate((n[i, :3], n[i, 6:])).tobytes()]
        assert abs(env.compute_reward(n[i, :3], n[i, 3:6], info) - r[i]) < 1e-6
        assert np.array_equal(n[i, 3:6], obs_s[i, 3:6].cpu().numpy())
        checked += 1
    assert checked == V

    # transitions added after capture must be sampled, with a goal from the new episode
    width = rb.width
    marked = np.zeros((rb.n_envs, width), dtype=np.float32)
    marked[:, 6] = 12345.0
    marked[:, rb.i_next + 6] = 12345.0
    marked[:, rb.i_next] = 0.77  # achieved vx
    for t in range(200):
        ended = np.array([t == 199] * rb.n_envs)
        rb.add(torch.as_tensor(marked, device=DEVICE), ended)
    seen, goals_ok = False, True
    for _ in range(200):
        g.replay()
        m = obs_s[:, 6] == 12345.0
        if m.any():
            seen = True
            v = m[:V]
            if v.any():
                goals_ok &= bool((obs_s[:V][v, 3] == 0.77).all())
    assert seen, "graph never sampled data added after capture"
    assert goals_ok, "goal for a new episode came from outside it"
    for e in envs:
        e.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_adam_matches_eager_adam():
    torch.manual_seed(0)
    w0 = torch.randn(64, 8, device=DEVICE)
    x = torch.randn(256, 8, device=DEVICE)
    y = torch.randn(256, 64, device=DEVICE)
    def loss(w):
        return ((x @ w.T - y) ** 2).mean()
    plain = w0.clone().requires_grad_(True)
    opt_p = torch.optim.Adam([plain], lr=3e-4)
    for _ in range(40):
        opt_p.zero_grad()
        loss(plain).backward()
        opt_p.step()
    graphed = w0.clone().requires_grad_(True)
    opt_g = torch.optim.Adam([graphed], lr=3e-4, fused=True, capturable=True)
    def step():
        opt_g.zero_grad(set_to_none=False)
        loss(graphed).backward()
        opt_g.step()
    step()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for _ in range(39):
        g.replay()
    assert torch.allclose(plain, graphed, atol=1e-5), float((plain - graphed).abs().max())
    assert int(opt_g.state[graphed]["step"].item()) == 40


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_learner_updates_everything():
    rb, all_data, envs = filled_replay(batch=256)
    torch.manual_seed(1)
    learner = SacLearner(rb, 48, 12, DEVICE, mode="graph")
    for _ in range(5):
        learner.update()
    p0 = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    c0 = {k: v.clone() for k, v in learner.critic.state_dict().items()}
    t0 = {k: v.clone() for k, v in learner.critic_target.state_dict().items()}
    alpha0 = learner.log_alpha.clone()
    s0 = learner.read_stats()
    for _ in range(200):
        learner.update()
    torch.cuda.synchronize()
    s1 = learner.read_stats()
    assert s0 != s1
    assert all(not torch.equal(p0[k], v) for k, v in learner.actor.state_dict().items())
    assert all(not torch.equal(c0[k], v) for k, v in learner.critic.state_dict().items())
    assert all(not torch.equal(t0[k], v) for k, v in learner.critic_target.state_dict().items())
    # polyak: the target moves, but much less than the online critic
    k = "qf0.0.weight"
    assert (learner.critic_target.state_dict()[k] - t0[k]).abs().mean() < (learner.critic.state_dict()[k] - c0[k]).abs().mean()
    assert not torch.equal(alpha0, learner.log_alpha)
    assert learner.log_alpha.item() < alpha0.item(), "alpha should fall from 1.0 at the start of SAC"
    assert int(learner.actor_opt.state[next(iter(learner.actor.parameters()))]["step"].item()) == 206  # 1 eager + 205 replays
    # same batch math: an eager step after replays does not explode
    assert np.isfinite(list(s1.values())).all()
    for e in envs:
        e.close()


def _episodes(data):
    eps, cur = [], []
    for d in data:
        cur.append(d)
        if d["term"] or d["trunc"]:
            eps.append(cur)
            cur = []
    return eps


def test_n_step_returns_match_a_hand_calculation():
    n, gamma = 3, 0.99
    cfg, _ = env_config()
    env = HERLocoEnv(config=copy.deepcopy(cfg))
    env.reward_mode = "track"
    rc = RewardConfig.from_env(env)
    data = rollout(env, 1500)
    rb = GpuHerReplay(5000, 1, 512, 2, 48, 12, rc, DEVICE, n_step=n, gamma=gamma)
    for d in data:
        row = np.zeros((1, rb.width), dtype=np.float32)
        i = rb.i_rew
        info = d["info"]
        row[0, :48], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = d["obs"], d["nxt"], d["a"]
        row[0, i:] = (d["r"], d["term"], info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"])
        rb.add(torch.as_tensor(row, device=DEVICE), np.array([d["term"] or d["trunc"]]))
    index = {}
    for ep in _episodes(data):
        for t, d in enumerate(ep):
            index[np.concatenate((d["obs"][:3], d["obs"][6:])).tobytes()] = (ep, t)
    V = rb.nb_virtual
    short_seen = done_seen = False
    for _ in range(10):
        obs, act, rew, nxt, done, disc = [x.cpu().numpy() for x in rb.sample()]
        for j in range(len(obs)):
            ep, t = index[np.concatenate((obs[j, :3], obs[j, 6:])).tobytes()]
            window = ep[t:t + n]
            goal = obs[j, 3:6]
            if j < V:
                want = sum(gamma ** k * float(env.compute_reward(w["ach"], goal, w["info"])) for k, w in enumerate(window))
            else:
                assert np.array_equal(goal, ep[t]["obs"][3:6])
                want = sum(gamma ** k * float(w["r"]) for k, w in enumerate(window))
            assert abs(want - rew[j]) < 1e-4
            assert abs(disc[j] - gamma ** len(window)) < 1e-6
            assert np.array_equal(nxt[j, :3], window[-1]["nxt"][:3]) and np.array_equal(nxt[j, 6:], window[-1]["nxt"][6:])
            assert np.array_equal(nxt[j, 3:6], goal)
            assert done[j] == float(window[-1]["term"])
            short_seen |= len(window) < n
            done_seen |= bool(done[j])
    assert short_seen and done_seen, "the test must cover windows cut by an episode end"
    env.close()


def test_fastsac_options_run_in_graph_mode_and_export(tmp_path):
    rb, all_data, envs = filled_replay(batch=256)
    cfg, _ = env_config()
    variants = [dict(avg_q=True), dict(layer_norm=True), dict(init_log_std=-3.0, init_alpha=0.01)]
    for kw in variants:
        learner = SacLearner(rb, 48, 12, DEVICE, mode="graph", **kw)
        std0 = learner.actor.dist_params(rb.store[:64, :48])[1].exp().mean().item()
        if "init_log_std" in kw:
            assert abs(std0 - np.exp(-3.0)) < 0.02
            assert abs(learner.log_alpha.exp().item() - 0.01) < 1e-6
        for _ in range(50):
            learner.update()
        torch.cuda.synchronize()
        assert np.isfinite(list(learner.read_stats().values())).all()
        exporter = Sb3Exporter(cfg, 2, 256, 0.99, 0.005, 3e-4)
        path = tmp_path / ("x" + "".join(kw))
        exporter.save(learner, str(path), 100)
        policy = load_policy(str(path) + ".zip", cfg)
        o, _ = envs[0].reset(seed=3)
        x = torch.as_tensor(flatten_obs(o), device=DEVICE)[None]
        assert np.allclose(policy(o), learner.actor.deterministic(x)[0].detach().cpu().numpy(), atol=1e-5)
    for e in envs:
        e.close()


def test_default_critic_keeps_sb3_parameter_names():
    from BipedRobot.fast.nets import Critic
    assert list(Critic(48, 12).state_dict().keys()) == ["qf0.0.weight", "qf0.0.bias", "qf0.2.weight", "qf0.2.bias", "qf0.4.weight", "qf0.4.bias",
                                                        "qf1.0.weight", "qf1.0.bias", "qf1.2.weight", "qf1.2.bias", "qf1.4.weight", "qf1.4.bias"]


def test_update_math_is_identical_to_sb3():
    """Same initial weights, same batches, same random numbers: every parameter must match SB3 after many updates."""
    from stable_baselines3.common.type_aliases import DictReplayBufferSamples

    cfg, _ = env_config()
    env = HERLocoEnv(config=cfg)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs={"n_sampled_goal": 2, "copy_info_dict": True},
                buffer_size=1000, learning_starts=0, batch_size=64, gamma=0.99, tau=0.005, learning_rate=3e-4, ent_coef="auto", device="cpu", verbose=0)
    cpu = torch.device("cpu")
    rb = GpuHerReplay(1000, 1, 64, 2, 48, 12, RewardConfig.from_env(env), cpu)
    fast = SacLearner(rb, 48, 12, cpu, mode="eager", lr=3e-4, gamma=0.99, tau=0.005)
    fast.actor.load_state_dict(model.actor.state_dict(), strict=True)
    fast.critic.load_state_dict(model.critic.state_dict(), strict=True)
    fast.critic_target.load_state_dict(model.critic_target.state_dict(), strict=True)
    assert model.target_entropy == fast.target_entropy
    n = 60
    data = np.random.default_rng(1)
    batches = [dict(o=data.normal(size=(64, 48)).astype(np.float32), a=data.uniform(-1, 1, (64, 12)).astype(np.float32), r=data.normal(size=64).astype(np.float32),
                    x=data.normal(size=(64, 48)).astype(np.float32), d=(data.uniform(size=64) < 0.1).astype(np.float32)) for _ in range(n)]
    split = lambda v: {"achieved_goal": v[:, :3], "desired_goal": v[:, 3:6], "observation": v[:, 6:]}
    it = iter(batches)

    def sb3_sample(batch_size, env=None):
        b = next(it)
        t = lambda x: torch.as_tensor(x)
        return DictReplayBufferSamples(observations={k: t(v) for k, v in split(b["o"]).items()}, actions=t(b["a"]), next_observations={k: t(v) for k, v in split(b["x"]).items()},
                                       dones=t(b["d"])[:, None], rewards=t(b["r"])[:, None])

    model.replay_buffer.sample = sb3_sample
    model._current_progress_remaining = 1.0
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(None, []))
    torch.manual_seed(42)
    model.train(gradient_steps=n, batch_size=64)

    it2 = iter(batches)
    def fast_sample():
        b = next(it2)
        t = lambda x: torch.as_tensor(x)
        return t(b["o"]), t(b["a"]), t(b["r"]), t(b["x"]), t(b["d"]), torch.tensor(0.99)
    rb.sample = fast_sample
    torch.manual_seed(42)
    for _ in range(n):
        fast.update()

    worst = 0.0
    for mine, theirs in ((fast.actor, model.actor), (fast.critic, model.critic), (fast.critic_target, model.critic_target)):
        ours, sb3 = mine.state_dict(), theirs.state_dict()
        assert ours.keys() == sb3.keys()
        for k in ours:
            worst = max(worst, float((ours[k] - sb3[k]).abs().max()))
    worst = max(worst, float((fast.log_alpha - model.log_ent_coef).abs().max()))
    print(f"largest parameter difference after {n} updates: {worst:.2e}")
    assert worst < 1e-4
    env.close()
