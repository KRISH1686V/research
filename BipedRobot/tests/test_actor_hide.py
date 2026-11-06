"""Policy inputs hidden from the actor (--actor_hide). Run by path from the workspace root."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.budget_env import BudgetLocoEnv  # noqa: E402
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.learner import SacLearner  # noqa: E402
from BipedRobot.fast.nets import Actor, actor_from_checkpoint, actor_inputs  # noqa: E402
from BipedRobot.fast.replay import RewardConfig  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_and_config():
    c = yaml.safe_load(open(REPO / "config/v2_budget_window.yaml"))
    ec = build_env_config(c, "full", False)
    ec["validity"], ec["token_budget"] = c["validity"], c["token_budget"]
    env = BudgetLocoEnv(config=ec)
    env.reward_mode = "track"
    return env, c


def test_inputs_kept_are_the_ones_named():
    keep = actor_inputs(222, 6, 3, {"achieved", "levels"})
    assert keep == list(range(3, 219))
    assert actor_inputs(222, 6, 3, {"levels"}) == list(range(0, 219))
    assert actor_inputs(222, 6, 3, {"achieved"}) == list(range(3, 222))


def test_hidden_inputs_cannot_change_the_action():
    torch.manual_seed(0)
    actor = Actor(222, 12, keep=actor_inputs(222, 6, 3, {"achieved", "levels"})).eval()
    obs = torch.randn(4, 222)
    other = obs.clone()
    other[:, :3] = 9.0
    other[:, -3:] = -9.0
    assert torch.equal(actor.deterministic(obs), actor.deterministic(other))
    changed = obs.clone()
    changed[:, 3] += 1.0    # the command is still seen
    assert not torch.equal(actor.deterministic(obs), actor.deterministic(changed))
    changed = obs.clone()
    changed[:, 218] += 1.0  # the last proprioceptive reading is still seen
    assert not torch.equal(actor.deterministic(obs), actor.deterministic(changed))


def test_default_actor_has_no_mask_in_its_state():
    actor = Actor(222, 12)
    assert actor.keep is None and "keep" not in actor.state_dict()
    assert actor.latent_pi[0].in_features == 222


def test_checkpoint_round_trip_keeps_the_mask():
    torch.manual_seed(1)
    actor = Actor(222, 12, keep=actor_inputs(222, 6, 3, {"achieved", "levels"})).eval()
    state = {"actor": actor.state_dict(), "obs_dim": 222, "act_dim": 12, "steps": 0}
    back = actor_from_checkpoint(state)
    obs = torch.randn(3, 222)
    assert torch.equal(actor.deterministic(obs), back.deterministic(obs))
    plain = Actor(222, 12).eval()
    state = {"actor": plain.state_dict(), "obs_dim": 222, "act_dim": 12, "steps": 0}
    assert torch.equal(plain.deterministic(obs), actor_from_checkpoint(state).deterministic(obs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_learner_trains_the_masked_actor_and_critic_still_sees_everything():
    env, c = env_and_config()
    dim = sum(int(np.prod(s.shape)) for s in env.observation_space.spaces.values())
    rb = BudgetHerReplay(3000, 1, 256, 2, dim, 12, RewardConfig.from_env(env), DEVICE, c["token_budget"])
    keep = actor_inputs(dim, 6, 3, {"achieved", "levels"})
    rng = np.random.default_rng(0)
    o, _ = env.reset(seed=2)
    i = rb.i_rew
    for _ in range(800):
        a = rng.uniform(-0.5, 0.5, 12).astype(np.float32)
        o2, r, term, trunc, info = env.step(a)
        row = np.zeros((1, rb.width), dtype=np.float32)
        row[0, :dim], row[0, rb.i_next:rb.i_act], row[0, rb.i_act:i] = flatten_obs(o), flatten_obs(o2), a
        row[0, i:i + 10] = (r, term, info["r_smooth"], info["r_gait_move"], info["r_gait_still"], info["r_flight"], info["fell"], *info["budget_cost"])
        rb.add(torch.as_tensor(row, device=DEVICE), np.array([term or trunc]))
        o = o2
        if term or trunc:
            o, _ = env.reset()
    torch.manual_seed(2)
    learner = SacLearner(rb, dim, 12, DEVICE, mode="graph", layer_norm=True, avg_q=True, ema_tau=0.002, actor_keep=keep)
    assert learner.actor.latent_pi[0].in_features == len(keep) == dim - 6
    assert learner.critic.qf0[0].in_features == dim + 12
    assert torch.equal(learner.actor_ema.keep, learner.actor.keep)
    before = learner.actor.latent_pi[0].weight.detach().clone()
    for _ in range(40):
        learner.update()
    torch.cuda.synchronize()
    assert not torch.equal(before, learner.actor.latent_pi[0].weight)
    assert np.isfinite(list(learner.read_stats().values())).all()
    act = learner.act(torch.zeros((3, dim), device=DEVICE))
    assert act.shape == (3, 12)
    env.close()
