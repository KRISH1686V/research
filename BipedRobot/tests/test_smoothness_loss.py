"""The policy smoothness loss (--caps_t, --caps_s): it matches its formula, stays off by default, and survives graph capture. Run by path from the workspace root."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.fast.learner import SacLearner  # noqa: E402
from BipedRobot.tests.test_fast_learner import DEVICE, filled_replay  # noqa: E402


def replay():
    rb, _, envs = filled_replay(steps=600, n_envs=2, capacity=2000)
    for e in envs:
        e.close()
    return rb


def test_temporal_term_matches_its_formula_and_skips_episode_ends():
    rb = replay()
    learner = SacLearner(rb, 48, 12, DEVICE, mode="eager", caps_t=2.0)
    obs, _, _, nxt, done, _ = rb.sample()
    done = torch.zeros_like(done)
    done[:len(done) // 2] = 1.0
    with torch.no_grad():
        gap = (learner.actor.deterministic(obs) - learner.actor.deterministic(nxt)).pow(2).sum(-1)
        want = 2.0 * (gap * (1.0 - done)).mean()
        got = learner._smoothness(obs, nxt, done)
    assert torch.allclose(got, want)


def test_spatial_term_is_zero_with_no_noise():
    rb = replay()
    learner = SacLearner(rb, 48, 12, DEVICE, mode="eager", caps_s=2.0, caps_sigma=0.0)
    obs, _, _, nxt, done, _ = rb.sample()
    with torch.no_grad():
        assert float(learner._smoothness(obs, nxt, done)) == 0.0


def test_off_by_default(monkeypatch):
    rb = replay()
    learner = SacLearner(rb, 48, 12, DEVICE, mode="eager")
    monkeypatch.setattr(learner, "_smoothness", lambda *a: pytest.fail("must not run when off"))
    learner.update()


def test_graph_capture_runs_and_makes_the_policy_smoother():
    rb = replay()
    gaps = {}
    for name, kw in (("off", {}), ("on", {"caps_t": 3.0, "caps_s": 3.0})):
        torch.manual_seed(5)
        learner = SacLearner(rb, 48, 12, DEVICE, mode="graph", layer_norm=True, avg_q=True, **kw)
        for _ in range(400):
            learner.update()
        torch.manual_seed(9)
        obs, _, _, nxt, _, _ = rb.sample()
        with torch.no_grad():
            gaps[name] = float((learner.actor.deterministic(obs) - learner.actor.deterministic(nxt)).norm(dim=-1).mean())
    assert gaps["on"] < 0.7 * gaps["off"], gaps
