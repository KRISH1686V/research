"""Budget relabeling after the replay ring has been overwritten several times, the state every run is in after its first 1M steps.

test_budget_window.py checks the same arithmetic with a ring that never fills. Run by path from the working tree.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BipedRobot.tests.test_budget_her import BUDGET, fill, make_env, relabeled_truth, rollout  # noqa: E402
from BipedRobot.tests.test_budget_window import W, check_batch  # noqa: E402

CAPACITY = 300
STEPS = 2500


def _replay():
    budget = {**BUDGET, "window_steps": W, "alive_bonus": 0.2}
    env = make_env(window_steps=W, alive_bonus=0.2)
    data = rollout(env, STEPS, amp=0.6)
    return env, data, fill(env, data, capacity=CAPACITY, budget=budget)


def test_relabeling_stays_exact_after_the_ring_wraps():
    env, data, rb = _replay()
    assert rb.rows == CAPACITY and STEPS > 5 * CAPACITY, "the ring must be overwritten several times"
    index = relabeled_truth(env, rb, data)
    total = over = 0
    for _ in range(4):
        obs, _, rew, nxt, _, _ = rb.sample()
        c, o = check_batch(env, rb, *[x.cpu().numpy() for x in (obs, rew, nxt)], index, W)
        total += c
        over += o
    assert total > 1000 and over > 10, "the test must cover overdrawn relabels"
    env.close()


def test_valid_rows_are_whole_episodes_only():
    env, _, rb = _replay()
    flat = rb.valid[:rb._n_valid].cpu().numpy()
    starts, lengths = rb.ep_start.cpu().numpy()[flat], rb.ep_len.cpu().numpy()[flat]
    assert rb._n_valid == int((rb._ep_len_np > 0).sum())
    assert (lengths > 0).all()
    assert ((flat - starts) % rb.rows < lengths).all(), "a valid row must lie inside its own episode"
    newest = (rb.pos - 1) % rb.rows
    ages = (newest - flat) % rb.rows
    assert ages.max() < rb.rows
    # Every row of a stored episode is still stored, in order, with costs that add up.
    cum = rb.cum.cpu().numpy()
    cost = rb.store[:, rb.i_cost:rb.i_cost + 3].cpu().numpy().astype(np.float64)
    checked = 0
    for s, n in {(int(a), int(b)) for a, b in zip(starts, lengths)}:
        rows = (s + np.arange(n)) % rb.rows
        assert np.all(rb._ep_len_np[rows, 0] == n)
        assert np.allclose(cum[rows[0]], 0.0)
        assert np.allclose(cum[rows[1:]], cum[rows[:-1]] + cost[rows[:-1]], atol=1e-9)
        checked += 1
    assert checked >= 2
    env.close()
