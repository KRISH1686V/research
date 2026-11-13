"""GPU cost of the budget learner alone: same replay, networks and graph as training, on made-up data, with no env workers. Not a training run.

Usage: gpu_bench.py SECONDS START_AT_EPOCH [UPDATES_PER_VECTOR_STEP] [BATCH]
Prints updates per second. Each vector step queues the updates, runs the policy on 8 observations and reads the actions back, as train.py does.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch  # noqa: E402
from BipedRobot.fast.budget_replay import BudgetHerReplay  # noqa: E402
from BipedRobot.fast.learner import SacLearner  # noqa: E402
from BipedRobot.fast.mirror import action_perm, obs_maps, priv_maps  # noqa: E402
from BipedRobot.fast.replay import RewardConfig  # noqa: E402

DIM, PRIV, ACT, ENVS = 222, 46, 12, 8
BUDGET = dict(rate_min=[0.005, 0.02, 0.01], rate_max=[0.2, 0.2, 0.2], burst_steps=100, window_steps=100, p_keep=0.25, p_achieved=0.5, alive_bonus=0.2)


def main():
    seconds, start_at = float(sys.argv[1]), float(sys.argv[2])
    per_step = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    device = torch.device("cuda")
    cfg = RewardConfig(0.15, 0.15, 25.0, 0.3, False, True)
    op, os_ = obs_maps(5, 3)
    pp, ps = priv_maps()
    replay = BudgetHerReplay(1_000_000, ENVS, batch, 2, DIM, ACT, cfg, device, BUDGET, 500, priv_dim=PRIV,
                             mirror={"obs_perm": op, "obs_sign": os_, "act_perm": action_perm(), "priv_perm": pp, "priv_sign": ps})
    rng = np.random.default_rng(0)
    rows = np.zeros((ENVS, replay.width), dtype=np.float32)
    gpu = torch.zeros((ENVS, replay.width), device=device)
    for t in range(1500):
        rows[:] = rng.normal(0, 0.1, rows.shape)
        rows[:, replay.i_rew + 1] = 0.0
        rows[:, replay.i_rew + 6] = 0.0
        rows[:, replay.i_cost:replay.i_cost + 3] = rng.uniform(0, 0.1, (ENVS, 3))
        gpu.copy_(torch.from_numpy(rows))
        replay.add(gpu, np.full(ENVS, (t + 1) % 300 == 0), rows[:, replay.i_cost:replay.i_cost + 3])
    learner = SacLearner(replay, DIM, ACT, device, mode="graph", layer_norm=True, avg_q=True, priv_dim=PRIV, ema_tau=0.002)
    obs = torch.zeros((ENVS, DIM), device=device)
    learner.update()
    learner.act(obs)
    torch.cuda.synchronize()
    while time.time() < start_at:
        time.sleep(0.01)
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        for _ in range(per_step):
            learner.update()
        learner.act(obs).cpu()
        n += per_step
    print(f"{n / (time.perf_counter() - t0):.1f}", flush=True)


if __name__ == "__main__":
    main()
