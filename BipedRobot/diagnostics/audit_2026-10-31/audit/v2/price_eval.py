"""Read budget prices from a trained critic: how much value does one more unit of allowance add? For E-D, against the multipliers of the dual-ascent runs.

Usage: price_eval.py LABEL CHECKPOINT_PT RESOLVED_CONFIG OUT_JSON SEEDS(start:stop) BUDGET [BUDGET ...]
Runs the policy, then asks the critic at each visited state how Q changes when one balance is raised and lowered by half a step of WIDTH, in cost units. price_toy.py shows why a width is used and not
a local derivative, and why only states with a part-empty bucket are read. A budget that never drains shows up as 0 states, which is NOT DETERMINED, not a zero price.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import load_env_config  # noqa: E402
from BipedRobot.fast.budget_env import decode_rate  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.nets import Critic, actor_from_checkpoint  # noqa: E402
from budget_eval import BUDGETS, LoggedEnv  # noqa: E402

COSTS = ("flight", "slip", "torque")
LEVEL_WIDTH = 0.5   # raised and lowered by half of this, as a share of the bucket's size B, so the width matches the bucket whatever the rate
PART_EMPTY = (-0.5, 0.75)


def main():
    label, ckpt, rc, out, seeds = sys.argv[1:6]
    names = sys.argv[6:]
    a, b = (int(x) for x in seeds.split(":"))
    torch.set_num_threads(1)
    state = torch.load(ckpt, map_location="cpu")
    actor = actor_from_checkpoint(state)
    critic = Critic(state["obs_dim"] + state["priv_dim"], state["act_dim"], layer_norm=state["layer_norm"])
    critic.load_state_dict(state["critic"])
    critic.eval()
    cfg = copy.deepcopy(load_env_config(rc))
    cfg["realism"], cfg["push_enabled"] = "medium", False
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    env = LoggedEnv(config=cfg, render_mode=None)
    assert env.n_cost == 3, "price_eval reads three budgets: flight, slip and torque"
    env.reward_mode = "track"
    env.push_enabled = False
    results = {}
    for name in names:
        u = BUDGETS[name]
        rate = decode_rate(np.asarray(u, float), env.rate_min, env.rate_max)
        capacity = rate * env.burst_steps
        goal = np.array([0.2, 0.0, 0.0, *u], np.float32)
        obs_rows, priv_rows = [], []
        for seed in range(a, b):
            env.log = []
            o, _ = env.reset(seed=seed, options={"desired_goal": goal})
            for _ in range(500):
                x = flatten_obs(o)
                obs_rows.append(x)
                priv_rows.append(env.privileged().copy())
                with torch.no_grad():
                    act = actor.deterministic(torch.as_tensor(x)[None])[0].numpy()
                o, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    break
        x = torch.as_tensor(np.array(obs_rows))
        p = torch.as_tensor(np.array(priv_rows))
        with torch.no_grad():
            acts = actor.deterministic(x)
            res = {"states": len(x), "rate": rate.tolist(), "capacity": capacity.tolist()}
            for i, cost in enumerate(COSTS):
                level = x[:, -3 + i]
                mask = (level > PART_EMPTY[0]) & (level < PART_EMPTY[1])
                res[cost] = {"part_empty_states": int(mask.sum()), "share": float(mask.float().mean())}
                if mask.sum() >= 20:
                    hi, lo = x[mask].clone(), x[mask].clone()
                    hi[:, -3 + i] += LEVEL_WIDTH / 2
                    lo[:, -3 + i] -= LEVEL_WIDTH / 2
                    q_hi = torch.stack(critic(torch.cat((hi, p[mask]), 1), acts[mask])).mean(0)
                    q_lo = torch.stack(critic(torch.cat((lo, p[mask]), 1), acts[mask])).mean(0)
                    price = ((q_hi - q_lo) / (LEVEL_WIDTH * capacity[i])).numpy()   # value per unit of cost
                    res[cost].update(price_median=float(np.median(price)), price_p25=float(np.percentile(price, 25)), price_p75=float(np.percentile(price, 75)))
            results[name] = res
        line = " | ".join(f"{c} {res[c]['part_empty_states']} states" + (f", price {res[c]['price_median']:.2f} [{res[c]['price_p25']:.2f}, {res[c]['price_p75']:.2f}]" if "price_median" in res[c] else ", NOT DETERMINED") for c in COSTS)
        print(label, name, "rate", np.round(rate, 3).tolist(), line, flush=True)
    env.close()
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "seeds": seeds, "results": results}))


if __name__ == "__main__":
    main()
