"""Does a trained budget policy obey the budget it is given? Measured with evaluation_suite_v2.run_episode, unchanged.

Usage: budget_eval.py LABEL CHECKPOINT_PT RESOLVED_CONFIG REALISM OUT_JSON [SEEDS start:stop]
Budgets are given as u per cost (0 tightest, 1 loosest). Delay stays at the training range in full realism; no pushes.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import EvalSettings, Thresholds, aggregate, load_env_config, run_episode  # noqa: E402
from BipedRobot.fast.budget_env import BudgetLocoEnv, decode_rate  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.nets import Actor  # noqa: E402

BUDGETS = {
    "all_loose": (1, 1, 1), "all_mid": (0.5, 0.5, 0.5), "all_tight": (0.25, 0.25, 0.25),
    "flight_mid": (0.5, 1, 1), "flight_tight": (0.0, 1, 1),
    "slip_mid": (1, 0.5, 1), "slip_tight": (1, 0.0, 1),
    "torque_mid": (1, 1, 0.5), "torque_tight": (1, 1, 0.0),
    # Extra budgets, by name, for runs whose token_budget.costs add "steps" or "lowsteps". Steps: u 0.2314 is rate 0.02 touchdowns per control step,
    # so a steady 2 steps per second at most, because the 100-step window allows up to twice the rate. Low steps: u 0 is the tightest, rate_min.
    "loose_steps2": ((1, 1, 1), {"steps": 0.2314}), "mid_steps2": ((0.5, 0.5, 0.5), {"steps": 0.2314}),
    "loose_lift": ((1, 1, 1), {"lowsteps": 0.0}), "mid_lift": ((0.5, 0.5, 0.5), {"lowsteps": 0.0}),
    "loose_steps2_lift": ((1, 1, 1), {"steps": 0.2314, "lowsteps": 0.0}), "mid_steps2_lift": ((0.5, 0.5, 0.5), {"steps": 0.2314, "lowsteps": 0.0}),
    # Tilted stance: rate 0.02 to 2 feet per control step; u 0.4375 is rate 0.15, so at most about 0.3 tilted feet per step on average,
    # which leaves room for heel strike and toe-off. "human" sets all three style budgets together.
    "loose_flat": ((1, 1, 1), {"tiltstance": 0.4375}), "mid_flat": ((0.5, 0.5, 0.5), {"tiltstance": 0.4375}),
    "loose_human": ((1, 1, 1), {"steps": 0.2314, "lowsteps": 0.0, "tiltstance": 0.4375}),
    "mid_human": ((0.5, 0.5, 0.5), {"steps": 0.2314, "lowsteps": 0.0, "tiltstance": 0.4375}),
}


def pad_budget(u, env):
    """u in env.cost_names order. A plain tuple gives flight, slip, torque; (tuple, {name: u}) adds named extras. Costs not given are loose (u = 1).
    None when the budget names a cost the env does not have."""
    base, extra = (u, {}) if not isinstance(u[0], (tuple, list)) else u
    names = tuple(getattr(env, "cost_names", ("flight", "slip", "torque")))
    if len(base) > 3 or set(extra) - set(names):
        return None
    value = {n: 1.0 for n in names}
    value.update(zip(("flight", "slip", "torque"), (float(x) for x in base)))
    value.update({k: float(v) for k, v in extra.items()})
    return tuple(value[n] for n in names)


class LoggedEnv(BudgetLocoEnv):
    def step(self, action):
        out = super().step(action)
        self.log.append((*out[4]["budget_cost"], float(out[4]["budget_overdrawn"])))
        return out


def main():
    label, ckpt, rc, realism, out = sys.argv[1:6]
    a, b = (int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "7000:7030").split(":"))
    torch.set_num_threads(1)
    cfg = copy.deepcopy(load_env_config(rc))
    settings = EvalSettings(realism=realism)
    cfg["realism"] = realism
    cfg["push_enabled"] = False
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    state = torch.load(ckpt, map_location="cpu")
    actor = Actor(state["obs_dim"], state["act_dim"])
    actor.load_state_dict(state["actor"])
    actor.eval()
    env = LoggedEnv(config=cfg, render_mode=None)
    env.reward_mode = "track"
    env.push_enabled = False

    @torch.no_grad()
    def policy(obs):
        return actor.deterministic(torch.as_tensor(flatten_obs(obs))[None])[0].numpy()

    results = {}
    for name, u in BUDGETS.items():
        u = pad_budget(u, env)
        if u is None:
            continue
        command = np.array([0.2, 0.0, 0.0, *u], np.float32)
        episodes, rows = [], []
        for seed in range(a, b):
            env.log = []
            ep = run_episode(policy, env, command, seed, settings, Thresholds())
            log = np.array(env.log)
            steady = log[50:] if len(log) > 50 else log
            rows.append({"seed": seed, "survived": ep["survived"], "steps": ep["steps"], "cost_mean": log[:, :3].mean(0).tolist(), "cost_mean_steady": steady[:, :3].mean(0).tolist(),
                         "overdrawn_frac": float(log[:, env.n_cost].mean()), "ever_overdrawn": bool(log[:, env.n_cost].any())})
            episodes.append(ep)
        rate = decode_rate(np.asarray(u, float), env.rate_min, env.rate_max)
        ok = [r for r in rows if r["steps"] > 100]
        results[name] = {"u": list(u), "rate": rate.tolist(), "aggregate": aggregate(episodes), "per_episode": rows,
                         "measured_cost": np.mean([r["cost_mean_steady"] for r in ok], axis=0).tolist() if ok else None,
                         "episodes_with_overdraft": int(sum(r["ever_overdrawn"] for r in rows)), "n": len(rows)}
    env.close()
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "realism": realism, "results": results}, default=str))


if __name__ == "__main__":
    main()
