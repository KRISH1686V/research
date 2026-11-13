"""Per-step budget costs of standing still and of uniform random actions in the OP3 env (exploration before learning_starts), for setting rate ranges.
Usage: python diagnostics/op3/random_costs.py CONFIG OUT.json   (CPU only)"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from BipedRobot.fast.collector import env_class  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

out = {}
c = yaml.safe_load(open(sys.argv[1]))
for realism in ("easy", "full"):
    ec = build_env_config(c, realism, False)
    for k in ("validity", "token_budget"):
        ec[k] = copy.deepcopy(c[k])
    env = env_class(ec)(config=ec)
    env.reward_mode = "track"
    for name, amp in (("stand", 0.0), ("uniform_0.5", 0.5), ("uniform_1.0", 1.0)):
        rng = np.random.default_rng(0)
        env.reset(seed=0)
        costs, falls = [], 0
        for _ in range(3000):
            _, _, term, trunc, info = env.step(rng.uniform(-amp, amp, 12).astype(np.float32))
            costs.append(info["budget_cost"])
            if term or trunc:
                falls += int(term)
                env.reset()
        cm = np.mean(costs, 0)
        out[f"{realism}/{name}"] = {"cost_mean": cm.tolist(), "falls_per_3000_steps": falls}
        print(realism, name, np.round(cm, 7), "falls", falls, flush=True)
    env.close()
Path(sys.argv[2]).write_text(json.dumps({"config": sys.argv[1], "cost_names": ["flight", "slip", "jshake"], "results": out}, indent=1))
