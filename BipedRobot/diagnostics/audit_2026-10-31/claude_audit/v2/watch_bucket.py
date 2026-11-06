"""Watch the three budget buckets drain: random actions at the tightest budget, printing costs, bucket levels and reward every 10 steps. For learning the code."""
import sys
import numpy as np
import yaml
sys.path.insert(0, "/home/rpuser1/yasho/rl")
from BipedRobot.train_her_biped import build_env_config
from BipedRobot.fast.budget_env import BudgetLocoEnv

c = yaml.safe_load(open("/home/rpuser1/yasho/rl/BipedRobot/config/v2_deploy_budget_alpha30.yaml"))
ec = build_env_config(c, "medium", False)
ec["validity"], ec["token_budget"] = c["validity"], c["token_budget"]
env = BudgetLocoEnv(config=ec)
env.reward_mode = "track"
env.reset(seed=1, options={"desired_goal": np.array([0.2, 0, 0, 0.0, 0.0, 0.0], np.float32)})  # u = 0: the tightest budgets
rng = np.random.default_rng(0)
for t in range(60):
    o, r, term, trunc, info = env.step(rng.uniform(-0.8, 0.8, 12).astype(np.float32))
    if t % 10 == 0:
        print(t, "cost", info["budget_cost"].round(3), "levels", env.bucket_levels().round(2), "over", info["budget_overdrawn"], "reward", round(r, 3))
    if term:
        print("fell at step", t)
        break
