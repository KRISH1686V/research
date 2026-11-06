"""What breaks a policy? One change at a time from medium realism, and the robot's own servo settings. No retraining. Same evaluator as survive_eval.py.

Usage: stress_eval.py LABEL CHECKPOINT RESOLVED_CONFIG OUT_JSON SEEDS(start:stop) BUDGET CONDITION [CONDITION ...]
Conditions are the names in CONDITIONS. Anything not named stays at medium realism. Command 0.2 m/s unless the condition sets it.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import SCALAR_METRICS, EvalSettings, Thresholds, aggregate, load_env_config, run_episode  # noqa: E402
from BipedRobot.fast.budget_env import decode_rate  # noqa: E402
from budget_eval import BUDGETS, pad_budget  # noqa: E402
from survive_eval import SpeedLoggedEnv, load_policy  # noqa: E402

# realism, then attributes to set on the env. torque: fraction of the full servo limit. target_rate: largest joint target speed, rad/s, as the servo speed setting would impose.
CONDITIONS = {
    "medium": ("medium", {}),
    "pushes": ("medium", {"push_enabled": True}),
    "pushes_half": ("medium", {"push_enabled": True, "dr_push_magnitude": 1.5}),
    "delay_0": ("medium", {"delay_range": (0.0, 0.0)}),
    "delay_30": ("medium", {"delay_range": (0.03, 0.03)}),
    "delay_50": ("medium", {"delay_range": (0.05, 0.05)}),
    "backlash": ("medium", {"enable_backlash": True}),
    "full": ("full", {}),
    "full_pushes": ("full", {"push_enabled": True}),
    "torque_70": ("medium", {"torque": 0.7}),
    "torque_50": ("medium", {"torque": 0.5}),
    "rate_4.6": ("medium", {"target_rate": 4.6}),
    "rate_2.3": ("medium", {"target_rate": 2.3}),
    "rate_1.2": ("medium", {"target_rate": 1.2}),
    "cmd_0.10": ("medium", {"command": 0.10}),
    "cmd_0.15": ("medium", {"command": 0.15}),
    "cmd_0.25": ("medium", {"command": 0.25}),
    "cmd_0.30": ("medium", {"command": 0.30}),
}


class StressEnv(SpeedLoggedEnv):
    target_rate = None
    _last_target = None

    def _action_to_target(self, action):
        target = super()._action_to_target(action)
        if self.target_rate:
            if self._last_target is None:
                self._last_target = self.default_qpos.copy()
            target = self._last_target + np.clip(target - self._last_target, -self.target_rate * self.dt, self.target_rate * self.dt)
            self._last_target = target.astype(np.float32).copy()
        return target

    def reset(self, **kwargs):
        self._last_target = None
        return super().reset(**kwargs)


def main():
    label, ckpt, rc, out, seeds, budget = sys.argv[1:7]
    names = sys.argv[7:]
    a, b = (int(x) for x in seeds.split(":"))
    torch.set_num_threads(1)
    policy, _ = load_policy(ckpt)
    u = BUDGETS[budget]
    results = {}
    for name in names:
        realism, change = CONDITIONS[name]
        cfg = copy.deepcopy(load_env_config(rc))
        cfg["realism"], cfg["push_enabled"] = realism, False
        cfg.setdefault("shaping", {})["reset_velocity"] = False
        env = StressEnv(config=cfg, render_mode=None)
        env.reward_mode = "track"
        env.push_enabled = False
        speed = 0.2
        for key, value in change.items():
            if key == "command":
                speed = value
            elif key == "torque":
                env._cap *= value
                env._stall *= value
                env.torque_limit *= value
            else:
                setattr(env, key, value)
        settings = EvalSettings(realism=realism, pushes=bool(env.push_enabled))
        command = np.array([speed, 0.0, 0.0, *pad_budget(u, env)], np.float32)
        rows, episodes = [], []
        for seed in range(a, b):
            env.log = []
            env.peak_joint_speed = 0.0
            ep = run_episode(policy, env, command, seed, settings, Thresholds())
            log = np.array(env.log)
            rows.append({"seed": seed, "survived": ep["survived"], "gait": ep["gait_class"], "cause": ep["termination_cause"], "steps": ep["steps"], "ever_overdrawn": bool(log[:, 3].any()),
                         "peak_joint_speed": env.peak_joint_speed, **{m: ep[m] for m in SCALAR_METRICS}})
            episodes.append(ep)
        env.close()
        results[name] = {"command": speed, "aggregate": aggregate(episodes), "per_episode": rows}
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "budget": budget, "rate": decode_rate(np.asarray(u, float), env.rate_min, env.rate_max).tolist(), "seeds": seeds, "results": results}, default=str))


if __name__ == "__main__":
    main()
