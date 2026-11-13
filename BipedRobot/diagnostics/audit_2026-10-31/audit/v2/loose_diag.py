"""What makes a policy fall, and which limit it breaks first? For diagnosis only, on seeds kept out of any goal check. Measured with evaluation_suite_v2.run_episode, unchanged.

Usage: loose_diag.py LABEL CHECKPOINT_PT RESOLVED_CONFIG OUT_JSON SEEDS(start:stop) BUDGETS(comma list)
Medium realism, no pushes, 10 s, command 0.2 m/s. Each episode records the randomized physics, when it first went over budget and on which cost, and how much of the last second was spent off the ground.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import EvalSettings, Thresholds, load_env_config, run_episode  # noqa: E402
from budget_eval import BUDGETS, pad_budget  # noqa: E402
from survive_eval import SpeedLoggedEnv, load_policy  # noqa: E402

COSTS = ("flight", "slip", "torque")
LAST_S = 50  # control steps in the last second


class DiagEnv(SpeedLoggedEnv):
    def step(self, action):
        if not self.log:
            self.params = [float(x) for x in self.privileged()[-4:]]
        out = super().step(action)
        self.actions.append(np.array(action, dtype=np.float32))
        if self.first_over is None and out[4]["budget_overdrawn"]:
            self.first_over = (len(self.log), [self.cost_names[i] for i in np.flatnonzero(self._deficit > self._capacity)])
        return out


def main():
    label, ckpt, rc, out, seeds, names = sys.argv[1:7]
    a, b = (int(x) for x in seeds.split(":"))
    torch.set_num_threads(1)
    cfg = copy.deepcopy(load_env_config(rc))
    settings = EvalSettings(realism="medium", max_steps=500, pushes=False)
    cfg["realism"], cfg["push_enabled"], cfg["max_episode_steps"] = "medium", False, 500
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    policy, steps = load_policy(ckpt)
    env = DiagEnv(config=cfg, render_mode=None)
    env.reward_mode = "track"
    env.push_enabled = False
    results = {}
    for name in names.split(","):
        command = np.array([0.2, 0.0, 0.0, *pad_budget(BUDGETS[name], env)], np.float32)
        rows = []
        for seed in range(a, b):
            env.log, env.actions, env.first_over, env.peak_joint_speed = [], [], None, 0.0
            ep = run_episode(policy, env, command, seed, settings, Thresholds())
            log, act = np.array(env.log), np.array(env.actions)
            rows.append({"seed": seed, "survived": ep["survived"], "steps": ep["steps"], "cause": ep["termination_cause"], "gait": ep["gait_class"],
                         "friction": env.params[0], "mass": env.params[1], "delay": env.params[2], "torque": env.params[3],
                         "first_over_step": env.first_over[0] if env.first_over else None, "first_over_costs": env.first_over[1] if env.first_over else [],
                         "flight_last_s": float(log[-LAST_S:, 0].mean()), "flight_all": float(log[:, 0].mean()),
                         "slip_all": float(log[:, 1].mean()), "torque_all": float(log[:, 2].mean()),
                         "action_step_rms": float(np.sqrt(np.mean(np.diff(act, axis=0) ** 2))) if len(act) > 1 else None,
                         "action_at_limit": float(np.mean(np.abs(act) > 0.95)),
                         "speed": ep["speed_x_disp_steady"], "touchdown_hz": ep["touchdown_hz"], "clearance_mm": ep["swing_clearance_mm_median"],
                         "heading_rad": ep["heading_change_rad"], "peak_joint_speed": env.peak_joint_speed})
        results[name] = rows
    env.close()
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "steps": steps, "seeds": seeds, "results": results}, default=str))


if __name__ == "__main__":
    main()
