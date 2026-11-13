"""Stick-figure GIF of one test episode, so a gait can be looked at instead of read from numbers. Same evaluator as survive_eval.py.

Usage: trace_gif.py LABEL CHECKPOINT RESOLVED_CONFIG OUT_FOLDER REALISM PUSHES(0|1) SECONDS SEED BUDGET
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import GRAVITY, EvalSettings, Thresholds, load_env_config, render_trace_gif, run_episode  # noqa: E402
from budget_eval import BUDGETS, LoggedEnv  # noqa: E402
from survive_eval import load_policy  # noqa: E402


def main():
    label, ckpt, rc, out, realism, pushes, seconds, seed, budget = sys.argv[1:10]
    torch.set_num_threads(1)
    max_steps = int(round(float(seconds) * 50))
    cfg = copy.deepcopy(load_env_config(rc))
    settings = EvalSettings(realism=realism, max_steps=max_steps, pushes=bool(int(pushes)))
    cfg["realism"], cfg["push_enabled"], cfg["max_episode_steps"] = realism, False, max_steps
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    policy, _ = load_policy(ckpt)
    env = LoggedEnv(config=cfg, render_mode=None)
    env.reward_mode = "track"
    env.push_enabled = settings.pushes
    env.log = []
    command = np.array([0.2, 0.0, 0.0, *BUDGETS[budget]], np.float32)
    ep = run_episode(policy, env, command, int(seed), settings, Thresholds(), record_trace=True)
    trace = ep.pop("_trace")
    Path(out).mkdir(parents=True, exist_ok=True)
    path = Path(out) / f"{label}_{realism}{'_push' if settings.pushes else ''}_{budget}_seed{seed}.gif"
    title = f"{label} {realism} {budget} seed {seed} [{ep['gait_class']}] {ep['speed_x_disp']:.3f} m/s"
    render_trace_gif(trace, path, title=title, weight_n=ep["mass_kg"] * GRAVITY)
    env.close()
    print(path, title, "survived", ep["survived"], "steps", ep["steps"])


if __name__ == "__main__":
    main()
