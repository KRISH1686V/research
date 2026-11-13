"""Harder survival test for budget policies than budget_eval.py: pushes on, longer episodes, new seeds. Measured with evaluation_suite_v2.run_episode, unchanged.

Usage: survive_eval.py LABEL CHECKPOINT_PT RESOLVED_CONFIG OUT_JSON REALISM PUSHES(0|1) SECONDS SEEDS(start:stop) BUDGETS(comma list, or all) [HIDE]
Budgets are the named ones in budget_eval.py. Keeps every episode's metrics, so spreads can be read back without rerunning.
HIDE, a comma list, replaces inputs the real robot cannot measure before the policy sees them. The physics and costs are untouched:
  vel_zero     measured forward and sideways speed reads 0
  vel_cmd      measured forward and sideways speed reads as the command
  levels_full  budget balances read as full
Also in that list, hold_heading turns the yaw command into a heading hold, wz = clip(-gain * yaw), as the Pi could do from its IMU. No retraining and no sideways correction.
hold_gyro does the same with no true heading: yaw comes from adding up the sensed yaw rate, with its noise and bias, so only what the robot can measure is used.
The forward command is 0.2 m/s unless the environment variable SURVIVE_SPEED gives another (added 6 Nov 2026 for the straight-step arm, trained at 0.05-0.15).
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import SCALAR_METRICS, EvalSettings, Thresholds, aggregate, load_env_config, run_episode  # noqa: E402
from BipedRobot.fast.budget_env import decode_rate  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.nets import actor_from_checkpoint  # noqa: E402
from budget_eval import BUDGETS, LoggedEnv, pad_budget  # noqa: E402


class SpeedLoggedEnv(LoggedEnv):
    def step(self, action):
        out = super().step(action)
        self.peak_joint_speed = max(self.peak_joint_speed, float(np.abs(self.data.qvel[self.joint_dof_adrs]).max()))
        return out


SPEED = float(os.environ.get("SURVIVE_SPEED", "0.2"))
HEADING_GAIN, HEADING_MAX_WZ = 1.0, 0.15  # 0.15 rad/s is the largest yaw command in training


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def sensed_yaw_rate(env, obs):
    """Yaw rate in rad/s from the newest frame the policy sees, with the sensor's noise and bias. Frames come oldest first, then the three balances."""
    flat = obs["observation"]
    frame = (len(flat) - env.n_cost) // env.history
    return float(flat[(env.history - 1) * frame + 2 * env.num_joints + 2]) / env.obs_scale["base_ang_vel"]


def hide_inputs(obs, hide, n_levels=3):
    if not hide:
        return obs
    obs = {k: v.copy() for k, v in obs.items()}
    if "vel_zero" in hide:
        obs["achieved_goal"][:2] = 0.0
    if "vel_cmd" in hide:
        obs["achieved_goal"][:2] = obs["desired_goal"][:2]
    if "levels_full" in hide:
        obs["observation"][-n_levels:] = 1.0
    return obs


def load_policy(ckpt):
    """A plain-weights checkpoint from fast/train.py, or a zip and normalizer from fast/train_ppo.py. Returns the policy on one observation dict, and its training steps."""
    if ckpt.endswith(".zip"):
        from stable_baselines3 import PPO
        model = PPO.load(ckpt, device="cpu")
        z = np.load(ckpt[:-4] + "_norm.npz")
        mean, var, clip, keep = z["mean"], z["var"], float(z["clip"]), z["keep"]
        return lambda obs: model.predict(np.clip((flatten_obs(obs)[keep] - mean) / np.sqrt(var + 1e-8), -clip, clip), deterministic=True)[0], model.num_timesteps
    state = torch.load(ckpt, map_location="cpu")
    actor = actor_from_checkpoint(state)

    @torch.no_grad()
    def policy(obs):
        return actor.deterministic(torch.as_tensor(flatten_obs(obs))[None])[0].numpy()
    return policy, state.get("steps")


def strict_overdrawn(costs, rate, capacity):
    """The plain bucket with no limit on its memory, whatever the env's own window is, so runs stay comparable."""
    d = np.zeros(len(rate))
    for c in costs:
        d = np.maximum(0.0, d + c - rate)
        if (d > capacity).any():
            return True
    return False


def main():
    label, ckpt, rc, out, realism, pushes, seconds, seeds, names = sys.argv[1:10]
    hide = set(sys.argv[10].split(",")) if len(sys.argv) > 10 else set()
    a, b = (int(x) for x in seeds.split(":"))
    max_steps = int(round(float(seconds) * 50))
    budgets = BUDGETS if names == "all" else {n: BUDGETS[n] for n in names.split(",")}
    torch.set_num_threads(1)
    cfg = copy.deepcopy(load_env_config(rc))
    settings = EvalSettings(realism=realism, max_steps=max_steps, pushes=bool(int(pushes)))
    cfg["realism"] = realism
    cfg["push_enabled"] = False
    cfg["max_episode_steps"] = max_steps
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    raw_policy, trained_steps = load_policy(ckpt)
    env = SpeedLoggedEnv(config=cfg, render_mode=None)
    env.reward_mode = "track"
    env.push_enabled = settings.pushes

    gyro_yaw = [0.0]

    def policy(obs):
        seen = hide_inputs(obs, hide, env.n_cost)
        if "hold_heading" in hide:
            seen = {k: v.copy() for k, v in seen.items()}
            seen["desired_goal"][2] = float(np.clip(-HEADING_GAIN * yaw_of(env.data.qpos[3:7]), -HEADING_MAX_WZ, HEADING_MAX_WZ))
        if "hold_gyro" in hide:
            if not env.log:
                gyro_yaw[0] = 0.0
            seen = {k: v.copy() for k, v in seen.items()}
            seen["desired_goal"][2] = float(np.clip(-HEADING_GAIN * gyro_yaw[0], -HEADING_MAX_WZ, HEADING_MAX_WZ))
            gyro_yaw[0] += sensed_yaw_rate(env, obs) * env.dt
        return raw_policy(seen)

    results = {}
    for name, u in budgets.items():
        u = pad_budget(u, env)
        if u is None:
            continue
        command = np.array([SPEED, 0.0, 0.0, *u], np.float32)
        rate = decode_rate(np.asarray(u, float), env.rate_min, env.rate_max)
        episodes, rows = [], []
        for seed in range(a, b):
            env.log = []
            env.peak_joint_speed = 0.0
            ep = run_episode(policy, env, command, seed, settings, Thresholds())
            log = np.array(env.log)
            steady = log[50:] if len(log) > 50 else log
            rows.append({"seed": seed, "survived": ep["survived"], "steps": ep["steps"], "cause": ep["termination_cause"], "gait": ep["gait_class"],
                         "cost_mean_steady": steady[:, :env.n_cost].mean(0).tolist(), "overdrawn_frac": float(log[:, env.n_cost].mean()), "ever_overdrawn": bool(log[:, env.n_cost].any()),
                         "strict_ever_overdrawn": strict_overdrawn(log[:, :env.n_cost], rate, rate * env.burst_steps),
                         "peak_joint_speed": env.peak_joint_speed, **{m: ep[m] for m in SCALAR_METRICS}})
            episodes.append(ep)
        ok = [r for r in rows if r["steps"] > 100]
        results[name] = {"u": list(u), "rate": rate.tolist(), "aggregate": aggregate(episodes), "per_episode": rows,
                         "measured_cost": np.mean([r["cost_mean_steady"] for r in ok], axis=0).tolist() if ok else None,
                         "episodes_with_overdraft": int(sum(r["ever_overdrawn"] for r in rows)), "n": len(rows)}
    env.close()
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "steps": trained_steps, "realism": realism, "pushes": settings.pushes, "seconds": float(seconds), "speed": SPEED,
                                     "seeds": seeds, "hide": sorted(hide), "results": results}, default=str))


if __name__ == "__main__":
    main()
