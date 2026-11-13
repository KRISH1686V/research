"""Open-loop scripted walk for the OP3 env, for measuring per-step budget costs (flight, slip, jshake) before any training. Not used in training.

A hand-made gait: per leg, the ankle's path relative to its hip (stance foot slides back under the hip at the walking speed, swing foot returns forward on
a sin^2 lift), the torso swaying over the stance foot, solved by analytic two-link inverse kinematics with the soles kept level. Right foot in stance around
a quarter cycle, left foot half a cycle later. The joint targets go to the env as actions in its policy coordinates, inverting the action filter, so the env
runs exactly as in training (filter, delay, PD servo, costs).

Usage: python diagnostics/op3/scripted_walk.py CONFIG OUT.json [--realism easy|medium|full] [--seeds 0,1] [--noise 0.1] [--grid full|walk]   (CPU only)
--noise adds Gaussian noise of that std to every action (after the filter inversion, so the filter smooths it), as SAC's sampled actions do in training.
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from BipedRobot.fast.collector import env_class  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

THIGH, SHANK = 0.11015, 0.11  # hip pitch to knee, knee to ankle pitch (op3.xml)


def leg_ik(dx: float, dy: float, h: float) -> np.ndarray:
    """Left-leg joint angles [yaw, pitch, roll, knee, ank_pitch, ank_roll] that put the ankle at (dx forward, dy left, h below) the hip, sole level."""
    L = np.hypot(dy, h)
    roll = -np.arctan2(dy, h)  # left hip roll joint (axis -x): positive moves the foot inward
    Ls = np.hypot(dx, L)
    Ls = min(Ls, THIGH + SHANK - 1e-4)
    knee = np.pi - np.arccos(np.clip((THIGH ** 2 + SHANK ** 2 - Ls ** 2) / (2 * THIGH * SHANK), -1, 1))
    flex = np.arctan2(dx, L) + np.arccos(np.clip((THIGH ** 2 + Ls ** 2 - SHANK ** 2) / (2 * THIGH * Ls), -1, 1))
    return np.array([0.0, -flex, roll, knee, knee - flex, roll])


def foot_path(ph: float, p: dict) -> tuple:
    """Right ankle offset from its standing place (dx, dz) at phase ph, stance centred on a quarter cycle."""
    T, D, v = p["period_s"], p["duty"], p["speed"]
    X = v * T * D / 2.0
    s_st = ((ph - (0.25 - D / 2.0)) % 1.0) / D
    if s_st <= 1.0:
        return X - 2.0 * X * s_st, 0.0
    s = (s_st - 1.0) * D / (1.0 - D)
    Ts = T * (1.0 - D)
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * (-X) + h10 * (-v * Ts) + h01 * X + h11 * (-v * Ts), p["lift"] * np.sin(np.pi * s) ** 2


def targets(t: float, p: dict, h0: float) -> np.ndarray:
    """Joint targets in env order (right leg then left, each yaw, pitch, roll, knee, ank_pitch, ank_roll), joint coordinates."""
    ramp = float(np.clip(t / p["ramp_s"], 0.0, 1.0))
    ph = (t / p["period_s"]) % 1.0
    sway = ramp * p["sway"] * np.sin(2 * np.pi * ph)  # feet move left of the hips while the torso is over the right foot
    out = []
    for leg_ph, mirror in ((ph, True), ((ph + 0.5) % 1.0, False)):
        dx, dz = foot_path(leg_ph, p)
        dy = sway * (-1.0 if mirror else 1.0)  # in the left-leg convention the right leg's lateral offset is mirrored
        q = leg_ik(ramp * dx, dy, h0 - ramp * dz)
        out.append(-q if mirror else q)
    return np.concatenate(out)


def run(env, p: dict, seed: int, steps: int = 500, noise: float = 0.0) -> dict:
    o, _ = env.reset(seed=seed, options={"desired_goal": np.r_[[p["speed"], 0.0, 0.0], np.ones(env.n_cost)].astype(np.float32)})
    q0 = env.default_qpos.astype(np.float64)
    flex, knee = -q0[7], q0[9]  # left leg's standing hip flexion and knee bend
    h0 = float(THIGH * np.cos(flex) + SHANK * np.cos(knee - flex))  # standing ankle depth below the hip pitch axis
    alpha = env.action_filter_alpha if env.action_filter_enabled else 1.0
    filt = np.zeros(12)
    costs, vx, x0, fell = [], [], float(env.data.qpos[0]), False
    clipped = 0
    rng = np.random.default_rng(1000 + seed)
    for k in range(steps):
        u = (targets(k * env.dt, p, h0) - q0) / (env.side * env.action_scale)  # wanted filter output, policy coordinates
        a = (u - (1 - alpha) * filt) / alpha
        clipped += int(np.any(np.abs(a) > 1.0))
        a = np.clip(a, -1, 1)
        filt = alpha * a + (1 - alpha) * filt  # the pattern's own filter state; noise below is not fed back, so the walk stays the same
        sent = np.clip(a + noise * rng.standard_normal(12), -1, 1) if noise else a
        o, r, term, trunc, info = env.step(sent.astype(np.float32))
        costs.append(np.asarray(info["budget_cost"], dtype=np.float64))
        vx.append(float(o["achieved_goal"][0]))
        if term:
            fell = True
            break
    c = np.array(costs)
    after = c[50:] if len(c) > 100 else c  # leave out the ramp-in
    return {"seed": seed, "steps": len(c), "fell": fell, "distance_m": float(env.data.qpos[0] - x0), "mean_vx": float(np.mean(vx[50:])) if len(vx) > 60 else None,
            "cost_mean": after.mean(0).tolist(), "cost_p90_window100": _window(after, 100), "clipped_steps": clipped}


def _window(c: np.ndarray, w: int) -> list:
    """90th percentile over windows of w steps of the window's mean cost per step: what a bucket of window w would have to allow."""
    if len(c) < w:
        return c.mean(0).tolist()
    s = np.cumsum(np.vstack([np.zeros(c.shape[1]), c]), axis=0)
    m = (s[w:] - s[:-w]) / w
    return np.percentile(m, 90, axis=0).tolist()


GRIDS = {"full": dict(period_s=[0.6, 0.8, 1.0], duty=[0.6], speed=[0.1, 0.15, 0.2, 0.25], lift=[0.015, 0.025, 0.035], sway=[0.015, 0.025], ramp_s=[1.0]),
         "walk": dict(period_s=[0.6, 0.8, 1.0], duty=[0.6], speed=[0.1, 0.15, 0.2, 0.25], lift=[0.025, 0.035], sway=[0.025], ramp_s=[1.0])}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--realism", default="easy")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--grid", default="full", choices=sorted(GRIDS))
    args = ap.parse_args()
    cfg_path, out, realism, noise = args.config, args.out, args.realism, args.noise
    seeds = [int(s) for s in args.seeds.split(",")]
    c = yaml.safe_load(open(cfg_path))
    ec = build_env_config(c, realism, False)
    for k in ("validity", "token_budget"):
        if c.get(k):
            ec[k] = copy.deepcopy(c[k])
    ec["shaping"]["reset_velocity"] = False
    env = env_class(ec)(config=copy.deepcopy(ec))
    env.reward_mode = "track"
    grid = GRIDS[args.grid]
    results = []
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid.keys(), vals))
        rs = [run(env, p, s, noise=noise) for s in seeds]
        results.append({"params": p, "runs": rs})
        ok = sum(not r["fell"] for r in rs)
        print(json.dumps({**p, "survived": ok, "vx": [r["mean_vx"] for r in rs], "cost": [np.round(r["cost_mean"], 6).tolist() for r in rs]}), flush=True)
    Path(out).write_text(json.dumps({"config": cfg_path, "realism": realism, "seeds": seeds, "noise": noise, "grid": args.grid, "cost_names": list(env.cost_names), "results": results}, indent=1))
    env.close()


if __name__ == "__main__":
    main()
