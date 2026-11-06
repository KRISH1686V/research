"""Physically grounded evaluation for HERLocoEnv policies (v2).

Standalone successor to `evaluation_suite.py`, which is left unchanged because the
trainer callback imports it. Differences from v1:

- Contact is force-based (mj_contactForce, default 0.5 N) and sampled at every
  physics substep (200 Hz) instead of geometric contact at 50 Hz.
- Speed is displacement-based (world x and path), not the achieved-goal window.
- Clearance is the lowest foot-mesh vertex above the floor, not foot-origin z.
- Stance slip is the contact-point velocity of the loaded foot, not data.cvel
  (which is the velocity at the root subtree CoM).
- Hopping requires real flight phases and synchronous touchdowns; the gait is
  classified on the steady window only (after `steady_start_s`).
- `reset_velocity` is off by default, and pushes work in any realism when asked.
- Torque RMS, saturation, mechanical power and cost of transport are reported.
- Metrics that cannot be determined are None ("NOT DETERMINED"), never 0.

Validation (2026-10-31): with v1 settings it reproduces v1's achieved-goal vx and
torque saturation exactly; a zero-action stand classifies as "standing" with
1.2 mm/s slip; forced bilateral flight (vertical torso impulses) classifies as
"bilateral_hop"; pushes apply in easy realism. The "single_leg_hop" rule has
NOT been validated (no stable one-leg hopping control could be constructed).

Foot index 0 is the "r_foot" geom (world -y, the physical right foot). Note the
XML joint names are swapped relative to the physical side: the r_* joints drive
the +y (left) leg.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import mujoco  # noqa: E402
import BipedRobot.envs.her_loco_env as _hle  # noqa: E402
from BipedRobot.envs.her_loco_env import HERLocoEnv  # noqa: E402

PolicyFn = Callable[[Dict[str, np.ndarray]], np.ndarray]
GRAVITY = 9.81

LEG_BODIES = {
    "right": ["root", "r_upper_hip", "r_hip", "r_upper_leg", "r_lower_leg", "r_ankle", "r_foot"],
    "left": ["root", "l_upper_hip", "l_lower_hip", "l_upper_leg", "l_lower_leg", "l_ankle", "l_foot"],
}


@dataclass
class EvalSettings:
    realism: str = "easy"
    max_steps: int = 500
    reset_velocity: bool = False
    pushes: bool = False
    force_threshold_n: float = 0.5
    steady_start_s: float = 1.0
    min_flight_ms: float = 20.0
    sync_window_ms: float = 20.0


@dataclass
class Thresholds:
    """Gait-classification thresholds, recorded with every result."""
    walk_max_flight_frac: float = 0.05
    walk_max_lr_corr: float = -0.5
    walk_min_alternation: float = 0.8
    hop_min_flight_phases_per_s: float = 0.5
    hop_min_sync_frac: float = 0.5
    single_leg_max_duty: float = 0.05
    still_max_speed: float = 0.03
    still_max_touchdown_hz: float = 0.5
    min_steady_s: float = 2.0


# ----------------------------------------------------------------------------
# Environment / policy construction
# ----------------------------------------------------------------------------
def load_env_config(resolved_config: str | Path) -> Dict[str, Any]:
    import yaml
    resolved = yaml.safe_load(Path(resolved_config).read_text(encoding="utf-8"))
    return json.loads(json.dumps(resolved["resolved_env_config"]))


def make_eval_env(env_config: Dict[str, Any], settings: EvalSettings) -> HERLocoEnv:
    cfg = copy.deepcopy(env_config)
    cfg["realism"] = settings.realism
    cfg["push_enabled"] = False
    cfg.setdefault("shaping", {})["reset_velocity"] = bool(settings.reset_velocity)
    env = HERLocoEnv(config=cfg, render_mode=None)
    env.push_enabled = bool(settings.pushes)  # overrides the realism switch on purpose
    return env


def load_policy(checkpoint: str | Path, env_config: Dict[str, Any], device: str = "cpu") -> PolicyFn:
    from stable_baselines3 import SAC
    load_env = HERLocoEnv(config=copy.deepcopy(env_config), render_mode=None)
    model = SAC.load(str(checkpoint), env=load_env, device=device)  # HER needs an env
    load_env.close()
    return lambda obs: model.predict(obs, deterministic=True)[0]


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Substep recording
# ----------------------------------------------------------------------------
class _StepProxy:
    """Stands in for the env module's `mujoco` so every mj_step is observed."""

    def __init__(self, callback):
        self._callback = callback

    def __getattr__(self, name):
        return getattr(mujoco, name)

    def mj_step(self, m, d):
        mujoco.mj_step(m, d)
        self._callback()


@contextlib.contextmanager
def _record_substeps(callback):
    original = _hle.mujoco
    _hle.mujoco = _StepProxy(callback)
    try:
        yield
    finally:
        _hle.mujoco = original


def _foot_vertices(env: HERLocoEnv) -> List[np.ndarray]:
    out = []
    for gid in env.foot_geom_ids:
        mid = int(env.model.geom_dataid[gid])
        start, count = int(env.model.mesh_vertadr[mid]), int(env.model.mesh_vertnum[mid])
        out.append(env.model.mesh_vert[start:start + count].copy())
    return out


def _substep_state(env: HERLocoEnv, foot_verts: List[np.ndarray]) -> Dict[str, Any]:
    m, d = env.model, env.data
    fn = np.zeros(2)
    slip_weighted = np.zeros(2)
    nonfoot = 0.0
    f6 = np.zeros(6)
    v6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if env.floor_geom_id not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == env.floor_geom_id else c.geom1
        mujoco.mj_contactForce(m, d, i, f6)
        normal = abs(f6[0])
        if other in env.foot_geom_ids:
            k = env.foot_geom_ids.index(other)
            fn[k] += normal
            body = env.foot_body_ids[k]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, v6, 0)
            point_vel = v6[3:] + np.cross(v6[:3], c.pos - d.xpos[body])
            slip_weighted[k] += normal * float(np.linalg.norm(point_vel[:2]))
        else:
            nonfoot += normal
    clearance = np.array([
        float(np.min(d.geom_xpos[g, 2] + foot_verts[k] @ d.geom_xmat[g].reshape(3, 3)[2, :]))
        for k, g in enumerate(env.foot_geom_ids)
    ])
    return {
        "fn": fn,
        "slip": np.where(fn > 1e-9, slip_weighted / np.maximum(fn, 1e-9), 0.0),
        "nonfoot": nonfoot,
        "clear": clearance,
        "xy": d.qpos[:2].copy(),
        "z": float(d.qpos[2]),
    }


def _runs(mask: np.ndarray) -> List[tuple[int, int]]:
    """(start, end) index pairs of True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _termination_cause(env: HERLocoEnv) -> str:
    z = float(env.data.qpos[2])
    if z < env.min_torso_height:
        return "torso_low"
    if z > env.max_torso_height:
        return "torso_high"
    if env._upright_cosine() < env.min_upright_cosine:
        return "tilt"
    return "unknown"


# ----------------------------------------------------------------------------
# Episode evaluation
# ----------------------------------------------------------------------------
def run_episode(
    policy_fn: PolicyFn,
    env: HERLocoEnv,
    command: np.ndarray,
    seed: int,
    settings: EvalSettings,
    thresholds: Thresholds = Thresholds(),
    record_trace: bool = False,
) -> Dict[str, Any]:
    command = np.asarray(command, dtype=np.float32)
    foot_verts = _foot_vertices(env)
    obs, _ = env.reset(seed=seed, options={"desired_goal": command})
    reset_vx = float(env.data.qvel[0])
    start_xy = env.data.qpos[:2].copy()
    start_yaw = _yaw(env.data.qpos[3:7])
    subs: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    actions, torques, joint_vel, achieved = [], [], [], []
    body_ids = {name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for leg in LEG_BODIES.values() for name in leg}
    terminated = truncated = False
    steps = 0
    with _record_substeps(lambda: subs.append(_substep_state(env, foot_verts))):
        while not (terminated or truncated) and steps < settings.max_steps:
            action = np.asarray(policy_fn(obs), dtype=np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            torques.append(np.asarray(env.data.actuator_force[env.actuator_ids], float).copy())
            joint_vel.append(np.array([env.data.qvel[a] for a in env.joint_dof_adrs]))
            achieved.append(float(obs["achieved_goal"][0]))
            if record_trace:
                trace.append({
                    "bodies": {n: env.data.xpos[i].copy() for n, i in body_ids.items()},
                    "fn": subs[-1]["fn"].copy(),
                    "clear": subs[-1]["clear"].copy(),
                })
            steps += 1
    survived = bool(steps >= settings.max_steps and not terminated)
    m_total = float(np.sum(env.model.body_mass))
    weight = m_total * GRAVITY
    dt, sdt = env.dt, env.sim_dt
    duration = steps * dt

    fn = np.array([s["fn"] for s in subs])
    slip = np.array([s["slip"] for s in subs])
    clear = np.array([s["clear"] for s in subs])
    nonfoot = np.array([s["nonfoot"] for s in subs])
    xy = np.array([s["xy"] for s in subs])
    z = np.array([s["z"] for s in subs])
    tau = np.array(torques)
    qd = np.array(joint_vel)
    acts = np.array(actions)

    s0 = int(round(settings.steady_start_s / sdt))
    steady = slice(s0, None)
    steady_s = max(0.0, (len(subs) - s0) * sdt)
    loaded = fn > settings.force_threshold_n
    n_loaded = loaded.sum(1)
    flight = (n_loaded == 0) & (nonfoot <= settings.force_threshold_n)

    # Steady-window gait statistics
    ld, fl = loaded[steady], flight[steady]
    min_flight = max(1, int(round(settings.min_flight_ms / 1e3 / sdt)))
    sync_win = max(1, int(round(settings.sync_window_ms / 1e3 / sdt)))
    touchdowns = []
    for k in range(2):
        onsets = np.flatnonzero(ld[1:, k] & ~ld[:-1, k]) + 1
        touchdowns += [(int(i), k) for i in onsets if i >= 2 and not ld[i - 2:i, k].any()]
    touchdowns.sort()
    seq = [k for _, k in touchdowns]
    by_foot = {k: np.array([i for i, f in touchdowns if f == k]) for k in (0, 1)}
    sync = sum(1 for i, k in touchdowns if by_foot[1 - k].size and np.min(np.abs(by_foot[1 - k] - i)) <= sync_win)
    flight_runs = [e - s for s, e in _runs(fl)]
    long_flights = [r for r in flight_runs if r >= min_flight]
    lr_corr = None
    if ld.shape[0] > 10 and ld[:, 0].std() > 0 and ld[:, 1].std() > 0:
        lr_corr = float(np.corrcoef(ld[:, 0].astype(float), ld[:, 1].astype(float))[0, 1])
    alternation = float(np.mean([seq[i] != seq[i - 1] for i in range(1, len(seq))])) if len(seq) > 2 else None

    swing_peaks = []
    for k in range(2):
        for start, end in _runs(~ld[:, k]):
            if end - start >= int(round(0.04 / sdt)) and end < len(ld):
                swing_peaks.append(float(clear[steady][start:end, k].max()))

    stance_slip = slip[steady][ld]
    heavy = fn[steady] > 0.3 * weight
    dx = float(xy[-1, 0] - start_xy[0]) if len(xy) else 0.0
    path = float(np.sum(np.linalg.norm(np.diff(np.vstack([start_xy, xy]), axis=0), axis=1))) if len(xy) else 0.0
    dx_steady = float(xy[-1, 0] - xy[s0 - 1, 0]) if len(xy) > s0 else None
    power = np.abs(tau * qd).sum(1) if len(tau) else np.zeros(0)
    duty = [float(ld[:, 0].mean()), float(ld[:, 1].mean())] if len(ld) else [None, None]
    td_rate = len(touchdowns) / steady_s if steady_s > 0 else None
    speed_steady = dx_steady / steady_s if dx_steady is not None and steady_s > 0 else None

    def nd(compute, ok=True):
        """Evaluate lazily; None means NOT DETERMINED."""
        return compute() if ok else None

    metrics = {
        "seed": int(seed),
        "command": command.tolist(),
        "survived": survived,
        "steps": steps,
        "termination_cause": None if survived else _termination_cause(env),
        "reset_vx": reset_vx,
        "pushes_applied": int(sum(1 for t in range(1, steps) if t % env.dr_push_interval == 0)) if env.push_enabled else 0,
        # speed
        "speed_x_disp": dx / duration if duration > 0 else None,
        "speed_x_disp_steady": speed_steady,
        "speed_path": path / duration if duration > 0 else None,
        "speed_achieved_goal_mean": float(np.mean(achieved)) if achieved else None,
        "rmse_vx_achieved_goal": float(np.sqrt(np.mean((np.array(achieved) - command[0]) ** 2))) if achieved else None,
        "lateral_drift_m": float(xy[-1, 1] - start_xy[1]) if len(xy) else None,
        "heading_change_rad": float(np.angle(np.exp(1j * (_yaw(env.data.qpos[3:7]) - start_yaw)))),
        # body
        "torso_z_mean": float(z.mean()) if len(z) else None,
        "torso_z_std": float(z.std()) if len(z) else None,
        # contact and gait (steady window)
        "steady_window_s": steady_s,
        "duty_factor": duty,
        "flight_frac": nd(lambda: float(fl.mean()), len(fl) > 0),
        "double_support_frac": nd(lambda: float((n_loaded[steady] == 2).mean()), len(fl) > 0),
        "flight_phases_per_s": nd(lambda: len(long_flights) / steady_s, steady_s > 0),
        "flight_ms_max": nd(lambda: float(max(flight_runs) * sdt * 1e3) if flight_runs else 0.0, steady_s > 0),
        "touchdown_hz": td_rate,
        "sync_touchdown_frac": nd(lambda: sync / len(touchdowns), len(touchdowns) > 0),
        "alternation": alternation,
        "lr_load_corr": lr_corr,
        "nonfoot_floor_contact_frac": nd(lambda: float((nonfoot[steady] > settings.force_threshold_n).mean()), len(fl) > 0),
        "load_over_weight": nd(lambda: float(fn[steady].sum(1).mean() / weight), len(fl) > 0),
        "swing_clearance_mm_median": nd(lambda: float(np.median(swing_peaks) * 1e3) if swing_peaks else None, bool(swing_peaks)),
        "n_swings": len(swing_peaks),
        "stance_slip_mps_mean": nd(lambda: float(stance_slip.mean()) if stance_slip.size else None, stance_slip.size > 0),
        "stance_slip_mps_p95": nd(lambda: float(np.percentile(stance_slip, 95)) if stance_slip.size else None, stance_slip.size > 0),
        "loaded_slip_mps_mean": nd(lambda: float(slip[steady][heavy].mean()) if heavy.any() else None, bool(heavy.any())),
        "slip_m_per_m_travel": nd(lambda: float((slip[steady] * ld).sum() * sdt / abs(dx_steady)) if dx_steady else None,
                                  dx_steady is not None and abs(dx_steady) > 0.05),
        # effort
        "torque_rms_nm": nd(lambda: float(np.sqrt(np.mean(tau ** 2))), len(tau) > 0),
        "torque_sat_frac": nd(lambda: float(np.mean(np.abs(tau) >= 0.98 * env.torque_limit)), len(tau) > 0),
        "torque_sat_per_joint": np.mean(np.abs(tau) >= 0.98 * env.torque_limit, 0).round(4).tolist() if len(tau) else None,
        "action_sat_frac": nd(lambda: float(np.mean(np.abs(acts) > 0.95)), len(acts) > 0),
        "action_rate_rms": nd(lambda: float(np.sqrt(np.mean(np.diff(acts, axis=0) ** 2))), len(acts) > 1),
        "mech_power_w_mean": nd(lambda: float(power.mean()), len(power) > 0),
        "cost_of_transport": nd(lambda: float(power.sum() * dt / (weight * abs(dx))), abs(dx) > 0.05),
        "mass_kg": m_total,
    }
    metrics["gait_class"] = classify_gait(metrics, command, thresholds)
    if record_trace:
        metrics["_trace"] = trace
    return metrics


def classify_gait(m: Dict[str, Any], command: np.ndarray, t: Thresholds) -> str:
    """Physical gait label from the steady window. See `Thresholds`."""
    if m["steady_window_s"] < t.min_steady_s or m["flight_frac"] is None:
        return "not_determined"
    moving_cmd = float(np.linalg.norm(command[:2])) > 0.1
    duty = m["duty_factor"]
    fps = m["flight_phases_per_s"] or 0.0
    sync = m["sync_touchdown_frac"] or 0.0
    if min(duty) < t.single_leg_max_duty and (m["touchdown_hz"] or 0) > 1.0:
        return "single_leg_hop"
    if fps >= t.hop_min_flight_phases_per_s and sync >= t.hop_min_sync_frac:
        return "bilateral_hop"
    if fps >= t.hop_min_flight_phases_per_s:
        return "running_or_bounding"
    speed = abs(m["speed_x_disp_steady"] or 0.0)
    if (m["touchdown_hz"] or 0) <= t.still_max_touchdown_hz and speed < t.still_max_speed:
        return "standing"
    if (
        m["flight_frac"] < t.walk_max_flight_frac
        and m["lr_load_corr"] is not None and m["lr_load_corr"] <= t.walk_max_lr_corr
        and m["alternation"] is not None and m["alternation"] >= t.walk_min_alternation
    ):
        return "alternating_walk" if (moving_cmd or speed >= t.still_max_speed) else "stepping_in_place"
    return "irregular"


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> List[Optional[float]]:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def bootstrap(values: List[Any], n_boot: int = 10000, seed: int = 20261031) -> Dict[str, Any]:
    x = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if x.size == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, x.size)).mean(1)
    return {"mean": float(x.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "n": int(x.size)}


SCALAR_METRICS = [
    "steps", "speed_x_disp", "speed_x_disp_steady", "speed_path", "speed_achieved_goal_mean",
    "rmse_vx_achieved_goal", "lateral_drift_m", "heading_change_rad", "torso_z_mean", "torso_z_std",
    "flight_frac", "double_support_frac", "flight_phases_per_s", "touchdown_hz", "sync_touchdown_frac",
    "alternation", "lr_load_corr", "nonfoot_floor_contact_frac", "load_over_weight",
    "swing_clearance_mm_median", "stance_slip_mps_mean", "stance_slip_mps_p95", "loaded_slip_mps_mean",
    "slip_m_per_m_travel", "torque_rms_nm", "torque_sat_frac", "action_sat_frac", "action_rate_rms",
    "mech_power_w_mean", "cost_of_transport",
]


def aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(episodes)
    k = sum(e["survived"] for e in episodes)
    survivors = [e for e in episodes if e["survived"]]
    classes: Dict[str, int] = {}
    for e in episodes:
        classes[e["gait_class"]] = classes.get(e["gait_class"], 0) + 1
    causes: Dict[str, int] = {}
    for e in episodes:
        if e["termination_cause"]:
            causes[e["termination_cause"]] = causes.get(e["termination_cause"], 0) + 1
    return {
        "n": n,
        "survival": k,
        "survival_rate": k / n if n else None,
        "survival_wilson95": wilson(k, n),
        "termination_causes": causes,
        "gait_classes": classes,
        "all_episodes": {m: bootstrap([e[m] for e in episodes]) for m in SCALAR_METRICS},
        "survivors_only": {m: bootstrap([e[m] for e in survivors]) for m in SCALAR_METRICS},
    }


# ----------------------------------------------------------------------------
# Visual trace (no OpenGL needed): side + front stick figure with load strip.
# Lines join body origins from "root" (the "torso" body origin sits near the
# floor in this XML) down each leg; filled foot circle = loaded (> 0.5 N).
# ----------------------------------------------------------------------------
def render_trace_gif(trace: List[Dict[str, Any]], path: str | Path, title: str = "", every: int = 2,
                     weight_n: float = 14.4) -> None:
    from PIL import Image, ImageDraw
    W, H, scale = 640, 360, 600.0  # px per metre
    colours = {"right": (220, 60, 60), "left": (40, 110, 220)}
    frames = []
    hist = [t["fn"] for t in trace]
    for idx in range(0, len(trace), every):
        t = trace[idx]
        img = Image.new("RGB", (W, H), (250, 250, 250))
        dr = ImageDraw.Draw(img)
        cx = float(t["bodies"]["root"][0])
        cy = float(t["bodies"]["root"][1])
        floor_y = 270
        for panel, (ox, axis, centre, label) in enumerate([(160, 0, cx, "side (x-z)"), (480, 1, cy, "front (y-z)")]):
            dr.line([(ox - 150, floor_y), (ox + 150, floor_y)], fill=(120, 120, 120), width=2)
            dr.text((ox - 150, 8), label, fill=(0, 0, 0))
            for leg, names in LEG_BODIES.items():
                pts = [(ox + (float(t["bodies"][nm][axis]) - centre) * scale * (1 if axis == 0 else -1),
                        floor_y - float(t["bodies"][nm][2]) * scale) for nm in names]
                dr.line(pts, fill=colours[leg], width=4)
                foot = pts[-1]
                k = 0 if leg == "right" else 1
                loaded = t["fn"][k] > 0.5
                r = 6
                dr.ellipse([foot[0] - r, foot[1] - r, foot[0] + r, foot[1] + r],
                           fill=colours[leg] if loaded else (255, 255, 255), outline=colours[leg], width=2)
        # load strip: last 2 s of per-foot normal force
        x0, y0, sw = 20, 300, 600
        window = hist[max(0, idx - 100):idx + 1]
        for k, leg in enumerate(("right", "left")):
            yb = y0 + 22 * k
            dr.text((x0, yb - 2), leg[0].upper(), fill=colours[leg])
            for j, f in enumerate(window):
                h = min(18.0, 18.0 * f[k] / weight_n)
                xx = x0 + 15 + j * (sw - 15) / 100
                dr.line([(xx, yb + 18), (xx, yb + 18 - h)], fill=colours[leg], width=3)
        dr.text((20, 30), f"{title}  t={idx * 0.02:.2f}s  clear R {t['clear'][0] * 1e3:.0f}mm L {t['clear'][1] * 1e3:.0f}mm",
                fill=(0, 0, 0))
        frames.append(img)
    if frames:
        frames[0].save(str(path), save_all=True, append_images=frames[1:], duration=int(20 * every), loop=0)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def evaluate(checkpoint: str, resolved_config: str, command: List[float], seeds: List[int],
             settings: EvalSettings, gif_dir: Optional[str] = None, n_gifs: int = 0,
             label: str = "") -> Dict[str, Any]:
    env_config = load_env_config(resolved_config)
    policy = load_policy(checkpoint, env_config)
    env = make_eval_env(env_config, settings)
    thresholds = Thresholds()
    episodes = []
    for i, seed in enumerate(seeds):
        ep = run_episode(policy, env, np.asarray(command, np.float32), seed, settings, thresholds,
                         record_trace=bool(gif_dir) and i < n_gifs)
        trace = ep.pop("_trace", None)
        if trace:
            Path(gif_dir).mkdir(parents=True, exist_ok=True)
            render_trace_gif(trace, Path(gif_dir) / f"{label}_{settings.realism}_cmd{command[0]:.2f}_seed{seed}.gif",
                             title=f"{label} {settings.realism} cmd {command[0]:.2f} seed {seed} [{ep['gait_class']}]",
                             weight_n=ep["mass_kg"] * GRAVITY)
        episodes.append(ep)
    env.close()
    return {
        "provenance": {
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "resolved_config": str(resolved_config), "evaluator": "evaluation_suite_v2",
            "evaluator_sha256": sha256(__file__), "repo_commit": _git_commit(),
        },
        "settings": asdict(settings),
        "thresholds": asdict(thresholds),
        "command": list(command),
        "seeds": list(seeds),
        "aggregate": aggregate(episodes),
        "episodes": episodes,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--resolved-config", required=True)
    p.add_argument("--realism", choices=["easy", "medium", "full"], default="easy")
    p.add_argument("--command", type=float, nargs=3, default=[0.2, 0.0, 0.0])
    p.add_argument("--seeds", default="5000:5030", help="start:stop (stop exclusive)")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--reset-velocity", action="store_true", help="keep the training-time reset kick")
    p.add_argument("--pushes", action="store_true", help="apply the configured pushes in any realism")
    p.add_argument("--gif-dir", default=None)
    p.add_argument("--n-gifs", type=int, default=0)
    p.add_argument("--label", default="policy")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    s0, s1 = map(int, a.seeds.split(":"))
    settings = EvalSettings(realism=a.realism, max_steps=a.max_steps,
                            reset_velocity=a.reset_velocity, pushes=a.pushes)
    result = evaluate(a.checkpoint, a.resolved_config, a.command, list(range(s0, s1)), settings,
                      gif_dir=a.gif_dir, n_gifs=a.n_gifs, label=a.label)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
