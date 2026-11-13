"""Offline Stage 0 diagnostics for the HERLocoEnv walking reference."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.train_her_biped import build_env_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml"
JOINT_NAMES = [
    "r_hip_z", "r_hip_x", "r_hip_y", "r_knee", "r_ankle_y", "r_ankle_x",
    "l_hip_z", "l_hip_x", "l_hip_y", "l_knee", "l_ankle_y", "l_ankle_x",
]


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def config_without_dr(raw: dict) -> dict:
    config = build_env_config(raw, "easy", False)
    config["domain_randomization"] = {
        **config.get("domain_randomization", {}),
        "enabled": False,
    }
    config["push_enabled"] = False
    return config


def model_geometry(env: HERLocoEnv) -> dict:
    model = env.model
    body_masses = model.body_mass.copy()
    total_mass = float(np.sum(body_masses))
    qpos = env.data.qpos.copy()
    com_height = float(env.data.subtree_com[env.root_body_id, 2])

    mesh_bounds = {}
    for geom_id in env.foot_geom_ids:
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start : start + count]
        scale = model.mesh_scale[mesh_id]
        extent = vertices.ptp(axis=0) if len(vertices) else np.zeros(3)
        mesh_bounds[str(model.geom(geom_id).name)] = extent.tolist()

    joint_limits = {
        name: [float(lo), float(hi)]
        for name, lo, hi in zip(
            JOINT_NAMES,
            env.jnt_min,
            env.jnt_max,
        )
    }
    if not joint_limits:
        joint_limits = {
            str(i): [float(lo), float(hi)]
            for i, (lo, hi) in enumerate(zip(env.jnt_min, env.jnt_max))
        }
    hip_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("l_lower_hip", "r_hip")
    ]
    foot_positions = np.asarray(
        [model.body_pos[body_id] for body_id in hip_body_ids], dtype=np.float64
    )
    leg_length = float(
        np.linalg.norm(
            env.data.xpos[env.foot_body_ids[0]]
            - env.data.xpos[hip_body_ids[0]]
        )
    )
    plus = env._action_to_target(np.ones(env.num_joints, dtype=np.float32))
    minus = env._action_to_target(-np.ones(env.num_joints, dtype=np.float32))
    hip_pitch_indices = (2, 8)
    hip_pitch_swing = [
        float(plus[i] - minus[i]) for i in hip_pitch_indices
    ]
    max_reachable_step = float(2.0 * leg_length)
    return {
        "total_mass_kg": total_mass,
        "com_height_m": com_height,
        "foot_mesh_extents_m": mesh_bounds,
        "joint_limits_rad": joint_limits,
        "torque_limit_nm": float(env.torque_limit),
        "action_range_fraction": float(env.action_range_fraction),
        "leg_length_m": leg_length,
        "hip_pitch_swing_range_rad": hip_pitch_swing,
        "max_reachable_step_length_m": max_reachable_step,
        "initial_hip_positions": foot_positions.tolist(),
        "qpos_root": qpos[:7].tolist(),
    }


def action_mapping(env: HERLocoEnv) -> dict:
    zero = env.default_qpos.copy()
    plus = env._action_to_target(np.ones(env.num_joints, dtype=np.float32))
    minus = env._action_to_target(-np.ones(env.num_joints, dtype=np.float32))
    return {
        "zero_target_rad": zero.tolist(),
        "minus_one_target_rad": minus.tolist(),
        "plus_one_target_rad": plus.tolist(),
    }


def actuator_response(config: dict) -> dict:
    results = []
    for frequency in (0.5, 1.0, 1.5, 2.0, 3.0):
        for amplitude in (0.5, 1.0):
            env = HERLocoEnv(config=copy.deepcopy(config))
            obs, _ = env.reset(seed=123, options={"desired_goal": [0.0, 0.0, 0.0]})
            values = []
            times = []
            for step in range(500):
                phase = 2.0 * np.pi * frequency * step * env.dt
                action = np.zeros(12, dtype=np.float32)
                action[2] = amplitude * np.sin(phase)
                env.step(action)
                values.append(float(env.data.qpos[env.joint_qpos_adrs[2]]))
                times.append(step * env.dt)
            env.close()
            values = np.asarray(values)
            times = np.asarray(times)
            keep = times >= 2.0
            signal = np.sin(2.0 * np.pi * frequency * times[keep])
            centered = values[keep] - np.mean(values[keep])
            sin_proj = 2.0 * np.mean(centered * signal)
            cos_proj = 2.0 * np.mean(
                centered * np.cos(2.0 * np.pi * frequency * times[keep])
            )
            achieved_amplitude = float(np.hypot(sin_proj, cos_proj))
            phase_lag = float(np.arctan2(cos_proj, sin_proj))
            results.append(
                {
                    "frequency_hz": frequency,
                    "action_amplitude": amplitude,
                    "achieved_joint_amplitude_rad": achieved_amplitude,
                    "phase_lag_rad": phase_lag,
                    "phase_lag_ms": phase_lag / (2.0 * np.pi * frequency) * 1000.0,
                }
            )
    return {"hip_pitch_joint_index": 2, "results": results}


def standing_recovery(config: dict) -> dict:
    results = []
    for push_n in (0.0, 0.1, 0.5, 1.0):
        env = HERLocoEnv(config=copy.deepcopy(config))
        env.reset(seed=321, options={"desired_goal": [0.0, 0.0, 0.0]})
        max_tilt = 0.0
        fell = False
        for step in range(500):
            if push_n and step == 50:
                env.data.xfrc_applied[env.torso_body_id, 0] = push_n
            else:
                env.data.xfrc_applied[env.torso_body_id, :3] = 0.0
            _, _, terminated, truncated, _ = env.step(np.zeros(12, dtype=np.float32))
            gravity = env._get_projected_gravity()
            tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
            max_tilt = max(max_tilt, tilt)
            fell |= bool(terminated)
            if terminated or truncated:
                break
        results.append(
            {
                "push_force_n": push_n,
                "steps": step + 1,
                "survived": not fell and step + 1 == 500,
                "max_tilt_rad": max_tilt,
            }
        )
        env.close()
    return {"results": results}


def feasibility(leg_length: float) -> dict:
    rows = []
    for command in (0.2, 0.4, 0.6):
        for frequency in (1.5, 2.0, 2.5):
            step_length = command / frequency
            rows.append(
                {
                    "command_vx": command,
                    "steps_per_second": frequency,
                    "required_step_length_m": step_length,
                    "max_reachable_step_length_m": 2.0 * leg_length,
                    "geometrically_feasible": step_length <= 2.0 * leg_length,
                    "froude_number": command * command / (9.81 * leg_length),
                }
            )
    return {"rows": rows}


def throughput(config: dict) -> dict:
    env = HERLocoEnv(config=copy.deepcopy(config))
    env.reset(seed=456, options={"desired_goal": [0.2, 0.0, 0.0]})
    started = time.perf_counter()
    steps = 2000
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(np.zeros(12, dtype=np.float32))
        if terminated or truncated:
            env.reset(seed=456, options={"desired_goal": [0.2, 0.0, 0.0]})
    elapsed = time.perf_counter() - started
    env.close()
    return {"steps": steps, "elapsed_s": elapsed, "steps_per_second": steps / elapsed}


def main() -> None:
    with CONFIG_PATH.open() as handle:
        raw = yaml.safe_load(handle)
    config = config_without_dr(raw)
    env = HERLocoEnv(config=copy.deepcopy(config))
    env.reset(seed=0, options={"desired_goal": [0.2, 0.0, 0.0]})
    output = {
        "git_commit": git_hash(),
        "config_path": str(CONFIG_PATH),
        "config": config,
        "model_geometry": model_geometry(env),
        "action_mapping": action_mapping(env),
        "actuator_response": actuator_response(config),
        "standing_recovery": standing_recovery(config),
        "feasibility": feasibility(model_geometry(env)["leg_length_m"]),
        "single_process_throughput": throughput(config),
    }
    env.close()
    destination = ROOT / "BipedRobot/paper_results/reference_stage0.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
