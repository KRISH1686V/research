"""DR-off single-leg-lift ablation; probe only, no training or controller edits."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from BipedRobot.envs.her_loco_env import HERLocoEnv

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml"


def probe_ik(env, desired, fraction):
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    scratch.qvel[:] = env.data.qvel
    mujoco.mj_forward(env.model, scratch)
    torso = env.torso_body_id
    rotation = scratch.xmat[torso].reshape(3, 3)
    origin = scratch.xpos[torso].copy()
    chains = ((6, 7, 8, 9, 10, 11), (0, 1, 2, 3, 4, 5))
    for leg, indices in enumerate(chains):
        target = origin + rotation @ desired[leg]
        for _ in range(10):
            mujoco.mj_forward(env.model, scratch)
            error = target - scratch.xpos[env.foot_body_ids[leg]]
            jacp = np.zeros((3, env.model.nv))
            jacr = np.zeros((3, env.model.nv))
            mujoco.mj_jacBody(env.model, scratch, jacp, jacr, env.foot_body_ids[leg])
            cols = [env.joint_dof_adrs[i] for i in indices]
            jac = jacp[:, cols]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.03**2 * np.eye(3), error)
            for increment, joint in zip(delta, indices):
                adr = env.joint_qpos_adrs[joint]
                scratch.qpos[adr] = np.clip(
                    scratch.qpos[adr] + increment, env.jnt_min[joint], env.jnt_max[joint]
                )
    targets = np.asarray([scratch.qpos[adr] for adr in env.joint_qpos_adrs])
    raw = np.zeros(env.num_joints, dtype=np.float32)
    for i, target in enumerate(targets):
        scale = env.jnt_max[i] - env.default_qpos[i] if target >= env.default_qpos[i] else env.default_qpos[i] - env.jnt_min[i]
        raw[i] = (target - env.default_qpos[i]) / max(scale * fraction, 1e-6)
    return np.clip(raw, -1.0, 1.0).astype(np.float32), raw


def roll_value(env):
    gravity = env._get_projected_gravity()
    return float(np.arctan2(gravity[1], -gravity[2]))


def trial(raw_config, variant, leg, shift, feedback, seed):
    config = copy.deepcopy(raw_config)
    config["domain_randomization"]["enabled"] = False
    fraction = 0.75 if variant["range075"] else 0.50
    if variant["actuators_off"]:
        config["actuator"].update(
            enable_delay=False, enable_lag=False, enable_backlash=False
        )
    env = HERLocoEnv(config=config, render_mode=None)
    env.action_range_fraction = fraction
    env.reset(seed=seed)
    support = 1 - leg
    clearances, shifts, action_sat = [], [], []
    torque_sat = np.zeros(env.num_joints, dtype=np.int64)
    upright = True
    onset = None
    for step in range(120):
        feet = [
            env.data.xmat[env.torso_body_id].reshape(3, 3).T
            @ (env.data.xpos[env.foot_body_ids[i]] - env.data.xpos[env.torso_body_id])
            for i in range(2)
        ]
        if step < 50:
            amount = shift * (step + 1) / 50.0
            feet[leg][1] += amount
            feet[support][1] -= amount
        elif step < 75:
            feet[leg][2] += 0.04
        elif step < 100:
            feet[leg][2] += 0.04 * (100 - step) / 25.0
        if feedback:
            correction = np.clip(-0.25 * roll_value(env), -0.02, 0.02)
            feet[0][1] += correction
            feet[1][1] += correction
        action, raw = probe_ik(env, feet, fraction)
        _, _, terminated, truncated, info = env.step(action)
        if step == 50:
            com = float(env.data.subtree_com[env.root_body_id, 1])
            support_center = float(env.data.xpos[env.foot_body_ids[support], 1])
            geom_id = env.foot_geom_ids[support]
            foot_center = float(env.data.geom_xpos[geom_id, 1])
            half_width = float(env.model.geom_size[geom_id, 1])
            onset = {
                "com_relative_to_support_center_m": com - support_center,
                "com_relative_to_foot_center_m": com - foot_center,
                "foot_edges_relative_to_support_center_m": [
                    foot_center - half_width - support_center,
                    foot_center + half_width - support_center,
                ],
            }
        clearances.append(float(env.data.xpos[env.foot_body_ids[leg], 2]))
        shifts.append(abs(float(env.data.subtree_com[env.root_body_id, 1] - env.data.xpos[env.foot_body_ids[support], 1])))
        action_sat.append(float(np.mean(np.abs(raw) > 0.95)))
        forces = np.abs(np.asarray(env.data.actuator_force)[env.actuator_ids])
        torque_sat += (forces >= env.torque_limit * 0.999).astype(np.int64)
        upright &= not terminated
        if terminated or truncated:
            break
    hip_spacing = abs(
        float(env.data.xpos[env.foot_body_ids[0], 1] - env.data.xpos[env.foot_body_ids[1], 1])
    )
    foot_width = float(2 * env.model.geom_size[env.foot_geom_ids[0], 1])
    env.close()
    return {
        "variant": variant["name"],
        "leg": "right" if leg == 0 else "left",
        "feedback": feedback,
        "requested_shift_m": shift,
        "lift_onset": onset,
        "shift_distance_m": float(max(shifts) if shifts else 0.0),
        "hip_spacing_m": hip_spacing,
        "foot_width_m": foot_width,
        "achieved_clearance_m": float(max(clearances) if clearances else 0.0),
        "upright": bool(upright),
        "steps_survived": len(clearances),
        "action_saturation_fraction": float(np.mean(action_sat) if action_sat else 0.0),
        "per_joint_torque_saturation_fraction": (torque_sat / max(1, len(clearances))).tolist(),
    }


def mapping_symmetry(raw_config):
    config = copy.deepcopy(raw_config)
    config["domain_randomization"]["enabled"] = False
    env = HERLocoEnv(config=config, render_mode=None)
    env.reset(seed=8000)
    out = {}
    for axis, indices in {"pitch": (1, 7), "lateral": (2, 8)}.items():
        values = []
        for index in indices:
            row = []
            for sign in (-1.0, 1.0):
                action = np.zeros(12, dtype=np.float32)
                action[index] = sign
                foot_index = 1 if index < 6 else 0
                before = env.data.xpos[env.foot_body_ids[foot_index]].copy()
                env.step(action)
                after = env.data.xpos[env.foot_body_ids[foot_index]].copy()
                row.append((after - before).tolist())
                env.reset(seed=8000)
            values.append(row)
        out[axis] = values
    env.close()
    return out


def main():
    with BASE.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    variants = [
        {"name": "current", "actuators_off": False, "range075": False},
        {"name": "actuators_off", "actuators_off": True, "range075": False},
        {"name": "range075", "actuators_off": False, "range075": True},
        {"name": "both", "actuators_off": True, "range075": True},
    ]
    results = []
    for variant in variants:
        for leg in (0, 1):
            for shift in (0.02, 0.04, 0.06):
                for feedback in (False, True):
                    results.append(trial(raw, variant, leg, shift, feedback, 7100 + len(results)))
    output = {
        "mapping_symmetry": mapping_symmetry(raw),
        "variants": results,
        "interpretation": {
            "current": "pipeline can step only if both legs hold 0.5 s",
            "actuators_off": "works only with lag off means lag is limiter",
            "all_fail": "morphology or ankle authority is limiting",
        },
    }
    destination = ROOT / "BipedRobot/paper_results/feasibility_ablation.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
