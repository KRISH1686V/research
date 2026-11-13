"""Corrected DR-off lift probe using mesh vertices and contact-force splits."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.reference.lift_diagnosis import ik, local_feet

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "BipedRobot/config/pilot_gait_smooth_a_v2.yaml"


def mesh_extent(env, geom_id):
    geom = env.model.geom(geom_id)
    mesh_id = int(np.asarray(geom.dataid).reshape(-1)[0])
    vert_start = int(env.model.mesh_vertadr[mesh_id])
    vert_count = int(env.model.mesh_vertnum[mesh_id])
    vertices = env.model.mesh_vert[vert_start : vert_start + vert_count]
    scale = np.asarray(geom.size)
    points = vertices * scale
    return {
        "lateral_m": [float(points[:, 1].min()), float(points[:, 1].max())],
        "sagittal_m": [float(points[:, 0].min()), float(points[:, 0].max())],
    }


def foot_force(env, geom_id):
    total = 0.0
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        if env.floor_geom_id in (contact.geom1, contact.geom2) and geom_id in (
            contact.geom1,
            contact.geom2,
        ):
            force = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, i, force)
            total += abs(float(force[0]))
    return total


def run_leg(config, leg):
    env = HERLocoEnv(config=copy.deepcopy(config), render_mode=None)
    env.reset(seed=8800 + leg)
    support = 1 - leg
    body_weight = float(np.sum(env.model.body_mass) * 9.81)
    records = []
    outcome = {"leg": leg, "body_weight_N": body_weight, "targets": []}
    for target in (0.01, 0.02, 0.04):
        env.reset(seed=8800 + leg)
        fell = None
        highest = 0.0
        hold_steps = 0
        for step in range(210):
            feet = local_feet(env)
            shift = 0.06 * min(1.0, (step + 1) / 100.0)
            feet[leg][1] += shift
            feet[support][1] -= shift
            if step >= 100:
                feet[leg][2] += target * min(1.0, (step - 99) / 100.0)
            action, raw_action, desired = ik(env, feet, 0.5)
            actual = np.asarray([env.data.qpos[a] for a in env.joint_qpos_adrs])
            _, _, terminated, truncated, _ = env.step(action)
            stance_force = foot_force(env, env.foot_geom_ids[support])
            swing_force = foot_force(env, env.foot_geom_ids[leg])
            clearance = float(env.data.xpos[env.foot_body_ids[leg], 2])
            highest = max(highest, clearance)
            if step >= 150 and clearance >= target * 0.9 and not terminated and not truncated:
                hold_steps += 1
            records.append({
                "leg": leg, "target_m": target, "step": step,
                "phase": "shift" if step < 100 else "lift",
                "stance_force_N": stance_force, "swing_force_N": swing_force,
                "total_force_N": stance_force + swing_force,
                "body_weight_N": body_weight, "clearance_m": clearance,
                "servo_error_rad": (desired - actual).tolist(),
                "torque_saturation_fraction": float(np.mean(np.abs(env.data.actuator_force[env.actuator_ids]) >= env.torque_limit * 0.999)),
            })
            if terminated or truncated:
                fell = {"step": step, "direction": "lateral" if abs(float(env._get_projected_gravity()[1])) > abs(float(env._get_projected_gravity()[0])) else "sagittal"}
                break
        outcome["targets"].append({
            "target_cm": target * 100,
            "highest_clearance_cm": highest * 100,
            "held_0_5s": hold_steps >= 25,
            "fall": fell,
        })
    env.close()
    return outcome, records


def main():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    env = HERLocoEnv(config=copy.deepcopy(config), render_mode=None)
    extents = {str(i): mesh_extent(env, gid) for i, gid in enumerate(env.foot_geom_ids)}
    env.close()
    outcomes, records = [], []
    for leg in (0, 1):
        outcome, rows = run_leg(config, leg)
        outcomes.append(outcome)
        records.extend(rows)
    output = {"foot_mesh_extents": extents, "outcomes": outcomes, "per_step": records}
    path = ROOT / "BipedRobot/paper_results/lift_probe_v2.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"foot_mesh_extents": extents, "outcomes": outcomes}, indent=2))


if __name__ == "__main__":
    main()
