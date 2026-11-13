"""Detailed DR-off single-support diagnosis; does not alter project configs."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml"

from BipedRobot.envs.her_loco_env import HERLocoEnv


def local_feet(env):
    rot = env.data.xmat[env.torso_body_id].reshape(3, 3)
    origin = env.data.xpos[env.torso_body_id]
    return [rot.T @ (env.data.xpos[b] - origin) for b in env.foot_body_ids]


def ik(env, desired, fraction):
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    scratch.qvel[:] = env.data.qvel
    mujoco.mj_forward(env.model, scratch)
    rot = scratch.xmat[env.torso_body_id].reshape(3, 3)
    origin = scratch.xpos[env.torso_body_id]
    chains = ((6, 7, 8, 9, 10, 11), (0, 1, 2, 3, 4, 5))
    for leg, indices in enumerate(chains):
        target = origin + rot @ desired[leg]
        for _ in range(10):
            mujoco.mj_forward(env.model, scratch)
            err = target - scratch.xpos[env.foot_body_ids[leg]]
            jp = np.zeros((3, env.model.nv)); jr = np.zeros((3, env.model.nv))
            mujoco.mj_jacBody(env.model, scratch, jp, jr, env.foot_body_ids[leg])
            jac = jp[:, [env.joint_dof_adrs[i] for i in indices]]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.03**2 * np.eye(3), err)
            for inc, i in zip(delta, indices):
                adr = env.joint_qpos_adrs[i]
                scratch.qpos[adr] = np.clip(scratch.qpos[adr] + inc, env.jnt_min[i], env.jnt_max[i])
    target_angles = np.asarray([scratch.qpos[a] for a in env.joint_qpos_adrs])
    raw = np.zeros(env.num_joints, dtype=np.float32)
    for i, target in enumerate(target_angles):
        scale = env.jnt_max[i] - env.default_qpos[i] if target >= env.default_qpos[i] else env.default_qpos[i] - env.jnt_min[i]
        raw[i] = (target - env.default_qpos[i]) / max(scale * fraction, 1e-6)
    return np.clip(raw, -1, 1).astype(np.float32), raw, target_angles


def contact_force(env, geom_id):
    force = 0.0
    for index in range(env.data.ncon):
        c = env.data.contact[index]
        if {c.geom1, c.geom2} == {geom_id, env.floor_geom_id}:
            six = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, index, six)
            force += abs(float(six[0]))
    return force


def run_variant(raw, actuator_off):
    config = copy.deepcopy(raw)
    config.setdefault("domain_randomization", {})["enabled"] = False
    if actuator_off:
        config["actuator"].update(enable_delay=False, enable_lag=False, enable_backlash=False)
    rows = []
    summary = []
    for leg in (0, 1):
        env = HERLocoEnv(config=config, render_mode=None)
        env.reset(seed=7200 + leg)
        support = 1 - leg
        fell = None
        onset = None
        for step in range(120):
            feet = local_feet(env)
            phase = "shift" if step < 50 else "lift_ramp" if step < 75 else "lower"
            if step < 50:
                amount = 0.04 * (step + 1) / 50
                feet[leg][1] += amount
                feet[support][1] -= amount
            elif step < 75:
                feet[leg][2] += 0.04
            elif step < 100:
                feet[leg][2] += 0.04 * (100 - step) / 25
            action, raw_action, targets = ik(env, feet, 0.5)
            q = np.asarray([env.data.qpos[a] for a in env.joint_qpos_adrs])
            before = {
                "step": step,
                "phase": phase,
                "com_y": float(env.data.subtree_com[env.root_body_id, 1]),
                "support_foot_y": float(env.data.xpos[env.foot_body_ids[support], 1]),
                "support_roll_rad": float(np.arctan2(env._get_projected_gravity()[1], -env._get_projected_gravity()[2])),
                "support_contacts": int(env._feet_contact()[support] > 0.5),
                "force_right_N": contact_force(env, env.foot_geom_ids[0]),
                "force_left_N": contact_force(env, env.foot_geom_ids[1]),
                "action_saturation": float(np.mean(np.abs(raw_action) > 0.95)),
                "torque_saturation": float(np.mean(np.abs(env.data.actuator_force[env.actuator_ids]) >= env.torque_limit * 0.999)),
                "foot_z_lifted": float(env.data.xpos[env.foot_body_ids[leg], 2]),
            }
            if step == 50:
                gid = env.foot_geom_ids[support]
                center = float(env.data.geom_xpos[gid, 1])
                half = float(env.model.geom_size[gid, 1])
                onset = {
                    "com_relative_to_support_center_m": before["com_y"] - before["support_foot_y"],
                    "support_edges_relative_to_support_center_m": [-half, half],
                    "hip_spacing_m": abs(float(env.data.xpos[env.foot_body_ids[0], 1] - env.data.xpos[env.foot_body_ids[1], 1])),
                    "foot_width_m": 2 * half,
                    "servo_error_angles_rad": (targets - q).tolist(),
                    "torques_Nm": np.asarray(env.data.actuator_force[env.actuator_ids]).tolist(),
                    "stance_force_N": before["force_right_N"] if support == 0 else before["force_left_N"],
                    "swing_force_N": before["force_left_N"] if support == 0 else before["force_right_N"],
                }
            obs, reward, terminated, truncated, info = env.step(action)
            after = dict(before)
            after.update({"reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)})
            rows.append({"variant": "actuator_off" if actuator_off else "current", "leg": leg, **after})
            if terminated or truncated:
                direction = "lateral" if abs(float(env._get_projected_gravity()[1])) > abs(float(env._get_projected_gravity()[0])) else "sagittal"
                fell = {"step": step, "direction": direction, "phase": phase}
                break
        env.close()
        summary.append({"variant": "actuator_off" if actuator_off else "current", "leg": leg, "fall": fell, "lift_onset": onset, "steps_survived": len([r for r in rows if r["leg"] == leg and r["variant"] == ("actuator_off" if actuator_off else "current")])})
    return rows, summary


def main():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    current_rows, current_summary = run_variant(raw, False)
    off_rows, off_summary = run_variant(raw, True)
    output = {"summary": current_summary + off_summary, "per_step": current_rows + off_rows}
    out = ROOT / "BipedRobot/paper_results/lift_diagnosis.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
