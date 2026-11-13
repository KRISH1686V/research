"""DR-off action-range and loaded single-leg-lift feasibility probe."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.reference.walking_controller import ClosedLoopWalkingController


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml"


def action_limited_reach(env: HERLocoEnv) -> dict:
    # MuJoCo names hip pitch as *_hip_x; *_hip_y is the lateral axis.
    hip_indices = [7, 1]
    excursions = []
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    for index in hip_indices:
        positive = env.default_qpos[index] + env.action_range_fraction * (
            env.jnt_max[index] - env.default_qpos[index]
        )
        negative = env.default_qpos[index] - env.action_range_fraction * (
            env.default_qpos[index] - env.jnt_min[index]
        )
        excursions.append(
            {
                "joint_index": index,
                "positive_rad": float(positive - env.default_qpos[index]),
                "negative_rad": float(negative - env.default_qpos[index]),
            }
        )
    foot_ranges = []
    for leg, index in enumerate(hip_indices):
        baseline = env.data.xpos[env.foot_body_ids[leg], 0]
        values = []
        for target in (
            env.default_qpos[index]
            + env.action_range_fraction * (env.jnt_max[index] - env.default_qpos[index]),
            env.default_qpos[index]
            - env.action_range_fraction * (env.default_qpos[index] - env.jnt_min[index]),
        ):
            scratch.qpos[:] = env.data.qpos
            scratch.qpos[env.joint_qpos_adrs[index]] = target
            mujoco.mj_forward(env.model, scratch)
            values.append(float(scratch.xpos[env.foot_body_ids[leg], 0] - baseline))
        foot_ranges.append(values)
    max_step = max(abs(value) for values in foot_ranges for value in values) * 2.0
    return {
        "hip_pitch_excursions_rad": excursions,
        "foot_x_displacements_m": foot_ranges,
        "max_step_length_m": float(max_step),
        "action_range_fraction": env.action_range_fraction,
        "required_step_lengths_m": {
            str(v): {str(f): float(v / f) for f in (1.5, 2.0, 2.5)}
            for v in (0.2, 0.4, 0.6)
        },
    }


def lift_trial(env: HERLocoEnv, leg: int) -> dict:
    controller = ClosedLoopWalkingController(env, 0.0)
    support = 1 - leg
    clearances = []
    lateral_shift = []
    saturation = []
    torque_saturation = []
    upright = True
    for step in range(110):
        feet = [controller._foot_local(0), controller._foot_local(1)]
        if step < 50:
            feet[leg][1] += 0.04 * (step + 1) / 50.0
            feet[support][1] -= 0.04 * (step + 1) / 50.0
        elif step < 75:
            feet[leg][2] += 0.04
        elif step < 100:
            feet[leg][2] += 0.04 * (100 - step) / 25.0
        action = probe_ik(env, feet, controller)
        _, _, terminated, truncated, info = env.step(action)
        clearances.append(float(env.data.xpos[env.foot_body_ids[leg], 2]))
        lateral_shift.append(
            abs(
                float(
                    env.data.subtree_com[env.root_body_id, 1]
                    - env.data.xpos[env.foot_body_ids[support], 1]
                )
            )
        )
        saturation.append(float(controller.mean_action_saturation))
        torque_saturation.append(float(info.get("torque_saturation_ratio", 0.0)))
        upright &= not bool(terminated or truncated)
        if terminated or truncated:
            break
    return {
        "leg": "right" if leg == 0 else "left",
        "achieved_peak_foot_clearance_m": float(max(clearances) if clearances else 0.0),
        "upright": bool(upright),
        "action_saturation_fraction": float(controller.mean_action_saturation),
        "mean_torque_saturation_fraction": float(np.mean(torque_saturation)),
        "max_lateral_com_shift_m": float(max(lateral_shift) if lateral_shift else 0.0),
        "steps": len(clearances),
    }


def probe_ik(env: HERLocoEnv, desired_feet: list[np.ndarray], controller) -> np.ndarray:
    """Use the XML's actual chain order; this is only for the feasibility probe."""
    controller.scratch.qpos[:] = env.data.qpos
    controller.scratch.qvel[:] = env.data.qvel
    mujoco.mj_forward(env.model, controller.scratch)
    torso = env.torso_body_id
    rotation = controller.scratch.xmat[torso].reshape(3, 3)
    origin = controller.scratch.xpos[torso].copy()
    leg_indices = ((6, 7, 8, 9, 10, 11), (0, 1, 2, 3, 4, 5))
    for leg, indices in enumerate(leg_indices):
        body_id = env.foot_body_ids[leg]
        target_world = origin + rotation @ desired_feet[leg]
        for _ in range(controller.params.ik_iterations):
            mujoco.mj_forward(env.model, controller.scratch)
            error = target_world - controller.scratch.xpos[body_id]
            jacp = np.zeros((3, env.model.nv))
            jacr = np.zeros((3, env.model.nv))
            mujoco.mj_jacBody(env.model, controller.scratch, jacp, jacr, body_id)
            columns = [env.joint_dof_adrs[i] for i in indices]
            jacobian = jacp[:, columns]
            damped = jacobian @ jacobian.T + controller.params.ik_damping**2 * np.eye(3)
            delta = jacobian.T @ np.linalg.solve(damped, error)
            for increment, joint_index in zip(delta, indices):
                adr = env.joint_qpos_adrs[joint_index]
                controller.scratch.qpos[adr] = np.clip(
                    controller.scratch.qpos[adr] + increment,
                    env.jnt_min[joint_index],
                    env.jnt_max[joint_index],
                )
    targets = np.asarray(
        [controller.scratch.qpos[adr] for adr in env.joint_qpos_adrs], dtype=np.float32
    )
    raw_actions = np.zeros(env.num_joints, dtype=np.float32)
    for i, target in enumerate(targets):
        scale = (
            env.jnt_max[i] - env.default_qpos[i]
            if target >= env.default_qpos[i]
            else env.default_qpos[i] - env.jnt_min[i]
        )
        raw_actions[i] = (target - env.default_qpos[i]) / max(
            scale * env.action_range_fraction, 1e-6
        )
    controller.saturation_fraction += float(np.mean(np.abs(raw_actions) > 1.0))
    controller.action_count += 1
    return np.clip(raw_actions, -1.0, 1.0).astype(np.float32)


def main() -> None:
    with open(CONFIG, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["domain_randomization"]["enabled"] = False
    config["max_episode_steps"] = 500
    env = HERLocoEnv(config=config, render_mode=None)
    env.reset(seed=7000)
    result = {"action_limited_reach": action_limited_reach(env), "lift_trials": []}
    for leg in (0, 1):
        env.reset(seed=7000 + leg)
        result["lift_trials"].append(lift_trial(env, leg))
    result["contact_definition"] = {
        "evaluator": "HERLocoEnv._feet_contact; exact r_foot/l_foot versus floor geom contact",
        "controller": "same HERLocoEnv._feet_contact call",
        "threshold": "none; MuJoCo contact count is binary",
        "hover_2_4cm_counts_as_contact": False,
    }
    output = ROOT / "BipedRobot/paper_results/feasibility_probe.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
