"""Single deterministic evaluation path shared by training and reporting."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from BipedRobot.envs.her_loco_env import HERLocoEnv


PolicyFn = Callable[[Dict[str, np.ndarray]], np.ndarray]


def _step_gait_metrics(env: HERLocoEnv, vel_history: List[float], contact_history: List[np.ndarray], swing_heights: List[float], prev_contacts: np.ndarray) -> tuple[np.ndarray, List[float], List[float], np.ndarray]:
    """Track walking-related metrics over a single evaluation episode."""
    contacts = np.asarray(env._feet_contact(), dtype=np.float32)
    contact_history.append(contacts.copy())
    vel_history.append(float(env.data.qvel[0]))
    swing_step = []
    for foot_idx in range(len(env.foot_body_ids)):
        foot_z = float(env.data.xpos[env.foot_body_ids[foot_idx], 2])
        if contacts[foot_idx] < 0.5:
            swing_step.append(foot_z)
        else:
            swing_step.append(0.0)
    swing_heights.append(float(max(swing_step, default=0.0)))
    return contacts, vel_history, swing_heights, prev_contacts


def evaluate_policy_on_command(
    policy_fn: PolicyFn,
    env: HERLocoEnv,
    command: np.ndarray,
    seed: int,
    enable_pushes: bool = False,
    max_steps: int = 500,
) -> Dict[str, float]:
    """Evaluate one deterministic episode using the live achieved goal."""
    env.push_enabled = bool(enable_pushes and env.realism == "full")
    obs_dict, _ = env.reset(seed=seed, options={"desired_goal": command})
    errors = []
    torque_saturation = []
    vels = []
    contact_history = []
    swing_heights = []
    actions = []
    joint_velocities = []
    base_motion = []
    touchdown_sequence = []
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < max_steps:
        action = policy_fn(obs_dict)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        obs_dict, _, terminated, truncated, info = env.step(action)
        achieved = np.asarray(info.get("achieved_goal", obs_dict["achieved_goal"]))
        desired = np.asarray(obs_dict["desired_goal"])
        errors.append(float(achieved[0] - desired[0]))
        torque_saturation.append(float(info.get("torque_saturation_ratio", 0.0)))
        vels.append(float(achieved[0]))
        contacts = np.asarray(env._feet_contact(), dtype=np.float32)
        contact_history.append(contacts.copy())
        swing_heights.append(float(max([float(env.data.xpos[env.foot_body_ids[i], 2]) if contacts[i] < 0.5 else 0.0 for i in range(len(env.foot_body_ids))], default=0.0)))
        touchdown_events = np.asarray(info.get("touchdown_events", []), dtype=np.float32)
        if touchdown_events.shape == (2,):
            touchdown_sequence.extend(np.flatnonzero(touchdown_events).tolist())
        joint_velocities.append(
            np.asarray([env.data.qvel[adr] for adr in env.joint_dof_adrs], dtype=np.float32)
        )
        base_motion.append(
            (
                float(env.data.qvel[2]),
                float(np.arctan2(-env._get_projected_gravity()[0], -env._get_projected_gravity()[2])),
                float(np.arctan2(env._get_projected_gravity()[1], -env._get_projected_gravity()[2])),
                float(env.data.qpos[2]),
            )
        )
        steps += 1

    if contact_history:
        contact_arr = np.asarray(contact_history, dtype=np.float32)
        double_support_fraction = float(np.mean(np.sum(contact_arr > 0.5, axis=1) >= 2))
        touchdown_counts = []
        for foot_idx in range(contact_arr.shape[1]):
            foot_contacts = contact_arr[:, foot_idx]
            touchdown_events = np.logical_and(foot_contacts[1:] > 0.5, foot_contacts[:-1] <= 0.5)
            touchdown_counts.append(float(np.sum(touchdown_events)) / max((steps * env.dt), 1e-6))
        mean_peak_swing_foot_height = float(np.mean([max(float(x) for x in np.asarray(swing_heights)[np.asarray(contact_arr[:, i]) < 0.5]) if np.any(contact_arr[:, i] < 0.5) else 0.0 for i in range(contact_arr.shape[1])]))
    else:
        double_support_fraction = 0.0
        touchdown_counts = [0.0, 0.0]
        mean_peak_swing_foot_height = 0.0

    t_r, t_l = env._foot_contact_times[0], env._foot_contact_times[1]
    symmetry = 1.0 - abs(t_r - t_l) / max(0.1, t_r + t_l)
    steady_start = min(int(round(1.0 / env.dt)), len(actions))
    steady_actions = np.asarray(actions[steady_start:], dtype=np.float32)
    steady_joint_vel = np.asarray(joint_velocities[steady_start:], dtype=np.float32)
    steady_motion = np.asarray(base_motion[steady_start:], dtype=np.float32)
    if len(steady_actions) > 1:
        action_rate_rms = float(
            np.sqrt(np.mean(np.square(np.diff(steady_actions, axis=0))))
        )
    else:
        action_rate_rms = 0.0
    if len(steady_joint_vel) > 1:
        joint_accel_rms = float(
            np.sqrt(np.mean(np.square(np.diff(steady_joint_vel, axis=0) / env.dt)))
        )
    else:
        joint_accel_rms = 0.0
    stride_times = []
    last_touchdown = {}
    alternations = 0
    for event_index, foot_idx in enumerate(touchdown_sequence):
        if foot_idx in last_touchdown:
            stride_times.append((event_index - last_touchdown[foot_idx]) * env.dt)
        if event_index > 0 and touchdown_sequence[event_index - 1] != foot_idx:
            alternations += 1
        last_touchdown[foot_idx] = event_index
    stride_cv = (
        float(np.std(stride_times) / max(np.mean(stride_times), 1e-6))
        if stride_times
        else 0.0
    )
    lr_asymmetry = float(
        abs(t_r - t_l) / max(0.1, t_r + t_l)
    )
    vx_std_averaged = float(np.std(np.convolve(np.asarray(vels), np.ones(50) / 50, mode="valid"))) if len(vels) >= 50 else 0.0
    if len(steady_motion):
        base_pitch_roll_height_std = [
            float(np.std(steady_motion[:, 1])),
            float(np.std(steady_motion[:, 2])),
            float(np.std(steady_motion[:, 3])),
        ]
    else:
        base_pitch_roll_height_std = [0.0, 0.0, 0.0]
    moving = float(np.linalg.norm(command[:2])) > 0.1
    exploit_flags = {
        "hopping": bool(moving and double_support_fraction < 0.05 and mean_peak_swing_foot_height > 0.03),
        "foot_dragging": bool(moving and mean_peak_swing_foot_height < 0.01),
        "vibration": bool(moving and action_rate_rms > 0.5),
        "stepping_in_place_at_zero": bool(
            not moving and abs(float(np.mean(vels))) < 0.03
            and sum(touchdown_counts) > 0.5
        ),
    }
    result = {
        "survived": 1.0 if not terminated and steps >= max_steps else 0.0,
        "episode_length": float(steps),
        "mean_vx": float(np.mean(vels)) if vels else 0.0,
        "rmse_vx": float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0,
        "torque_saturation_ratio": float(np.mean(torque_saturation))
        if torque_saturation
        else 0.0,
        "gait_symmetry": float(symmetry),
        "double_support_fraction": float(double_support_fraction),
        "touchdowns_per_second_per_foot": touchdown_counts,
        "mean_peak_swing_foot_height": float(mean_peak_swing_foot_height),
        "touchdown_alternation": float(alternations / max(1, len(touchdown_sequence) - 1)),
        "rms_action_rate": action_rate_rms,
        "rms_joint_acceleration": joint_accel_rms,
        "std_1s_averaged_vx": vx_std_averaged,
        "base_pitch_std": base_pitch_roll_height_std[0],
        "base_roll_std": base_pitch_roll_height_std[1],
        "base_height_std": base_pitch_roll_height_std[2],
        "stride_time_cv": stride_cv,
        "lr_stride_asymmetry": lr_asymmetry,
        "exploit_flags": exploit_flags,
    }
    return result


def evaluate_command_suite(
    policy_fn: PolicyFn,
    env: HERLocoEnv,
    command: Iterable[float],
    seeds: Iterable[int],
    *,
    enable_pushes: bool = False,
    max_steps: int = 500,
) -> Dict[str, Any]:
    """Evaluate a command over fixed seeds and return raw and aggregate metrics."""
    command_array = np.asarray(tuple(command), dtype=np.float32)
    seed_list = [int(seed) for seed in seeds]
    episodes = [
        evaluate_policy_on_command(
            policy_fn,
            env,
            command_array,
            int(seed),
            enable_pushes=enable_pushes,
            max_steps=max_steps,
        )
        for seed in seed_list
    ]
    result = {
        "command": command_array.tolist(),
        "seeds": seed_list,
        "episodes": episodes,
        "survival": int(sum(item["survived"] for item in episodes)),
        "mean_episode_length": float(np.mean([item["episode_length"] for item in episodes])),
        "mean_vx": float(np.mean([item["mean_vx"] for item in episodes])),
        "rmse_vx": float(np.mean([item["rmse_vx"] for item in episodes])),
        "torque_saturation_ratio": float(
            np.mean([item["torque_saturation_ratio"] for item in episodes])
        ),
        "double_support_fraction": float(np.mean([item["double_support_fraction"] for item in episodes])),
        "touchdowns_per_second_per_foot": [float(np.mean([ep["touchdowns_per_second_per_foot"][i] for ep in episodes])) for i in range(2)],
        "mean_peak_swing_foot_height": float(np.mean([item["mean_peak_swing_foot_height"] for item in episodes])),
        "touchdown_alternation": float(np.mean([item["touchdown_alternation"] for item in episodes])),
        "exploit_flags": {
            key: bool(any(item["exploit_flags"][key] for item in episodes))
            for key in episodes[0]["exploit_flags"]
        },
    }
    if (
        command_array[0] > 0.1
        and result["survival"] == len(episodes)
        and abs(result["mean_vx"] - command_array[0]) <= 0.2 * command_array[0]
        and result["rmse_vx"] <= 0.08
        and result["mean_peak_swing_foot_height"] > 0.03
        and not any(result["exploit_flags"].values())
    ):
        result["smoothness_metrics"] = {
            name: float(np.mean([item[name] for item in episodes]))
            for name in (
                "rms_action_rate",
                "rms_joint_acceleration",
                "std_1s_averaged_vx",
                "base_pitch_std",
                "base_roll_std",
                "base_height_std",
                "stride_time_cv",
                "lr_stride_asymmetry",
            )
        }
    else:
        result["smoothness_metrics"] = None
    return result
