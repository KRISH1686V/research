"""Shared vectorized reward logic shared by the environment and HER buffer."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _as_batch(array: np.ndarray | Sequence[float], *, min_rows: int = 1) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < min_rows:
        raise ValueError(f"Expected at least {min_rows} columns, got {arr.shape}")
    return arr


def _normalize_info_list(info: Any, size: int) -> list[dict[str, Any]]:
    if info is None:
        return [{} for _ in range(size)]
    if isinstance(info, dict):
        return [info for _ in range(size)]
    if isinstance(info, (list, tuple, np.ndarray)):
        info_list = list(info)
        if len(info_list) == size:
            return [i if isinstance(i, dict) else {} for i in info_list]
        return [{k: v for k, v in (item.items() if isinstance(item, dict) else {})} for item in info_list[:size]] + [{} for _ in range(size - len(info_list))]
    return [{} for _ in range(size)]


def _fell_mask(info: Any, size: int) -> np.ndarray:
    info_list = _normalize_info_list(info, size)
    mask = np.zeros(size, dtype=bool)
    for idx, item in enumerate(info_list):
        if isinstance(item, dict):
            mask[idx] = bool(item.get("fell", False))
    return mask


def _info_values(info: Any, size: int, key: str) -> np.ndarray:
    info_list = _normalize_info_list(info, size)
    values = np.zeros(size, dtype=np.float32)
    for idx, item in enumerate(info_list):
        if isinstance(item, dict):
            values[idx] = float(item.get(key, 0.0))
    return values


def tracking_reward_batch(
    achieved_goal: np.ndarray | Sequence[Sequence[float]],
    desired_goal: np.ndarray | Sequence[Sequence[float]],
    *,
    sigma_v: float = 0.35,
    sigma_w: float = 0.35,
) -> np.ndarray:
    achieved = _as_batch(achieved_goal, min_rows=3)
    desired = _as_batch(desired_goal, min_rows=3)
    if achieved.shape[0] != desired.shape[0]:
        raise ValueError("achieved_goal and desired_goal must have aligned batch sizes")
    v_diff = achieved[:, :2] - desired[:, :2]
    w_diff = achieved[:, 2] - desired[:, 2]
    v_err_sq = np.sum(v_diff ** 2, axis=-1)
    w_err_sq = w_diff ** 2
    return (
        np.exp(-v_err_sq / (2.0 * sigma_v ** 2))
        * np.exp(-w_err_sq / (2.0 * sigma_w ** 2))
    ).astype(np.float32)


def compute_reward_batch(
    achieved_goal: np.ndarray | Sequence[Sequence[float]],
    desired_goal: np.ndarray | Sequence[Sequence[float]],
    info: Any = None,
    *,
    reward_mode: str = "track",
    adaptive_reward_fn: Any = None,
    budget_goal_enabled: bool = False,
    fall_penalty: float = 5.0,
    sigma_v: float = 0.35,
    sigma_w: float = 0.35,
    stand_safe_weights: dict[str, float] | None = None,
    progress_weight: float = 0.0,
    gait_shaping: bool = False,
    smooth_penalties: bool = False,
) -> np.ndarray:
    achieved = _as_batch(achieved_goal, min_rows=3)
    desired = _as_batch(desired_goal, min_rows=3)
    if achieved.shape[0] != desired.shape[0]:
        raise ValueError("achieved_goal and desired_goal must have aligned batch sizes")

    if adaptive_reward_fn is not None:
        rewards = np.empty(len(achieved), dtype=np.float32)
        info_list = _normalize_info_list(info, len(achieved))
        for i in range(len(achieved)):
            reward, _ = adaptive_reward_fn.compute_reward(achieved[i], desired[i], info_list[i])
            rewards[i] = reward
        return rewards

    base_rewards = tracking_reward_batch(achieved, desired, sigma_v=sigma_v, sigma_w=sigma_w)
    if progress_weight:
        positive_command = desired[:, 0] > 0.1
        progress = np.zeros(len(achieved), dtype=np.float32)
        progress[positive_command] = np.clip(
            achieved[positive_command, 0] / desired[positive_command, 0], 0.0, 1.0
        )
        base_rewards = base_rewards + np.float32(progress_weight) * progress
    if smooth_penalties:
        base_rewards = base_rewards + _info_values(info, len(achieved), "r_smooth")
    if budget_goal_enabled:
        if achieved.shape[1] < 7 or desired.shape[1] < 7:
            raise ValueError("budget goal reward requires 7D achieved/desired goals")
        achieved_costs = np.clip(achieved[:, 3:], 0.0, None)
        requested = np.maximum(desired[:, 3:], 1e-6)
        compliance = np.minimum(1.0, requested / np.maximum(achieved_costs, 1e-6))
        compliance_factor = np.mean(compliance, axis=1)
        res = (base_rewards * compliance_factor).astype(np.float32)
        fell_mask = _fell_mask(info, len(achieved))
        if np.any(fell_mask):
            res[fell_mask] = -np.float32(fall_penalty)
        return res

    if gait_shaping:
        moving = np.linalg.norm(desired[:, :2], axis=1) > 0.1
        gait = np.where(
            moving,
            _info_values(info, len(achieved), "r_gait_move"),
            _info_values(info, len(achieved), "r_gait_still"),
        )
        res = base_rewards + _info_values(info, len(achieved), "r_flight") + gait
    else:
        res = base_rewards

    if reward_mode == "stand_safe":
        weights = stand_safe_weights or {"pose": 1.0, "symmetry": 1.0, "torque": 1.0, "action": 1.0}
        penalties = np.zeros(len(achieved), dtype=np.float32)
        info_list = _normalize_info_list(info, len(achieved))
        for i, item in enumerate(info_list):
            if not isinstance(item, dict):
                continue
            penalties[i] = (
                weights["pose"] * float(item.get("pose_deviation", 0.0))
                + weights["symmetry"] * float(item.get("stand_symmetry", 0.0))
                + weights["torque"] * float(item.get("torque_saturation_ratio", 0.0))
                + weights["action"] * float(item.get("action_saturation_ratio", 0.0))
            )
        res = (res - penalties).astype(np.float32)
        fell_mask = _fell_mask(info, len(achieved))
        if np.any(fell_mask):
            res[fell_mask] = -np.float32(fall_penalty)
        return res

    fell_mask = _fell_mask(info, len(achieved))
    res = res.astype(np.float32)
    if np.any(fell_mask):
        res[fell_mask] = -np.float32(fall_penalty)
    return res
