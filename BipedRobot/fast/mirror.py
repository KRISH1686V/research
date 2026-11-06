"""Left-right mirror maps for the V2 robot, for data augmentation.

Reflection across the sagittal plane. With V2's joint axes the mirrored angle of a right joint equals the left joint's angle (sign +1 for all
six joints), so mirroring swaps the legs. Vectors: velocity command (vx, vy, wz) -> (vx, -vy, -wz); gyro (wx, wy, wz) -> (-wx, wy, -wz);
gravity (gx, gy, gz) -> (gx, -gy, gz). tests/test_doe_features.py checks this against the simulator.
"""
from __future__ import annotations

import numpy as np

N = 12


def action_perm() -> np.ndarray:
    return np.r_[np.arange(N // 2, N), np.arange(0, N // 2)]


def frame_maps() -> tuple[np.ndarray, np.ndarray]:
    """One 42-number frame: joint pos, joint vel, gyro, gravity, previous action."""
    a = action_perm()
    perm = np.r_[a, 12 + a, 24, 25, 26, 27, 28, 29, 30 + a]
    sign = np.ones(42)
    sign[[24, 26, 28]] = -1.0
    return perm, sign


def obs_maps(frames: int, n_budget: int = 0, clock: bool = False, slow_yaw: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """[achieved 3, desired 3 + budget, frames x 42, gait clock sine and cosine, slow yaw rate, budget levels]. Swapping the legs is half a clock cycle, so the clock changes sign, and so does yaw."""
    goal_sign = np.array([1.0, -1.0, -1.0])
    perm, sign = [np.arange(3)], [goal_sign]
    perm.append(3 + np.arange(3)); sign.append(goal_sign)
    start = 6
    if n_budget:
        perm.append(start + np.arange(n_budget)); sign.append(np.ones(n_budget))
        start += n_budget
    fp, fs = frame_maps()
    for k in range(frames):
        perm.append(start + 42 * k + fp); sign.append(fs)
    start += 42 * frames
    if clock:
        perm.append(start + np.arange(2)); sign.append(-np.ones(2))
        start += 2
    if slow_yaw:
        perm.append(start + np.arange(1)); sign.append(-np.ones(1))
        start += 1
    if n_budget:
        perm.append(start + np.arange(n_budget)); sign.append(np.ones(n_budget))
    return np.concatenate(perm), np.concatenate(sign)


def priv_maps(extra: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Critic-only state: one clean frame plus physical parameters, which do not change under mirroring."""
    fp, fs = frame_maps()
    return np.r_[fp, 42 + np.arange(extra)], np.r_[fs, np.ones(extra)]
