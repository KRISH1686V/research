"""Left-right mirror maps for the OP3 env, same signatures and layout as fast/mirror.py (the biped's), which they reuse.

Reflection across the sagittal plane swaps the legs. Our biped's mirrored joint angles carry sign +1; in the OP3's joint conventions all 12 leg joints carry
sign -1 (a symmetric pose has opposite angles on the two legs, from the joint axes in op3.xml: hip yaw and both roll axes are the same on both legs and
change sign under reflection, the three pitch axes point opposite ways on the two legs). So the joint-angle and joint-speed entries of every frame change
sign; everything else is as for the biped. Actions are in the env's policy coordinates (right leg negated, envs/op3/op3_env.py), where mirroring is a pure
swap: action_perm is the biped's.
"""
from __future__ import annotations

import numpy as np

from BipedRobot.fast import mirror as _biped

action_perm = _biped.action_perm
FRAME = 42
JOINTS = 24  # joint angles (12) then joint speeds (12) at the start of each frame


def frame_maps() -> tuple[np.ndarray, np.ndarray]:
    perm, sign = _biped.frame_maps()
    sign = sign.copy()
    sign[:JOINTS] *= -1.0
    return perm, sign


def obs_maps(frames: int, n_budget: int = 0, clock: bool = False, slow_yaw: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """[achieved 3, desired 3 + budget, frames x 42, (clock 2), (slow yaw 1), budget levels], as fast/mirror.obs_maps."""
    if clock:
        raise ValueError("the OP3 env has no gait clock")
    perm, sign = _biped.obs_maps(frames, n_budget, clock, slow_yaw)
    sign = sign.copy()
    start = 6 + n_budget
    for k in range(frames):
        sign[start + FRAME * k:start + FRAME * k + JOINTS] *= -1.0
    return perm, sign


def priv_maps(extra: int = 4, signed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Critic-only state: one clean frame, `extra` physical parameters (unchanged), `signed` values that change sign."""
    perm, sign = _biped.priv_maps(extra, signed)
    sign = sign.copy()
    sign[:JOINTS] *= -1.0
    return perm, sign
