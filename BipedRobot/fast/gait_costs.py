"""Per-control-step gait costs, measured the way evaluation_suite_v2 measures them, plus an optional torque penalty.

HERLocoEnv stays unmodified. Its foot_slip uses data.cvel, the velocity at the subtree centre of mass, and its contact flag is
geometric at 50 Hz. These costs use contact-point velocity and normal force at every physics substep instead.

Each step's info gains:
  cost_flight      fraction of substeps with no loaded foot and no other floor load (0-1)
  cost_slip        mean tangential contact-point speed over loaded foot-substeps (m/s, 0 if none are loaded)
  cost_slip_n      number of loaded foot-substeps behind cost_slip, needed to average correctly over time
  cost_torque_sat  fraction of actuators at the torque limit (0-1), as torque_saturation_ratio

shaping.torque_penalty_weight (default 0) subtracts ramp * weight * mean((tau / limit)^2) through r_smooth, so stored and
relabelled rewards stay equal. It needs shaping.smooth_penalties on.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import mujoco
import numpy as np

from BipedRobot.envs import her_loco_env as _hle
from BipedRobot.envs.her_loco_env import HERLocoEnv

FORCE_THRESHOLD_N = 0.5


class _SubstepHook:
    """Stands in for her_loco_env's mujoco module during env.step so every mj_step is observed."""

    def __init__(self, callback):
        self._callback = callback
        self.inner = mujoco

    def __getattr__(self, name):
        return getattr(mujoco, name)

    def mj_step(self, m, d):
        self.inner.mj_step(m, d)
        self._callback()


class GaitCostEnv(HERLocoEnv):
    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        super().__init__(config=config, render_mode=render_mode)
        self.torque_penalty_weight = float(self.config.get("shaping", {}).get("torque_penalty_weight", 0.0))
        if self.torque_penalty_weight and not self.smooth_penalties:
            raise ValueError("torque_penalty_weight needs shaping.smooth_penalties true")
        self.force_threshold_n = FORCE_THRESHOLD_N
        self._hook = _SubstepHook(self._on_substep)
        self._f6 = np.zeros(6)
        self._v6 = np.zeros(6)
        self._n_sub = self._n_flight = self._n_loaded = 0
        self._slip_sum = 0.0
        self.substep_observer = None

    def substep_loads(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Normal force per foot, force-weighted contact-point slip per foot, and non-foot floor load."""
        m, d = self.model, self.data
        fn, weighted = np.zeros(2), np.zeros(2)
        nonfoot = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor_geom_id not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.floor_geom_id else c.geom1
            mujoco.mj_contactForce(m, d, i, self._f6)
            normal = abs(self._f6[0])
            if other in self.foot_geom_ids:
                k = self.foot_geom_ids.index(other)
                fn[k] += normal
                body = self.foot_body_ids[k]
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, self._v6, 0)
                # x and y of v + w x r, written out because np.cross took about a quarter of the env step's CPU time. Same arithmetic, so the same numbers.
                wx, wy, wz, vx, vy, _ = self._v6.tolist()
                rx, ry, rz = (c.pos - d.xpos[body]).tolist()
                weighted[k] += normal * float(np.hypot(vx + (wy * rz - wz * ry), vy + (wz * rx - wx * rz)))
            else:
                nonfoot += normal
        return fn, np.where(fn > 1e-9, weighted / np.maximum(fn, 1e-9), 0.0), nonfoot

    def _on_substep(self) -> None:
        fn, slip, nonfoot = self.substep_loads()
        loaded = fn > self.force_threshold_n
        self._n_sub += 1
        self._n_loaded += int(loaded.sum())
        self._slip_sum += float(slip[loaded].sum())
        if not loaded.any() and nonfoot <= self.force_threshold_n:
            self._n_flight += 1
        if self.substep_observer is not None:
            self.substep_observer(fn, slip, nonfoot)

    def _get_info(self, *args, **kwargs):
        info = super()._get_info(*args, **kwargs)
        if self.torque_penalty_weight:
            ramp = max(0.0, min(1.0, self.smoothness_ramp))
            info["r_torque"] = -ramp * self.torque_penalty_weight * float(info["torque_rms"]) ** 2
            info["r_smooth"] = float(info["r_smooth"]) + info["r_torque"]
        return info

    def step(self, action):
        self._n_sub = self._n_flight = self._n_loaded = 0
        self._slip_sum = 0.0
        original = _hle.mujoco
        self._hook.inner = original
        _hle.mujoco = self._hook
        try:
            obs, reward, terminated, truncated, info = super().step(action)
        finally:
            _hle.mujoco = original
        info["cost_flight"] = self._n_flight / max(1, self._n_sub)
        info["cost_slip"] = self._slip_sum / self._n_loaded if self._n_loaded else 0.0
        info["cost_slip_n"] = self._n_loaded
        info["cost_torque_sat"] = float(info["torque_saturation_ratio"])
        return obs, reward, terminated, truncated, info
