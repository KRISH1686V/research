"""Robotis OP3 (MuJoCo Menagerie robotis_op3) in the biped's training stack, for showing that the budget-as-goal method transfers to a second robot.

The OP3 runs through the same classes as the project's biped: HERLocoEnv (envs/her_loco_env.py, unchanged) -> ValidLocoEnv -> GaitCostEnv -> BudgetLocoEnv,
so fast/train.py, the collector, the HER replay, the token budgets and the reward relabeling run without change. Only an OP3 robot description is added:

  - joints, feet, torso: HERLocoEnv reads its joint, foot and torso names from module constants; they are swapped for the OP3's while HERLocoEnv.__init__
    runs and put back at once (op3_names), as GaitCostEnv swaps her_loco_env's mujoco module during a step.
  - 12 actuated leg joints, 6 per leg as on our biped, in our biped's order per leg (hip yaw, hip pitch, hip roll, knee, ankle pitch, ankle roll; right leg
    first). Arms and head are welded at a neutral pose in the training model (envs/op3/make_train_xml.py, which says why).
  - feet: each OP3 foot is two overlapping boxes, so contact, floor load, slip, the reset height and clearance count every sole box of a foot (Menagerie's
    shapes are kept). Flight, slip and jshake are defined as for the biped (fast/gait_costs.py, fast/budget_env.py), with one difference in slip, below.
  - slip point (env.op3.slip_reference, default "contact"): the biped's GaitCostEnv takes the foot body's velocity from mj_objectVelocity(mjOBJ_BODY),
    which MuJoCo gives at the body's centre of mass (xipos), and adds w x (contact - xpos), the lever from the body's frame origin. That is the speed of a
    point offset from the contact by the foot's centre-of-mass offset (26 mm on V2, 38 mm on the OP3), not of the contact point its docstring and the
    methods spec name. "contact" uses the lever from xipos, which is the contact point's speed (equal to mj_jac at the contact to machine precision,
    tests/test_op3_env.py); "biped_code" reproduces the biped's arithmetic exactly, for a like-for-like comparison.
  - actions in "policy coordinates": in the OP3's joint conventions a left-right symmetric pose has opposite angles on the two legs for all 12 leg joints
    (from the joint axes in op3.xml), so the right leg's actions are negated. One action value then means the same motion on either leg, and the mirror of an
    action is a pure swap of the legs, which is what the replay's mirror augmentation does (envs/op3/mirror.py). Joint target = standing pose + side * a *
    scale, symmetric about the standing pose, |a| <= 1 (the OP3 joints have no limits in the model, so the reachable range is set here).

Not ported (refused at construction): the biped's gait reference, clock and foot-mesh shaping terms, and costs that need the biped's foot mesh.
"""
from __future__ import annotations

import contextlib
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from BipedRobot.envs import her_loco_env as _hle
from BipedRobot.fast.budget_env import BudgetLocoEnv
from BipedRobot.fast.valid_env import ValidLocoEnv

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "robotis_op3"
XML = str(ASSETS / "scene_train.xml")
LEG_JOINTS = ("r_hip_yaw", "r_hip_pitch", "r_hip_roll", "r_knee", "r_ank_pitch", "r_ank_roll",
              "l_hip_yaw", "l_hip_pitch", "l_hip_roll", "l_knee", "l_ank_pitch", "l_ank_roll")
FOOT_BODIES = ("r_ank_roll_link", "l_ank_roll_link")
FOOT_GEOMS = ("r_sole_long", "l_sole_long")  # one per foot for HERLocoEnv; every collision box of a foot body counts as that foot
TORSO_BODY = "body_link"
FLOOR_GEOM = "floor"
SIDE = np.r_[-np.ones(6), np.ones(6)].astype(np.float32)  # right leg's actions are negated (policy coordinates)
# Standing pose, rad, joint coordinates: hips flexed 0.4, knees 0.8, ankles 0.4 (feet level), so the hips sit 18 mm lower than with straight legs. Mirror symmetric.
DEFAULT_QPOS = (0.0, 0.4, 0.0, -0.8, -0.4, 0.0, 0.0, -0.4, 0.0, 0.8, 0.4, 0.0)
# Reach of one action unit about the standing pose, rad, per leg: hip yaw, hip pitch, hip roll, knee, ankle pitch, ankle roll. The knee reaches straight (0)
# and 1.6 rad bent; it never bends backwards.
ACTION_SCALE = (0.3, 0.8, 0.3, 0.8, 0.6, 0.3)
TORQUE_LIMIT_NM = 5.0  # Menagerie op3.xml position actuators, forcerange -5 5
SHAPING_OK = {"gait_shaping", "action_filter", "prev_action_obs", "smooth_penalties", "reset_velocity", "action_filter_alpha", "progress_weight", "air_time_weight",
              "single_support_weight", "swing_clearance_weight", "flight_penalty_weight", "zero_touchdown_weight", "action_rate_weight", "action_second_diff_weight",
              "joint_accel_weight", "posture_weight", "target_torso_height", "base_motion_weight", "smoothness_ramp", "smoothness_ramp_fraction",
              "reset_velocity_probability", "upright", "tilt_limit", "raw_action_rate", "speed_margin", "straight", "torque_penalty_weight"}
VALIDITY_OK = {"friction", "motor_torque", "push_s", "servo_curve", "history", "obs_noise", "obs_bias", "encoder_rad", "joint_vel_from_position", "push_range_n",
               "com_offset_m", "duty_range", "obs_vibration", "friction_focus", "slow_yaw_s"}
COSTS_OK = {"flight", "slip", "torque", "shake", "jshake", "steps"}


@contextlib.contextmanager
def op3_names():
    """HERLocoEnv.__init__ looks its joints, feet and torso up by these module constants; use the OP3's while it runs."""
    names = dict(ACTUATED_JOINTS=list(LEG_JOINTS), FOOT_GEOMS=list(FOOT_GEOMS), FOOT_BODIES=list(FOOT_BODIES), TORSO_BODY=TORSO_BODY, ROOT_BODY=TORSO_BODY,
                 FLOOR_GEOM=FLOOR_GEOM)
    saved = {k: getattr(_hle, k) for k in names}
    try:
        for k, v in names.items():
            setattr(_hle, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(_hle, k, v)


def op3_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A copy of the env config with the OP3's model, standing pose and torque limit filled in where the config leaves them out, checked for biped-only options.
    HERLocoEnv fills missing sections from the biped's config/her_loco_config.yaml, so the OP3 must bring its own."""
    c = copy.deepcopy(dict(config or {}))
    c.setdefault("xml_path", XML)
    c.setdefault("robot", {}).setdefault("default_qpos", list(DEFAULT_QPOS))
    c.setdefault("actuator", {}).setdefault("torque_limit", TORQUE_LIMIT_NM)
    bad = {"shaping": set(c.get("shaping") or {}) - SHAPING_OK, "validity": set(c.get("validity") or {}) - VALIDITY_OK,
           "token_budget.costs": set((c.get("token_budget") or {}).get("costs", ())) - COSTS_OK}
    bad = {k: sorted(v) for k, v in bad.items() if v}
    if bad:
        raise ValueError(f"not ported to the OP3 (biped gait reference, clock or foot-mesh terms): {bad}")
    if (c.get("budget_goal") or {}).get("enabled"):
        raise ValueError("budget_goal is the old biped-only goal; use token_budget")
    return c


class Op3Mixin:
    """Put before a biped env class: class Op3BudgetEnv(Op3Mixin, BudgetLocoEnv)."""

    robot_type = "op3"

    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        config = op3_config(config)
        with op3_names():
            super().__init__(config=config, render_mode=render_mode)
        if self.xml_path != os.path.abspath(XML):
            raise RuntimeError(f"OP3 env loaded {self.xml_path}, expected {XML} (HERLocoEnv falls back to the biped when a path does not resolve)")
        missing = [n for n, i in zip(LEG_JOINTS, self.joint_ids) if i < 0] + [n for n, i in zip(LEG_JOINTS, self.actuator_ids) if i < 0]
        if missing or min(self.foot_body_ids + self.foot_geom_ids + [self.torso_body_id, self.floor_geom_id]) < 0 or self.model.nu != 12:
            raise RuntimeError(f"OP3 model does not match the robot description: missing {missing}, nu {self.model.nu}")
        q0 = self.default_qpos.astype(np.float64)
        if not np.allclose(q0[:6], -q0[6:]):
            raise ValueError("robot.default_qpos must be left-right symmetric (right = -left in the OP3's joint conventions) for the mirror maps to hold")
        self.side = SIDE.copy()
        self.action_scale = np.tile(np.asarray((config.get("op3") or {}).get("action_scale_rad", ACTION_SCALE), dtype=np.float32), 2)
        if self.action_scale.shape != (12,) or (self.action_scale <= 0).any():
            raise ValueError("op3.action_scale_rad needs 6 positive values per leg")
        self.jnt_min = (self.default_qpos - self.action_scale).astype(np.float32)  # the reachable targets; the model's joints are unlimited
        self.jnt_max = (self.default_qpos + self.action_scale).astype(np.float32)
        m = self.model
        self.foot_geom_sets = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])] for b in self.foot_body_ids]
        if any(not s or any(m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in s) for s in self.foot_geom_sets):
            raise RuntimeError("OP3 feet must be collision boxes")
        self._geom_foot = {g: k for k, s in enumerate(self.foot_geom_sets) for g in s}
        self._v6b = np.zeros(6)
        self.slip_reference = str((config.get("op3") or {}).get("slip_reference", "contact"))
        if self.slip_reference not in ("contact", "biped_code"):
            raise ValueError("op3.slip_reference must be contact or biped_code")

    # ---- feet -------------------------------------------------------------------------------------------------------------------------------------
    def sole_bottoms(self) -> np.ndarray:
        """Height of each foot's lowest sole corner above the floor (z = 0), [right, left], m."""
        d, m = self.data, self.model
        return np.array([min(float(d.geom_xpos[g, 2] - np.abs(d.geom_xmat[g].reshape(3, 3)[2, :]) @ m.geom_size[g]) for g in s) for s in self.foot_geom_sets])

    def foot_clearance(self) -> np.ndarray:
        return self.sole_bottoms()

    def _root_height_for_floor_contact(self) -> float:
        return float(self.data.qpos[2] - self.sole_bottoms().min())

    def _feet_contact(self) -> np.ndarray:
        """1 for a foot with any sole box touching the floor, [right, left]."""
        out = np.zeros(2, dtype=np.float32)
        f = self.floor_geom_id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if f in (c.geom1, c.geom2):
                k = self._geom_foot.get(c.geom2 if c.geom1 == f else c.geom1)
                if k is not None:
                    out[k] = 1.0
        return out

    def substep_loads(self) -> tuple:
        """GaitCostEnv.substep_loads with every sole box of a foot counted as that foot: normal force per foot, force-weighted contact-point slip per foot,
        and floor load on anything else. The biped's arithmetic, with the lever arm from the point mj_objectVelocity reports (see slip_reference)."""
        m, d = self.model, self.data
        origin = d.xipos if self.slip_reference == "contact" else d.xpos
        fn, weighted = np.zeros(2), np.zeros(2)
        nonfoot = 0.0
        vel = [None, None]
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor_geom_id not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.floor_geom_id else c.geom1
            mujoco.mj_contactForce(m, d, i, self._f6)
            normal = abs(self._f6[0])
            k = self._geom_foot.get(other)
            if k is None:
                nonfoot += normal
                continue
            fn[k] += normal
            body = self.foot_body_ids[k]
            if vel[k] is None:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, self._v6b, 0)
                vel[k] = self._v6b.tolist()
            wx, wy, wz, vx, vy, _ = vel[k]
            rx, ry, rz = (c.pos - origin[body]).tolist()
            weighted[k] += normal * float(np.hypot(vx + (wy * rz - wz * ry), vy + (wz * rx - wx * rz)))
        return fn, np.where(fn > 1e-9, weighted / np.maximum(fn, 1e-9), 0.0), nonfoot

    # ---- actions ----------------------------------------------------------------------------------------------------------------------------------
    def _action_to_target(self, action: np.ndarray) -> np.ndarray:
        """Policy coordinates to joint targets: standing pose + side * a * scale, clipped to the reachable range."""
        target = self.default_qpos + self.side * np.asarray(action, dtype=np.float32) * self.action_scale
        return np.clip(target, np.maximum(self.jnt_min, self.ctrl_min), np.minimum(self.jnt_max, self.ctrl_max)).astype(np.float32)

    def render(self):
        """RGB frame from a camera following the torso (HERLocoEnv's camera "side" is the biped's). Needs an offscreen GL context (MUJOCO_GL=egl or osmesa)."""
        if self.render_mode != "rgb_array":
            return None
        if self._viewer is None:
            self._viewer = mujoco.Renderer(self.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 1.0, 135.0, -15.0
        cam.lookat[:] = self.data.xpos[self.torso_body_id] - np.array([0.0, 0.0, 0.1])
        self._viewer.update_scene(self.data, camera=cam)
        return self._viewer.render()


class Op3Env(Op3Mixin, ValidLocoEnv):
    """OP3 with the realism corrections and no budgets."""


class Op3BudgetEnv(Op3Mixin, BudgetLocoEnv):
    """OP3 with token budgets: goal [vx, vy, wz, u_1..u_n], costs as fast/budget_env.py."""


def op3_env_class(env_config: dict):
    """The OP3 class for an env config, as fast/collector.env_class picks the biped's."""
    if env_config.get("robot_type") != "op3":
        raise ValueError(f"robot_type {env_config.get('robot_type')!r} is not op3")
    return Op3BudgetEnv if env_config.get("token_budget") else Op3Env
