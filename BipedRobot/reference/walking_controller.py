"""Closed-loop phase/state-machine walking reference for HERLocoEnv.

This controller is an offline reference, not a deployable policy: it reads
simulator torso velocity and contact state before issuing each env action.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class WalkingParameters:
    step_frequency: float = 1.0
    duty_factor: float = 0.65
    swing_height: float = 0.04
    nominal_height: float = 0.16
    k_v: float = 0.30
    pitch_gain: float = 0.04
    roll_gain: float = 0.03
    lateral_gain: float = 0.03
    foot_width: float = 0.045
    step_length_gain: float = 0.75
    stance_back_speed: float = 0.25
    ik_damping: float = 0.03
    ik_iterations: int = 8
    contact_early_fraction: float = 0.35


class ClosedLoopWalkingController:
    """Alternating swing/stance controller using scratch-data DLS IK."""

    def __init__(self, env, command_vx: float, params: WalkingParameters | None = None):
        self.env = env
        self.command_vx = float(command_vx)
        self.params = params or WalkingParameters()
        self.scratch = mujoco.MjData(env.model)
        self.half_period = 0.5 / self.params.step_frequency
        self.swing_duration = self.half_period * (1.0 - self.params.duty_factor)
        self.elapsed = 0.0
        self.swing_leg = 0
        self.swing_start = None
        self.swing_target = None
        self.last_contacts = np.zeros(2, dtype=bool)
        self.saturation_fraction = 0.0
        self.action_count = 0
        self.touchdowns = []
        self._start_phase()

    def reset(self):
        self.elapsed = 0.0
        self.swing_leg = 0
        self.swing_start = None
        self.swing_target = None
        self.last_contacts = self.env._feet_contact().astype(bool)
        self.saturation_fraction = 0.0
        self.action_count = 0
        self.touchdowns = []
        self._start_phase()

    def _torso_frame(self) -> tuple[np.ndarray, np.ndarray]:
        torso = self.env.torso_body_id
        rotation = self.env.data.xmat[torso].reshape(3, 3)
        origin = self.env.data.xpos[torso].copy()
        return origin, rotation

    def _foot_local(self, leg: int) -> np.ndarray:
        origin, rotation = self._torso_frame()
        return rotation.T @ (self.env.data.xpos[self.env.foot_body_ids[leg]] - origin)

    def _start_phase(self):
        current = self._foot_local(self.swing_leg)
        self.swing_start = current.copy()
        sign = -1.0 if self.swing_leg == 0 else 1.0
        measured_vx = float(self.env._get_achieved_goal()[0])
        correction = self.params.k_v * (measured_vx - self.command_vx)
        step = self.params.step_length_gain * self.command_vx * self.half_period
        self.swing_target = current.copy()
        self.swing_target[0] += step + correction
        self.swing_target[1] = sign * self.params.foot_width
        # Preserve the measured touchdown height; the spline adds lift only
        # during swing. The local foot height is not equal to -pelvis height.
        self.swing_target[2] = current[2]

    def _advance_state(self, contacts: np.ndarray):
        phase_time = self.elapsed % self.half_period
        if (
            phase_time >= self.swing_duration
            or (
                phase_time >= self.swing_duration * self.params.contact_early_fraction
                and contacts[self.swing_leg]
                and not self.last_contacts[self.swing_leg]
            )
        ):
            if phase_time >= self.swing_duration:
                self.swing_leg = 1 - self.swing_leg
                self._start_phase()
        self.last_contacts = contacts.astype(bool)

    def _desired_feet(self) -> list[np.ndarray]:
        desired = [self._foot_local(0), self._foot_local(1)]
        phase_time = self.elapsed % self.half_period
        if phase_time < self.swing_duration:
            u = np.clip(phase_time / max(self.swing_duration, 1e-6), 0.0, 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            desired[self.swing_leg] = (
                (1.0 - smooth) * self.swing_start + smooth * self.swing_target
            )
            desired[self.swing_leg][2] = (
                self.swing_start[2] * (1.0 - smooth)
                + self.swing_target[2] * smooth
                + self.params.swing_height * 4.0 * u * (1.0 - u)
            )
        else:
            desired[self.swing_leg] = self.swing_target.copy()
        desired[0][1] = -self.params.foot_width
        desired[1][1] = self.params.foot_width
        return desired

    def _ik(self, desired_feet: list[np.ndarray]) -> np.ndarray:
        env = self.env
        self.scratch.qpos[:] = env.data.qpos
        self.scratch.qvel[:] = env.data.qvel
        mujoco.mj_forward(env.model, self.scratch)
        torso = env.torso_body_id
        torso_rotation = self.scratch.xmat[torso].reshape(3, 3)
        torso_origin = self.scratch.xpos[torso].copy()
        leg_indices = ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11))
        for leg, indices in enumerate(leg_indices):
            body_id = env.foot_body_ids[leg]
            target_world = torso_origin + torso_rotation @ desired_feet[leg]
            for _ in range(self.params.ik_iterations):
                mujoco.mj_forward(env.model, self.scratch)
                error = target_world - self.scratch.xpos[body_id]
                jacp = np.zeros((3, env.model.nv))
                jacr = np.zeros((3, env.model.nv))
                mujoco.mj_jacBody(env.model, self.scratch, jacp, jacr, body_id)
                columns = list(env.joint_dof_adrs[i] for i in indices)
                jacobian = jacp[:, columns]
                damped = jacobian @ jacobian.T + (self.params.ik_damping ** 2) * np.eye(3)
                delta = jacobian.T @ np.linalg.solve(damped, error)
                for col, joint_index in zip(delta, indices):
                    adr = env.joint_qpos_adrs[joint_index]
                    self.scratch.qpos[adr] = np.clip(
                        self.scratch.qpos[adr] + col,
                        env.jnt_min[joint_index],
                        env.jnt_max[joint_index],
                    )
        targets = np.asarray(
            [self.scratch.qpos[adr] for adr in env.joint_qpos_adrs], dtype=np.float32
        )
        actions = np.zeros(env.num_joints, dtype=np.float32)
        for i, target in enumerate(targets):
            positive = env.jnt_max[i] - env.default_qpos[i]
            negative = env.default_qpos[i] - env.jnt_min[i]
            scale = positive if target >= env.default_qpos[i] else negative
            actions[i] = (target - env.default_qpos[i]) / max(scale * env.action_range_fraction, 1e-6)
        clipped = np.clip(actions, -1.0, 1.0)
        self.saturation_fraction += float(np.mean(np.abs(actions) > 1.0))
        self.action_count += 1
        return clipped.astype(np.float32)

    def action(self) -> np.ndarray:
        contacts = self.env._feet_contact()
        self._advance_state(contacts)
        desired = self._desired_feet()
        # Small torso-rate feedback is expressed through symmetric foot targets.
        pitch_rate = float(self.env.data.qvel[4])
        roll_rate = float(self.env.data.qvel[3])
        desired[0][0] -= self.params.pitch_gain * pitch_rate
        desired[1][0] -= self.params.pitch_gain * pitch_rate
        desired[0][1] -= self.params.roll_gain * roll_rate
        desired[1][1] -= self.params.roll_gain * roll_rate
        action = self._ik(desired)
        self.elapsed += self.env.dt
        return action

    @property
    def mean_action_saturation(self) -> float:
        return self.saturation_fraction / max(1, self.action_count)
