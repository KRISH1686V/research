"""HERLocoEnv with realism corrections. HERLocoEnv itself is unchanged, so earlier results stay reproducible.

Each correction is off unless named in config["validity"]:

  friction: true
      HERLocoEnv randomizes only the floor's friction, but MuJoCo uses the larger of the two geoms' values at equal priority and
      the foot has 1.0, so anything below 1.0 had no effect. Raising the floor's priority makes the floor's value apply.
  push_s: 0.1
      HERLocoEnv's push lasts 5 physics substeps (25 ms), although its comment says 0.1 s. This sets the duration in seconds.
  motor_torque: true
      HERLocoEnv's "motor strength" scales the position target and leaves the torque limit alone. This scales kp and the torque limit.
  servo_curve: {stall_nm: 2.94, noload_rad_s: 4.72, cap_nm: 2.94}
      Linear torque-speed limit of a DC servo, min(cap, stall * (1 - |speed| / noload)), applied at every physics substep to torque in
      the direction of motion. cap_nm defaults to stall_nm. Not identified on the real servos.
  obs_noise: {joint_pos: 0.002, joint_vel: 0.1, gyro: 0.05, gravity: 0.02}
      Gaussian sensor noise in physical units on the observation. Assumed values, not measured on the robot.
  obs_bias: {joint_pos: 0.0175, gyro: 0.02, gravity: 0.05}
      Constant per-episode offset, uniform within +- these values: servo mounting error, gyro bias, IMU misalignment.
  history: 5
      The policy observes the last N frames, oldest first, so it can infer delay and terrain.
  encoder_rad: 0.00153398
      Observed joint angles are rounded to this size, one step of a 4096-step servo encoder.
  joint_vel_from_position: true
      Observed joint speed is the difference of the last two observed angles over one control step, as on the real robot, whose driver drops the sign of the
      servo's speed reading. Set obs_noise.joint_vel to 0 with this, since angle noise and rounding already reach the speed.
  clock_hz: 2.0
      The policy also reads the sine and cosine of a gait clock, placed after the frames. The clock is a counter, so the real robot has it too.
  slow_yaw_s: 0.5
      The policy also reads the mean of its sensed yaw rate over this time, times 5, placed last. The gyro swings with every step, so a slow turn is invisible in a few frames.
  friction_focus: {share: 0.5, range: [0.3, 0.4]}
      Training only. A share of episodes, scaled by difficulty, redraw the floor friction from this range; the others keep the full range. Only the training workers switch it on
      (use_friction_focus), so every test still draws from domain_randomization.friction_range. Its coin and redraw use their own generator, so the other randomized values don't change.

Also, outside `validity`:
  set_difficulty(d), 0 to 1: scales randomization, noise, biases, pushes and backlash from easy to the configured values. d = 1 changes nothing.
  shaping.torque_penalty_weight: subtracts ramp * weight * mean((tau / limit)^2) through r_smooth.
  shaping.phase_weight (needs clock_hz and gait_shaping): adds weight * (matches - 1) through r_gait_move, where matches counts the feet (0, 1 or 2) doing what the clock asks: right foot down
  and left foot up in the first half of each cycle, the other way round in the second. This is a hand-tuned gait reward, so it belongs only with gait shaping on.
  privileged(): clean frame plus friction, mass scale, delay and mean torque scale, for an asymmetric critic. Never given to the policy.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

from gymnasium import spaces
import mujoco
import numpy as np

from BipedRobot.envs.her_loco_env import HERLocoEnv


class ValidLocoEnv(HERLocoEnv):
    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        self._push_remaining = 0
        self._push_substeps = None
        self._torque_scale = None
        self._hist = None
        self._priv = None
        self._difficulty = 1.0
        self._easy_active = False
        super().__init__(config=config, render_mode=render_mode)
        v = dict(self.config.get("validity", {}) or {})
        self.validity = v
        if v.get("friction"):
            self.model.geom_priority[self.floor_geom_id] = 1
        if v.get("push_s"):
            self._push_substeps = int(round(float(v["push_s"]) / self.sim_dt))
        self.motor_torque = bool(v.get("motor_torque"))
        self._kp0 = self.model.actuator_gainprm[self.actuator_ids, 0].copy()
        self._kb0 = self.model.actuator_biasprm[self.actuator_ids, 1].copy()
        self._torque_scale = np.ones(self.num_joints)
        curve = v.get("servo_curve")
        self._curve_w0 = float(curve["noload_rad_s"]) if curve else None
        self._stall = float(curve["stall_nm"]) if curve else self.torque_limit
        self._cap = float(curve.get("cap_nm", curve["stall_nm"])) if curve else self.torque_limit
        self._bias_cfg = {k: float(x) for k, x in (v.get("obs_bias") or {}).items()}
        self._bias = None
        self._noise = {k: float(x) for k, x in (v.get("obs_noise") or {}).items()}
        self._noise_rng = np.random.default_rng(0)
        self.torque_penalty_weight = float(self.config.get("shaping", {}).get("torque_penalty_weight", 0.0))
        focus = v.get("friction_focus") or {}
        self.focus_share = float(focus.get("share", 0.0))
        self.focus_range = tuple(focus.get("range", (0.0, 0.0)))
        self.use_friction_focus = False
        self._focus_rng = np.random.default_rng(0)
        self.clock_hz = float(v.get("clock_hz", 0.0))
        self.slow_yaw_s = float(v.get("slow_yaw_s", 0.0))
        self._slow_yaw = deque(maxlen=max(1, int(round(self.slow_yaw_s / self.dt)))) if self.slow_yaw_s else None
        self.phase_weight = float(self.config.get("shaping", {}).get("phase_weight", 0.0))
        if self.phase_weight and not (self.clock_hz and self.gait_shaping):
            raise ValueError("shaping.phase_weight needs validity.clock_hz and shaping.gait_shaping")
        self._clock_k = 0
        self._phase = 0.0
        self.encoder_rad = float(v.get("encoder_rad", 0.0))
        self.vel_from_position = bool(v.get("joint_vel_from_position", False))
        self._last_angles = None
        self.history = int(v.get("history", 1))
        if self.history > 1:
            self._hist = deque(maxlen=self.history)
            o = self.observation_space.spaces["observation"].shape[0]
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-np.inf, np.inf, (o * self.history,), np.float32),
                "achieved_goal": self.observation_space.spaces["achieved_goal"],
                "desired_goal": self.observation_space.spaces["desired_goal"],
            })
        extra = (2 if self.clock_hz else 0) + (1 if self.slow_yaw_s else 0)
        if extra:
            o = self.observation_space.spaces["observation"].shape[0]
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(-np.inf, np.inf, (o + extra,), np.float32),
                "achieved_goal": self.observation_space.spaces["achieved_goal"],
                "desired_goal": self.observation_space.spaces["desired_goal"],
            })
        self._hard = dict(friction=tuple(self.dr_friction_range), mass=tuple(self.dr_mass_scale_range), motor=tuple(self.dr_motor_strength_range),
                          delay=tuple(self.delay_range), push=float(self.dr_push_magnitude), backlash=float(self.backlash_deadband))

    # HERLocoEnv assigns the literal 5 when a push starts, then counts down.
    @property
    def _push_steps_remaining(self) -> int:
        return self._push_remaining

    @_push_steps_remaining.setter
    def _push_steps_remaining(self, value: int) -> None:
        if self._push_substeps is not None and self._push_remaining == 0 and value == 5:
            value = self._push_substeps
        self._push_remaining = value

    def set_difficulty(self, d: float) -> None:
        self._difficulty = float(min(1.0, max(0.0, d)))

    def _apply_difficulty(self) -> None:
        """Interpolate towards the configured ranges. At d = 1 the attributes are left alone, so test conditions can set them directly."""
        d, h = self._difficulty, self._hard
        if d >= 1.0:
            if self._easy_active:
                self.dr_friction_range, self.dr_mass_scale_range = h["friction"], h["mass"]
                self.dr_motor_strength_range, self.delay_range = h["motor"], h["delay"]
                self.dr_push_magnitude, self.backlash_deadband = h["push"], h["backlash"]
                self._easy_active = False
            return
        self._easy_active = True
        lerp = lambda easy, hard: tuple(e + d * (x - e) for e, x in zip(easy, hard))
        self.dr_friction_range = lerp((h["friction"][1], h["friction"][1]), h["friction"])
        self.dr_mass_scale_range = lerp((1.0, 1.0), h["mass"])
        self.dr_motor_strength_range = lerp((1.0, 1.0), h["motor"])
        mid = 0.5 * (h["delay"][0] + h["delay"][1])
        self.delay_range = lerp((mid, mid), h["delay"])
        self.dr_push_magnitude = d * h["push"]
        self.backlash_deadband = d * h["backlash"]

    def _apply_domain_randomization(self) -> None:
        self._apply_difficulty()
        super()._apply_domain_randomization()
        if self.use_friction_focus and self.focus_share and self._focus_rng.random() < self.focus_share * self._difficulty:
            self.model.geom_friction[self.floor_geom_id, 0] = self._nominal_geom_friction[self.floor_geom_id, 0] * self._focus_rng.uniform(*self.focus_range)
        if self.motor_torque:
            scale = self._motor_strength_scale.astype(np.float64)
            self._motor_strength_scale = np.ones(self.num_joints, dtype=np.float32)
            self._torque_scale = scale
            self.model.actuator_gainprm[self.actuator_ids, 0] = self._kp0 * scale
            self.model.actuator_biasprm[self.actuator_ids, 1] = self._kb0 * scale
            limit = self.torque_limit * scale
            self.model.actuator_forcerange[self.actuator_ids, 0] = -limit
            self.model.actuator_forcerange[self.actuator_ids, 1] = limit

    def _servo_curve(self, m, d) -> None:
        qd = d.qvel[self.joint_dof_adrs]
        limit = self._cap * self._torque_scale
        headroom = np.minimum(limit, np.clip(1.0 - np.abs(qd) / self._curve_w0, 0.0, 1.0) * self._stall * self._torque_scale)
        m.actuator_forcerange[self.actuator_ids, 1] = np.where(qd > 0, headroom, limit)
        m.actuator_forcerange[self.actuator_ids, 0] = -np.where(qd < 0, headroom, limit)

    def privileged(self) -> np.ndarray:
        return self._priv

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs = super()._get_obs_dict()
        params = (float(self.model.geom_friction[self.floor_geom_id, 0]), float(self.model.body_mass[self.torso_body_id] / self._nominal_torso_mass),
                  float(self._active_delay), float(np.mean(self._torque_scale)) if self._torque_scale is not None else 1.0)
        self._priv = np.concatenate((obs["observation"], params)).astype(np.float32)
        if self._noise or self._bias is not None:
            scale = self.obs_scale
            o = obs["observation"]
            n = self.num_joints
            r = self._noise_rng
            if self._bias is not None:
                o[:n] += self._bias["joint_pos"] * scale["joint_pos"]
                o[2 * n:2 * n + 3] += self._bias["gyro"] * scale["base_ang_vel"]
                o[2 * n + 3:2 * n + 6] += self._bias["gravity"] * scale["proj_gravity"]
            d = self._difficulty
            o[:n] += d * r.normal(0.0, self._noise.get("joint_pos", 0.0), n) * scale["joint_pos"]
            o[n:2 * n] += d * r.normal(0.0, self._noise.get("joint_vel", 0.0), n) * scale["joint_vel"]
            o[2 * n:2 * n + 3] += d * r.normal(0.0, self._noise.get("gyro", 0.0), 3) * scale["base_ang_vel"]
            o[2 * n + 3:2 * n + 6] += d * r.normal(0.0, self._noise.get("gravity", 0.0), 3) * scale["proj_gravity"]
        if getattr(self, "encoder_rad", 0.0) or getattr(self, "vel_from_position", False):
            self._sensed_joints(obs["observation"])
        if getattr(self, "_slow_yaw", None) is not None:
            self._slow_yaw.append(float(obs["observation"][2 * self.num_joints + 2]) / self.obs_scale["base_ang_vel"])
        if self._hist is not None:
            if not self._hist:
                self._hist.extend([obs["observation"].copy()] * self.history)
            else:
                self._hist.append(obs["observation"].copy())
            obs["observation"] = np.concatenate(self._hist).astype(np.float32)
        if getattr(self, "clock_hz", 0.0):
            self._phase = (self._clock_k * self.dt * self.clock_hz) % 1.0
            self._clock_k += 1
            angle = 2.0 * np.pi * self._phase
            obs["observation"] = np.concatenate((obs["observation"], [np.sin(angle), np.cos(angle)])).astype(np.float32)
        if getattr(self, "_slow_yaw", None) is not None:
            obs["observation"] = np.concatenate((obs["observation"], [5.0 * np.mean(self._slow_yaw)])).astype(np.float32)
        return obs

    def _sensed_joints(self, o: np.ndarray) -> None:
        """Joint angles and speeds as the robot's sensors give them. Works on one frame in place, once per control step."""
        n, s = self.num_joints, self.obs_scale
        angles = o[:n] / s["joint_pos"]
        if self.encoder_rad:
            angles = np.round(angles / self.encoder_rad) * self.encoder_rad
            o[:n] = angles * s["joint_pos"]
        if self.vel_from_position:
            speed = np.zeros(n, dtype=np.float32) if self._last_angles is None else (angles - self._last_angles) / self.dt
            o[n:2 * n] = speed * s["joint_vel"] + (o[n:2 * n] - self._true_speed_scaled())
        self._last_angles = angles.copy()

    def _true_speed_scaled(self) -> np.ndarray:
        return np.array([self.data.qvel[a] for a in self.joint_dof_adrs], dtype=np.float32) * self.obs_scale["joint_vel"]

    def _get_info(self, *args, **kwargs):
        info = super()._get_info(*args, **kwargs)
        w = getattr(self, "torque_penalty_weight", 0.0)
        if w:
            info["r_smooth"] = float(info["r_smooth"]) - max(0.0, min(1.0, self.smoothness_ramp)) * w * float(info["torque_rms"]) ** 2
        p = getattr(self, "phase_weight", 0.0)
        if p:
            right_down, left_down = self._feet_contact() > 0.5
            first_half = self._phase < 0.5
            info["r_gait_move"] = float(info["r_gait_move"]) + p * (int(right_down == first_half) + int(left_down != first_half) - 1)
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._noise_rng = np.random.default_rng(seed)
            self._focus_rng = np.random.default_rng(seed + 1)
        if self._hist is not None:
            self._hist.clear()
        self._last_angles = None
        self._clock_k = 0
        if getattr(self, "_slow_yaw", None) is not None:
            self._slow_yaw.clear()
        if getattr(self, "_bias_cfg", None):
            r, c, n = self._noise_rng, self._bias_cfg, self.num_joints
            d = self._difficulty
            self._bias = {"joint_pos": d * r.uniform(-c.get("joint_pos", 0.0), c.get("joint_pos", 0.0) + 1e-12, n).astype(np.float32),
                          "gyro": d * r.uniform(-c.get("gyro", 0.0), c.get("gyro", 0.0) + 1e-12, 3).astype(np.float32),
                          "gravity": d * r.uniform(-c.get("gravity", 0.0), c.get("gravity", 0.0) + 1e-12, 3).astype(np.float32)}
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self._curve_w0 is None:
            return super().step(action)
        mujoco.set_mjcb_control(self._servo_curve)
        try:
            return super().step(action)
        finally:
            mujoco.set_mjcb_control(None)
