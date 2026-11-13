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
  servo_planner: {accel_rad_s2: 9.0, accel_range: [7.5, 10.5], dead_s: [0.025, 0.045], speed_rad_s: 5.215}
      The real servo's own move planner, fitted on 6 Nov 2026 to the robot's step and sine tests (5 deg, hanging, speed 3400 and accel 254 as run_policy
      sends): at every physics substep the servo's setpoint moves toward the newest target at most accel_rad_s2 and speed_rad_s, braking so it would
      stop at the target, and the position actuator follows that setpoint. Each new target is a new plan, so a target that keeps moving is always braked
      for: that is what makes the real servo track 1 Hz well and 3 Hz badly. dead_s replaces the delay range (the middle of it at medium realism), and
      accel is drawn per episode from accel_range at full realism. Off at easy realism. Measured unloaded only: loaded behaviour is not determined.
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
  gait_ref: {table: config/gait_ref_v2_2hz.npz, residual: 0.25, ramp_s: 1.0} (needs clock_hz at the table's period)
      A hand-written walking pattern (fast/gait_ref.py) under the policy: joint target = pattern at the gait phase + action * residual * half the joint's
      range, clipped to the joint's range. The pattern is read at the time the delayed action was issued, so the servo delay applies to both. It ramps in
      from the standing pose over ramp_s after reset. The pattern does not depend on the command, so relabeling the command stays consistent, and its left
      leg is its right leg half a cycle later, so the mirror augmentation stays exact. A gait reference: report results with it apart from the rest.
  friction_focus: {share: 0.5, range: [0.3, 0.4]}
      Training only. A share of episodes, scaled by difficulty, redraw the floor friction from this range; the others keep the full range. Only the training workers switch it on
      (use_friction_focus), so every test still draws from domain_randomization.friction_range. Its coin and redraw use their own generator, so the other randomized values don't change.

Also, outside `validity`:
  set_difficulty(d), 0 to 1: scales randomization, noise, biases, pushes and backlash from easy to the configured values. d = 1 changes nothing.
  shaping.torque_penalty_weight: subtracts ramp * weight * mean((tau / limit)^2) through r_smooth.
  shaping.phase_weight (needs clock_hz and gait_shaping): adds weight * (matches - 1) through r_gait_move, where matches counts the feet (0, 1 or 2) doing what the clock asks: right foot down
  and left foot up in the first half of each cycle, the other way round in the second. This is a hand-tuned gait reward, so it belongs only with gait shaping on.
  shaping.clock_height: {weight: 0.3, height_m: 0.025, std_m: 0.012} (needs clock_hz and gait_shaping): adds weight * exp(-sum((clearance - target)^2) / std^2)
  through r_gait_move. Target per foot is the clock's swing: a sine bump up to height_m on the foot the clock lifts, 0 on the foot it puts down. Clearance is
  each foot's lowest mesh point above the floor. From the designer's 4 Nov 2026 feet_clock_height. A hand-tuned gait reward: report it as shaped.
  reward.track_rel: {v_rel: 0.7, v_min: 0.07, w_rel: 0.7, w_min: 0.15}: the tracking kernel's width follows the command,
  exp(-|v - v_cmd|^2 / max(v_rel |v_cmd|, v_min)^2) * exp(-(w - w_cmd)^2 / max(w_rel |w_cmd|, w_min)^2), in place of exp(-err^2 / (2 sigma^2)).
  Standing at 0.2 m/s then earns 0.13 of tracking, not 0.41. From the designer's 4 Nov 2026 track_lin_vel_xy_exp_rel; he adds the two kernels, this keeps
  the product. A tracking reward, not a gait reward. fast/replay.relabel_reward uses the same kernel.
  shaping.foot_flat: {l1: 1.0, l2: 2.0} (needs smooth_penalties): subtracts ramp * sum over feet on the floor of (l1 |sin(tilt)| + l2 sin(tilt)^2) through
  r_smooth, where tilt is the sole's angle to the floor (sole_angle: the sole's flat face from the foot mesh, 0 when it lies flat). Pitch and roll both count,
  so heel, toe and edge stances all pay. From the designer's foot_flat_l1 / foot_flat_l2, which count roll only; axes: roll does the same as his (the
  sole's sideways slope from sole_slope in place of sin(tilt)), so heel strike and toe-off are free and only edge stances pay. A hand-tuned gait reward:
  report it as shaped.
  shaping.foot_drag: {weight: 0.75, height_mm: 2} (needs gait_shaping): subtracts weight * sum over feet off the floor of (1 - clip(clearance / height, 0, 1))
  * the foot's horizontal speed, through r_gait_move: a low foot moving without carrying weight is rubbing the floor. The designer's foot_drag (-2.0).
  A hand-tuned gait reward: report it as shaped.
  privileged(): clean frame plus friction, mass scale, delay and mean torque scale, for an asymmetric critic. Never given to the policy.

Added 6 Nov 2026 for the clean-gait arms, all off unless named:
  shaping.gait_always: true (needs gait_shaping): r_gait_still takes r_gait_move's value, so the gait terms (air time, single support, feet on the
  clock, clock height, foot drag, swing contact) apply at every command. The shared reward uses r_gait_move only when |goal velocity| > 0.1 m/s and
  otherwise r_gait_still (-0.3 per touchdown), which would switch the gait terms off, and punish stepping, at the slow commands these arms train on.
  shaping.swing_contact: {weight: 1.0, load_frac: 0.05, margin: 0.05} (needs clock_hz and gait_shaping): subtracts weight * the sum over feet of the
  share of the step's physics substeps in which a foot carries more than load_frac of the robot's weight while the clock has it in swing, through
  r_gait_move. The right foot swings in the second half of each cycle, the left in the first; margin (a share of the cycle) leaves out that much at
  each end of the swing half, for the change of feet. A hand-tuned gait reward: report it as shaped.
  shaping.straight: {weight: 1.0, yaw_std: 0.10, y_std: 0.05} (needs smooth_penalties): adds weight * (0.5 exp(-(yaw - yaw0)^2 / yaw_std^2)
  + 0.5 exp(-y^2 / y_std^2)) through r_smooth, where yaw0 and the start position are taken at reset and y is the sideways offset from the start, along
  the start heading. Simulator truth, reward only. r_smooth is not relabelled. The two errors are added to privileged() (they change sign under the
  mirror; priv_signed says how many). A hand-tuned reward: report it as shaped.
  shaping.foot_flat.free_ends: 0.12 (with axes both and clock_hz): within this share of the cycle on either side of each change of feet, a loaded foot
  pays only for rolling sideways (as axes roll), so it may land heel first and push off from the toe; in the rest of the stance the whole angle pays,
  so standing on the toe or heel does. (7 Nov 2026)
  shaping.upright: {weight: 0.5, pitch_deg: 4, roll_deg: 8} (needs smooth_penalties): adds weight * exp(-(pitch / pitch_deg)^2 - (roll / roll_deg)^2)
  through r_smooth, from the torso's orientation (pitch > 0 nose down). HERLocoEnv's posture_weight term, (1 - cos tilt)^2, is about 2e-4 at 10 deg, so
  it does not hold the torso up. The robot senses the same tilt through its gravity reading. (7 Nov 2026)
  shaping.stance_width.band_m: [0.085, 0.115], with penalty: 1.0, penalty_span_m: 0.02 (8 Nov 2026, the owner: feet never under 80 or over 120 mm apart): stance_width_match is 1 inside the band and exp(-(distance outside / std_m)^2) beyond it; width_m is unused.
  penalty subtracts penalty * min(1, distance outside / span) through r_smooth at every step, outside comfort_gate. Without band_m, stance_width is as before.
  shaping.tilt_limit: {deg: 8, span_deg: 8, penalty: 1.0} (needs smooth_penalties; 8 Nov 2026, the owner: the torso must not tilt much, it risks a fall on the real robot): subtracts penalty * clip((tilt - deg) / span_deg, 0, 1) through r_smooth, tilt the torso's whole angle from vertical; outside comfort_gate.
  shaping.comfort_gate: {height_m: 0.02, frac: 0.5, min_bump_m: 0.003} (needs clock_hz, smooth_penalties; 8 Nov 2026): upright, stance_width, straight and hip_yaw_hold are paid in full only while the clock's swing foot is raised along the torso's
  axis (foot_bend) to at least frac of the clock's bump (height_m * sin(pi * swing share)), less in proportion below that, nothing at zero; where the bump is under min_bump_m they are paid in full. Measured 8 Nov: with upright 1.0 and stance_width 0.5 a robot standing still
  earned 3.17 a step against 3.70 for a real walk under its own reward. With the gate, standing and rigid tipping earn almost none of these. Goal-free, so relabeling stays exact. A hand-tuned gait reward: report it as shaped.
  shaping.bend_ramp: {weight: 1.2, height_m: 0.02}, shaping.knee_lift: {weight: 0.5, rad: 0.26}, shaping.min_clearance: {penalty: 0.5, height_m: 0.008} (need clock_hz, gait_shaping and clock_stance 0.5; 8 Nov 2026): through r_gait_move, at mid-swing
  only (40-60% of the clock's swing): weight * min(d_rel, height) / height, d_rel the swing foot's height above the other along the torso's axis (foot_bend, so a rigid tilt earns nothing); weight * min(swing knee's flexion - the other's, rad) / rad;
  - penalty if the swing foot is under height above the floor. A hand-tuned gait reward: report it as shaped.
  shaping.ref_imitate: {table: config/x.npz, weight: 1.5, std_rad: 0.2, rsi_prob: 0.3} (needs clock_hz at the table's period and gait_shaping; 8 Nov 2026): adds weight * exp(-mean over hip pitch, knee and ankle pitch of (q - table(phase))^2 / std^2) through r_gait_move;
  hip roll, ankle roll and hip yaw stay free. The action is not tied to the table (not validity.gait_ref). In training, rsi_prob of resets start at a random clock phase with the joints on the table and the feet on the floor; a reset naming its desired_goal (every evaluation)
  always starts standing at phase 0, as the robot does. Depends on the clock and the joints only, so relabeling vx stays exact. Report as gait-reference, apart from shaped.
  shaping.clock_bend: {weight: 1.2, height_m: 0.022, std_m: 0.02} (needs clock_hz and gait_shaping; 8 Nov 2026): adds weight * exp(-((d - target) / std)^2) through r_gait_move,
  d the right foot's height above the left along the torso's own vertical axis, less its value in the standing pose; target the clock's swing bump, + for the right foot's swing, - for the
  left's. Tipping the whole body with stiff legs leaves d at zero, unlike clock_height, which measures from the floor: lift_split.py found the leg shortening only 0-3 mm per step, with 55-107% of the lift
  from the hip rising. A hand-tuned gait reward: report it as shaped.
  shaping.stance_width: {weight: 0.5, width_m: 0.08, std_m: 0.04} (needs smooth_penalties; 8 Nov 2026): adds weight * exp(-((w - width_m) / std_m)^2) through
  r_smooth, w the sideways distance between the two foot centres across the heading. Nothing else holds it: trained policies landed anywhere from 76 to 147 mm
  (standing 104 mm; the designer's styles 60-90 mm). A hand-tuned reward: report it as shaped.
  shaping.hip_yaw_hold: {weight: 0.3, std_deg: 4} (needs smooth_penalties; 7 Nov 2026): adds weight * exp(-mean over the two hip yaw joints of
  ((q - standing) / std)^2) through r_smooth, so the hips stop twisting each step and the feet land pointing ahead.
  shaping.speed_margin: {weight: 0.5, frac: 0.8, noload_rad_s: 5.3} (needs smooth_penalties; 7 Nov 2026): subtracts weight * sum over joints of
  max(0, |speed| / (frac * noload) - 1)^2 through r_smooth: nothing below 80% of the servo's no-load speed, then a growing cost, so the gait keeps a
  margin for the slower loaded servos on the robot.
  validity.clock_stance: 0.58 (7 Nov 2026, needs clock_hz): each foot's share of the cycle in stance (default 0.5). Above 0.5 the clock has a double-
  support window of (stance - 0.5) around each change of feet, centred on it. phase_weight, clock_height's swing bump (now spread over the shorter
  swing) and swing_contact's swing windows all follow it. At 0.5 everything is exactly as before.
  validity.push_range_n: [2.0, 9.0] (7 Nov 2026): each random push gets its own force, uniform in this range and scaled by difficulty, in place of the
  fixed domain_randomization.push_magnitude. Direction and timing stay HERLocoEnv's (random direction, every push_interval steps); the 7 Nov push test
  showed 3 N survived, 6 N half, 9 N almost never.
  validity.com_offset_m: [0.012, 0.006] (7 Nov 2026): shifts the torso's centre of mass by up to this much forward/back and sideways, drawn per episode at
  reset and scaled by difficulty, for where the real robot's mass sits (wiring, battery, the Pi). [0, 0] or absent leaves the model as built.
  shaping.com_support: {weight: 0.3, std_m: 0.015} (needs smooth_penalties): adds weight * exp(-(d / std)^2) through r_smooth, where d is the whole
  robot's centre of mass minus the midpoint of the loaded feet, along the heading; nothing when no foot is loaded. (7 Nov 2026)
  shaping.phase_margin: 0.05 (with phase_weight): within this share of the cycle on either side of each change of feet, both feet down counts as
  both feet doing what the clock asks, so the change of feet may have a double-support phase (2 x margin of each half). (7 Nov 2026)
  shaping.raw_action_rate: 0.05 (needs smooth_penalties): subtracts ramp * weight * sum over joints of (a_t - a_{t-1})^2 through r_smooth, on the policy's
  own output before the action filter (shaping.action_rate_weight sees the filtered action). Zero on the first step after reset. (7 Nov 2026)
  validity.duty_range: [0.5, 1.0] (needs servo_curve): the servos' PWM duty limit (the torque-limit register; hardware.yaml's 500 of 1000 is 0.5),
  drawn per episode at reset, scaled by difficulty (1 - d (1 - u)). It scales stall torque, the torque cap and no-load speed together, as fitted on the
  6 Nov servo tests. Applied whether or not domain randomisation is on; [x, x] fixes it. The critic's torque-scale input is multiplied by it.

Added 13 Nov 2026 from the servo fit to the robot's own logs (paper/v3_work/servo_fit/servo_fit.md, Section 6), both off unless named:
  validity.servo_model: {kp_scale: 1.8, kv: 1.23, armature: 0.010, frictionloss: 0.12, armature_range: [0.007, 0.014]}
      The 12 joints' position gain (times kp_scale), damping kv (N m s/rad), joint armature (kg m^2) and friction loss (N m), set on the model at
      construction, at every realism, before anything else reads them, so motor-strength randomisation and the reset of servo_play keep the fitted
      values. Keys left out keep the XML value. armature_range: at full realism with domain randomisation on, each joint's armature is drawn per
      episode from this range at reset (scaled by difficulty around armature); otherwise it is armature. servo_play replaces the joint armature and
      kv while it is on.
  validity.delay_mode: zoh (default interp)
      The servo delay as a pure time shift of the held target, at physics-substep resolution: the target HERLocoEnv's chain writes in a control step
      reaches the actuators ceil(delay / substep) substeps later and is held until the next one does. HERLocoEnv interpolates the action buffer
      instead, which for delays between whole control steps averages two targets and low-passes the command (15-45% above 9 Hz at 10, 30, 50 ms);
      the robot's servo has no such filter. Works at every realism (medium: 10 ms, so 2 substeps).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from gymnasium import spaces
import mujoco
import numpy as np

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.envs.rewards.shared_reward import _fell_mask, tracking_reward_batch


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
        sm = dict(v.get("servo_model") or {})  # validity.servo_model (13 Nov 2026): set here, before _kp0, _kb0, _arm0 and _kv0 are read
        self._servo_model = sm or None
        if sm:
            unknown = set(sm) - {"kp_scale", "kv", "armature", "frictionloss", "armature_range"}
            if unknown:
                raise ValueError(f"validity.servo_model: unknown keys {sorted(unknown)}")
            ids, dofs = self.actuator_ids, self.joint_dof_adrs
            k = float(sm.get("kp_scale", 1.0))
            self.model.actuator_gainprm[ids, 0] = self.model.actuator_gainprm[ids, 0] * k
            self.model.actuator_biasprm[ids, 1] = self.model.actuator_biasprm[ids, 1] * k
            if "kv" in sm:
                self.model.actuator_biasprm[ids, 2] = -float(sm["kv"])
            if "armature" in sm:
                self.model.dof_armature[dofs] = float(sm["armature"])
            if "frictionloss" in sm:
                self.model.dof_frictionloss[dofs] = float(sm["frictionloss"])
        ar = sm.get("armature_range")
        self._arm_range = (float(ar[0]), float(ar[1])) if ar else None
        dm = str(v.get("delay_mode", "interp"))
        if dm not in ("interp", "zoh"):
            raise ValueError("validity.delay_mode must be interp or zoh")
        self.delay_zoh = dm == "zoh"
        self._zq, self._zoh_prev, self._zoh_new, self._zoh_s = [], None, False, 0
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
        # validity.obs_vibration: {gyro: 1.2, joint_vel: 0.8, hz: [3, 20], amp: [0.3, 1.0]}, rad/s. Off unless set; training (full realism) only. 9 Nov 2026. The designer's measured gear and
        # structure shake on his V2 walking (his observation_vibration_*): band-limited noise on the sensed gyro and joint speed, amplitude drawn per episode. Measured on his robot, not on ours.
        self._vib_cfg = dict(v.get("obs_vibration") or {})
        self._vib, self._vib_k = None, 0
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
        ch = self.config.get("shaping", {}).get("clock_height") or {}
        self.clock_height_weight = float(ch.get("weight", 0.0))
        self.clock_height_m = float(ch.get("height_m", 0.025))
        self.clock_height_std = float(ch.get("std_m", 0.012))
        if self.clock_height_weight and not (self.clock_hz and self.gait_shaping):
            raise ValueError("shaping.clock_height needs validity.clock_hz and shaping.gait_shaping")
        cb = self.config.get("shaping", {}).get("clock_bend") or {}
        self.clock_bend_weight = float(cb.get("weight", 0.0))
        self.clock_bend_m, self.clock_bend_std = float(cb.get("height_m", 0.022)), float(cb.get("std_m", 0.02))
        self._bend0 = None
        if self.clock_bend_weight and not (self.clock_hz and self.gait_shaping):
            raise ValueError("shaping.clock_bend needs validity.clock_hz and shaping.gait_shaping")
        ff = self.config.get("shaping", {}).get("foot_flat") or {}
        self.foot_flat_l1, self.foot_flat_l2 = float(ff.get("l1", 0.0)), float(ff.get("l2", 0.0))
        if (self.foot_flat_l1 or self.foot_flat_l2) and not self.smooth_penalties:
            raise ValueError("shaping.foot_flat needs shaping.smooth_penalties")
        self.foot_flat_axes = str(ff.get("axes", "both"))
        self.foot_flat_free_ends = float(ff.get("free_ends", 0.0))
        if self.foot_flat_free_ends and not (self.clock_hz and self.foot_flat_axes == "both"):
            raise ValueError("shaping.foot_flat.free_ends needs validity.clock_hz and axes both")
        if self.foot_flat_axes not in ("both", "roll"):
            raise ValueError("shaping.foot_flat.axes must be both or roll")
        fd = self.config.get("shaping", {}).get("foot_drag") or {}
        self.foot_drag_weight, self.foot_drag_h = float(fd.get("weight", 0.0)), float(fd.get("height_mm", 2.0)) / 1000.0
        if self.foot_drag_weight and not self.gait_shaping:
            raise ValueError("shaping.foot_drag needs shaping.gait_shaping")
        self._sole_normal = None
        sh = self.config.get("shaping", {}) or {}
        self.gait_always = bool(sh.get("gait_always", False))
        if self.gait_always and not self.gait_shaping:
            raise ValueError("shaping.gait_always needs shaping.gait_shaping")
        sc = sh.get("swing_contact") or {}
        self.swing_contact_weight = float(sc.get("weight", 0.0))
        self.swing_contact_load = float(sc.get("load_frac", 0.05))
        self.swing_contact_margin = float(sc.get("margin", 0.0))
        if self.swing_contact_weight and not (self.clock_hz and self.gait_shaping):
            raise ValueError("shaping.swing_contact needs validity.clock_hz and shaping.gait_shaping")
        self._swing_acc = np.zeros(2)
        self._swing_n = 0
        self._weight_n = None
        st = sh.get("straight") or {}
        self.straight_weight = float(st.get("weight", 0.0))
        self.straight_yaw_std, self.straight_y_std = float(st.get("yaw_std", 0.10)), float(st.get("y_std", 0.05))
        if self.straight_weight and not self.smooth_penalties:
            raise ValueError("shaping.straight needs shaping.smooth_penalties")
        self.priv_signed = 2 if self.straight_weight else 0
        self.phase_margin = float(sh.get("phase_margin", 0.0))
        hy = sh.get("hip_yaw_hold") or {}
        self.hip_yaw_weight, self.hip_yaw_std = float(hy.get("weight", 0.0)), np.radians(float(hy.get("std_deg", 4.0)))
        self._hip_z_idx = [k for k, j in enumerate(self.joint_ids) if self.model.joint(j).name.endswith("hip_z")]
        sm = sh.get("speed_margin") or {}
        self.speed_margin_weight = float(sm.get("weight", 0.0))
        self.speed_margin_limit = float(sm.get("frac", 0.8)) * float(sm.get("noload_rad_s", 5.3))
        if (self.hip_yaw_weight or self.speed_margin_weight) and not self.smooth_penalties:
            raise ValueError("shaping.hip_yaw_hold and shaping.speed_margin need shaping.smooth_penalties")
        cs = sh.get("com_support") or {}
        self.com_support_weight, self.com_support_std = float(cs.get("weight", 0.0)), float(cs.get("std_m", 0.015))
        if self.com_support_weight and not self.smooth_penalties:
            raise ValueError("shaping.com_support needs shaping.smooth_penalties")
        self.clock_stance = float(v.get("clock_stance", 0.5))
        if self.clock_stance != 0.5 and not (self.clock_hz and 0.5 < self.clock_stance < 0.75):
            raise ValueError("validity.clock_stance needs clock_hz and a value from 0.5 to 0.75")
        pr = v.get("push_range_n")
        self.push_range = tuple(float(x) for x in pr) if pr else None
        co = v.get("com_offset_m")
        self.com_offset = tuple(float(x) for x in co) if co else None
        self._torso_ipos0 = self.model.body_ipos[self.torso_body_id].copy()
        self._com_rng = np.random.default_rng(0)
        up = sh.get("upright") or {}
        self.upright_weight = float(up.get("weight", 0.0))
        self.upright_pitch, self.upright_roll = np.radians(float(up.get("pitch_deg", 4.0))), np.radians(float(up.get("roll_deg", 8.0)))
        if self.upright_weight and not self.smooth_penalties:
            raise ValueError("shaping.upright needs shaping.smooth_penalties")
        sw = sh.get("stance_width") or {}
        self.stance_width_weight = float(sw.get("weight", 0.0))
        self.stance_width_m, self.stance_width_std = float(sw.get("width_m", 0.08)), float(sw.get("std_m", 0.04))
        band = sw.get("band_m")
        self.stance_band = tuple(float(x) for x in band) if band else None
        self.stance_band_penalty, self.stance_band_span = float(sw.get("penalty", 0.0)), float(sw.get("penalty_span_m", 0.02))
        tl = sh.get("tilt_limit") or {}
        self.tilt_limit_penalty, self.tilt_limit_rad, self.tilt_limit_span = float(tl.get("penalty", 0.0)), np.radians(float(tl.get("deg", 8.0))), np.radians(float(tl.get("span_deg", 8.0)))
        if self.tilt_limit_penalty and not self.smooth_penalties:
            raise ValueError("shaping.tilt_limit needs shaping.smooth_penalties")
        if self.stance_width_weight and not self.smooth_penalties:
            raise ValueError("shaping.stance_width needs shaping.smooth_penalties")
        cg = sh.get("comfort_gate") or {}
        self.comfort_gate_m, self.comfort_gate_frac, self.comfort_gate_min = float(cg.get("height_m", 0.0)), float(cg.get("frac", 0.5)), float(cg.get("min_bump_m", 0.003))
        if self.comfort_gate_m and not (self.clock_hz and self.smooth_penalties and float(v.get("clock_stance", 0.5)) == 0.5):
            raise ValueError("shaping.comfort_gate needs validity.clock_hz, shaping.smooth_penalties and clock_stance 0.5")
        br, kl, mc, ri = (sh.get(k) or {} for k in ("bend_ramp", "knee_lift", "min_clearance", "ref_imitate"))
        self.bend_ramp_weight, self.bend_ramp_m = float(br.get("weight", 0.0)), float(br.get("height_m", 0.02))
        self.knee_lift_weight, self.knee_lift_rad = float(kl.get("weight", 0.0)), float(kl.get("rad", 0.26))
        self.min_clear_penalty, self.min_clear_m = float(mc.get("penalty", 0.0)), float(mc.get("height_m", 0.008))
        self.ref_imitate_weight, self.ref_imitate_std = float(ri.get("weight", 0.0)), float(ri.get("std_rad", 0.2))
        self.rsi_prob = float(ri.get("rsi_prob", 0.0))
        self._imit_table = None
        if self.bend_ramp_weight or self.knee_lift_weight or self.min_clear_penalty or self.ref_imitate_weight:
            if not (self.clock_hz and self.gait_shaping and float(v.get("clock_stance", 0.5)) == 0.5):
                raise ValueError("shaping.bend_ramp, knee_lift, min_clearance and ref_imitate need validity.clock_hz, shaping.gait_shaping and clock_stance 0.5")
            names = [self.model.joint(j).name for j in self.joint_ids]
            self._knee_idx = [names.index("r_knee"), names.index("l_knee")]
            self._imit_idx = [k for k, n in enumerate(names) if n.endswith(("hip_x", "knee", "ankle_y"))]
        if self.ref_imitate_weight:
            path = Path(ri["table"])
            path = path if path.is_absolute() else Path(__file__).resolve().parents[1] / path
            z = np.load(path)
            if abs(float(z["period_s"]) * self.clock_hz - 1.0) > 1e-6:
                raise ValueError(f"shaping.ref_imitate needs validity.clock_hz = 1 / {float(z['period_s'])} (the table's period)")
            self._imit_table = z["table"].astype(np.float64)
            self._imit_speed = float(z["speed_m_s"])
        # shaping.ref_full: the designer's imitation terms and kernel widths (his config_gait_reference.yaml and rewards.py), off unless set. 9 Nov 2026. See ref_full_terms.
        rf = sh.get("ref_full") or {}
        self.ref_full_w = {k: float(x) for k, x in (rf.get("weights") or {}).items()}
        self.ref_full_std = {k: float(x) for k, x in (rf.get("std") or {}).items()}
        self.ref_full_contact_m = float(rf.get("contact_height_m", 0.006))
        self.rsi_joint_noise = float(rf.get("rsi_joint_noise", 0.0))
        self._rf = None
        if self.ref_full_w:
            if self.ref_imitate_weight:
                raise ValueError("shaping.ref_full replaces shaping.ref_imitate: set one")
            if not (self.clock_hz and self.gait_shaping and float(v.get("clock_stance", 0.5)) == 0.5):
                raise ValueError("shaping.ref_full needs validity.clock_hz, shaping.gait_shaping and clock_stance 0.5")
            path = Path(rf["table"])
            path = path if path.is_absolute() else Path(__file__).resolve().parents[1] / path
            z = np.load(path)
            if abs(float(z["period_s"]) * self.clock_hz - 1.0) > 1e-6:
                raise ValueError(f"shaping.ref_full needs validity.clock_hz = 1 / {float(z['period_s'])} (the table's period)")
            self._rf = {k: z[k].astype(np.float64) for k in ("joint_pos", "joint_vel", "grav", "lin_vel", "ang_vel", "foot_pos", "foot_height", "contact")}
            self._imit_table, self._imit_speed = self._rf["joint_pos"], float(z["speed_m_s"])
            self.rsi_prob = float(rf.get("rsi_prob", 0.0))
            self._rf_feet = [self.model.body("r_foot").id, self.model.body("l_foot").id]
        self.raw_action_rate = float(sh.get("raw_action_rate", 0.0))
        if self.raw_action_rate and not self.smooth_penalties:
            raise ValueError("shaping.raw_action_rate needs shaping.smooth_penalties")
        self._raw_prev = self._raw_now = None
        self._start_pose = None
        dr = v.get("duty_range")
        self.duty_range = tuple(float(x) for x in dr) if dr else None
        if self.duty_range and self._curve_w0 is None:
            raise ValueError("validity.duty_range needs validity.servo_curve")
        self._duty = 1.0
        self._stall0, self._cap0, self._curve_w00 = self._stall, self._cap, self._curve_w0
        rel = (self.config.get("reward", {}) or {}).get("track_rel")
        self.track_rel = (float(rel.get("v_rel", 0.7)), float(rel.get("v_min", 0.07)), float(rel.get("w_rel", 0.7)), float(rel.get("w_min", 0.15))) if rel else None
        if self.track_rel and self.budget_goal_enabled:
            raise ValueError("reward.track_rel does not support budget_goal")
        self._clock_k = 0
        self._phase = 0.0
        ref = v.get("gait_ref")
        self._ref_table = None
        if ref:
            path = Path(ref["table"])
            path = path if path.is_absolute() else Path(__file__).resolve().parents[1] / path
            z = np.load(path)
            if not self.clock_hz or abs(float(z["period_s"]) * self.clock_hz - 1.0) > 1e-6:
                raise ValueError(f"validity.gait_ref needs validity.clock_hz = 1 / {float(z['period_s'])} (the table's period)")
            self._ref_table = z["table"].astype(np.float64)
            if self._ref_table.shape != (len(self._ref_table), self.num_joints):
                raise ValueError("gait_ref table must have one column per joint")
            self.ref_residual = float(ref.get("residual", 0.25))
            self.ref_ramp_s = float(ref.get("ramp_s", 1.0))
            self._ref_half = (self.jnt_max - self.jnt_min).astype(np.float64) / 2.0
        self._ref_t = 0.0
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
        sp = v.get("servo_planner")
        self._planner_on = bool(sp) and self.realism != "easy"
        self._plan_p = self._plan_v = self._plan_target = None
        self._plan_new = False
        # validity.servo_play: {deg: [1.0, 2.5], stiffness: 80.0, damping: 1.0, motor_inertia: 0.0234, link_armature: 0.005}, off unless set; training (full realism)
        # only, or when a test sets _play_on. 10 Nov 2026. Gear play INSIDE the servo's position loop, as on the STS3215 (magnetic encoder on the output shaft):
        # the motor side is its own state (inertia, back-EMF damping, torque and speed limits), driven by kp x (target - output angle); it reaches the
        # output only through the gear mesh, which is slack within +-play/2 and stiff beyond. The robot shook 2-5x more than the model on 10 Nov, with a
        # 12 Hz peak at full motion; HERLocoEnv's backlash_deadband (a deadband on the command) cannot produce that, so set enable_backlash false with this.
        pl = v.get("servo_play")
        self._play_cfg = dict(pl) if pl else None
        self._play_on = bool(pl) and self.realism == "full"
        self._play_half = self._play_m = self._play_w = self._play_target = None
        self._play_new = False
        self._arm0 = self.model.dof_armature[self.joint_dof_adrs].copy()
        self._kv0 = -self.model.actuator_biasprm[self.actuator_ids, 2].copy()
        if sp:
            self.planner_accel = float(sp["accel_rad_s2"])
            self.planner_accel_range = tuple(float(x) for x in sp.get("accel_range", (self.planner_accel, self.planner_accel)))
            self.planner_speed = float(sp.get("speed_rad_s", 5.215))
            self._plan_a = self.planner_accel
            if self._planner_on:
                lo, hi = (float(x) for x in sp["dead_s"])
                self.enable_delay = True
                self.nominal_delay = 0.5 * (lo + hi)
                self.delay_range = (lo, hi) if self.realism == "full" else (self.nominal_delay, self.nominal_delay)
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
        if self._planner_on:
            lo, hi = self.planner_accel_range
            a = self.planner_accel
            self._plan_a = a + self._difficulty * (self.np_random.uniform(lo, hi) - a) if self.realism == "full" and self.dr_enabled else a
        if self.motor_torque:
            scale = self._motor_strength_scale.astype(np.float64)
            self._motor_strength_scale = np.ones(self.num_joints, dtype=np.float32)
            self._torque_scale = scale
            self.model.actuator_gainprm[self.actuator_ids, 0] = self._kp0 * scale
            self.model.actuator_biasprm[self.actuator_ids, 1] = self._kb0 * scale
            limit = self.torque_limit * scale
            self.model.actuator_forcerange[self.actuator_ids, 0] = -limit
            self.model.actuator_forcerange[self.actuator_ids, 1] = limit

    def _servo_planner(self, m, d) -> None:
        """One physics substep of the servo's move planner. The first substep of a control step takes the target HERLocoEnv.step wrote to ctrl."""
        ids = self.actuator_ids
        if self._plan_new:
            self._plan_target = d.ctrl[ids].astype(np.float64)
            self._plan_new = False
        h, a = m.opt.timestep, self._plan_a
        e = self._plan_target - self._plan_p
        # fastest speed that can still brake to a stop on the target in whole substeps: v^2 / 2a + v h / 2 = |e|
        want = np.sign(e) * np.minimum(self.planner_speed, a * h * (np.sqrt(0.25 + 2.0 * np.abs(e) / (a * h * h)) - 0.5))
        v = self._plan_v + np.clip(want - self._plan_v, -a * h, a * h)
        p = self._plan_p + v * h
        arrived = (np.sign(self._plan_target - p) != np.sign(e)) & (np.abs(v) <= 2.0 * a * h)
        self._plan_p = np.where(arrived, self._plan_target, p)
        self._plan_v = np.where(arrived, 0.0, v)
        d.ctrl[ids] = self._plan_p

    def _servo_both(self, m, d) -> None:
        self._servo_planner(m, d)
        if self._curve_w0 is not None:
            self._servo_curve(m, d)

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
                  float(self._active_delay), (float(np.mean(self._torque_scale)) if self._torque_scale is not None else 1.0) * getattr(self, "_duty", 1.0))
        if getattr(self, "straight_weight", 0.0):
            params = params + self.straight_error()
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
        if getattr(self, "_vib", None) is not None:
            o, n, k = obs["observation"], self.num_joints, min(self._vib_k, len(self._vib[0]) - 1)
            o[2 * n:2 * n + 3] += self._vib[0][k] * self.obs_scale["base_ang_vel"]
            o[n:2 * n] += self._vib[1][k] * self.obs_scale["joint_vel"]
            self._vib_k += 1
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
            stance_r, stance_l = self.clock_stance_feet(self._phase)
            matches = int(right_down == stance_r) + int(left_down == stance_l)
            m = getattr(self, "phase_margin", 0.0)
            if m and right_down and left_down and min(self._phase % 0.5, 0.5 - self._phase % 0.5) < m:
                matches = 2  # double support around a change of feet
            info["r_gait_move"] = float(info["r_gait_move"]) + p * (matches - 1)
        if getattr(self, "foot_flat_l1", 0.0) or getattr(self, "foot_flat_l2", 0.0):
            down = self._feet_contact() > 0.5
            sin_t = (np.abs(self.sole_slope()[:, 1]) if self.foot_flat_axes == "roll" else np.sin(self.sole_angle())) * down
            fe = getattr(self, "foot_flat_free_ends", 0.0)
            if fe and min(self._phase % 0.5, 0.5 - self._phase % 0.5) < fe:
                sin_t = np.abs(self.sole_slope()[:, 1]) * down  # heel strike and toe-off: only the sideways roll pays
            info["foot_flat_cost"] = float(self.foot_flat_l1 * np.sum(np.abs(sin_t)) + self.foot_flat_l2 * np.sum(sin_t * sin_t))
            info["r_smooth"] = float(info["r_smooth"]) - max(0.0, min(1.0, self.smoothness_ramp)) * info["foot_flat_cost"]
        if getattr(self, "foot_drag_weight", 0.0):
            info["foot_drag"] = self.foot_drag_speed()
            info["r_gait_move"] = float(info["r_gait_move"]) - self.foot_drag_weight * info["foot_drag"]
        cw = getattr(self, "clock_height_weight", 0.0)
        if cw:
            err = self.foot_clearance() - self.clock_swing_target()
            info["clock_height_match"] = float(np.exp(-np.sum(err * err) / self.clock_height_std ** 2))
            info["r_gait_move"] = float(info["r_gait_move"]) + cw * info["clock_height_match"]
        if getattr(self, "bend_ramp_weight", 0.0) or getattr(self, "knee_lift_weight", 0.0) or getattr(self, "min_clear_penalty", 0.0):
            right_swings = self._phase >= 0.5  # first half of the cycle: right foot down, left swings
            swing, other = (0, 1) if right_swings else (1, 0)
            mid = 0.4 <= (self._phase % 0.5) / 0.5 <= 0.6
            extra = 0.0
            if self.bend_ramp_weight:
                d = self.foot_bend()
                info["bend_rel_m"] = d if right_swings else -d
                extra += mid * self.bend_ramp_weight * min(info["bend_rel_m"], self.bend_ramp_m) / self.bend_ramp_m
            if self.knee_lift_weight:
                flex = -self.data.qpos[[self.joint_qpos_adrs[k] for k in self._knee_idx]]  # the knee bends negative
                info["knee_extra_rad"] = float(flex[swing] - flex[other])
                extra += mid * self.knee_lift_weight * min(info["knee_extra_rad"], self.knee_lift_rad) / self.knee_lift_rad
            if self.min_clear_penalty and mid and self.foot_clearance()[swing] < self.min_clear_m:
                extra -= self.min_clear_penalty
            info["r_midswing"] = float(extra)
            info["r_gait_move"] = float(info["r_gait_move"]) + float(extra)
        if getattr(self, "ref_imitate_weight", 0.0):
            info["ref_match"] = self.ref_match()
            info["r_gait_move"] = float(info["r_gait_move"]) + self.ref_imitate_weight * info["ref_match"]
        if getattr(self, "_rf", None) is not None:
            terms = self.ref_full_terms()
            info["ref_full"] = float(sum(w * terms[k] for k, w in self.ref_full_w.items()))
            info["ref_full_foot_height"] = terms["foot_height"]
            info["r_gait_move"] = float(info["r_gait_move"]) + info["ref_full"]
        if getattr(self, "clock_bend_weight", 0.0):
            info["foot_bend_m"] = self.foot_bend()
            info["clock_bend_match"] = float(np.exp(-((info["foot_bend_m"] - self.clock_bend_target()) / self.clock_bend_std) ** 2))
            info["r_gait_move"] = float(info["r_gait_move"]) + self.clock_bend_weight * info["clock_bend_match"]
        if getattr(self, "swing_contact_weight", 0.0):
            info["swing_contact"] = float(np.sum(self._swing_acc) / max(1, self._swing_n))
            info["r_gait_move"] = float(info["r_gait_move"]) - self.swing_contact_weight * info["swing_contact"]
        if getattr(self, "straight_weight", 0.0):
            yaw_err, side = self.straight_error()
            info["straight_match"] = float(0.5 * np.exp(-(yaw_err / self.straight_yaw_std) ** 2) + 0.5 * np.exp(-(side / self.straight_y_std) ** 2))
            info["r_smooth"] = float(info["r_smooth"]) + self.straight_weight * info["straight_match"]
        if getattr(self, "raw_action_rate", 0.0) and self._raw_now is not None:
            prev = self._raw_now if self._raw_prev is None else self._raw_prev
            info["raw_action_rate"] = float(np.sum((self._raw_now - prev) ** 2))
            info["r_smooth"] = float(info["r_smooth"]) - max(0.0, min(1.0, self.smoothness_ramp)) * self.raw_action_rate * info["raw_action_rate"]
        if getattr(self, "hip_yaw_weight", 0.0):
            hz = self._hip_z_idx
            q = self.data.qpos[np.asarray(self.joint_qpos_adrs)[hz]]
            info["hip_yaw_match"] = float(np.exp(-np.mean(((q - self.default_qpos[hz]) / self.hip_yaw_std) ** 2)))
            info["r_smooth"] = float(info["r_smooth"]) + self.hip_yaw_weight * info["hip_yaw_match"]
        if getattr(self, "speed_margin_weight", 0.0):
            qd = np.abs(self.data.qvel[np.asarray(self.joint_dof_adrs)])
            info["speed_over"] = float(np.sum(np.maximum(0.0, qd / self.speed_margin_limit - 1.0) ** 2))
            info["r_smooth"] = float(info["r_smooth"]) - self.speed_margin_weight * info["speed_over"]
        if getattr(self, "com_support_weight", 0.0):
            d = self.com_ahead_of_support()
            info["com_support_match"] = 0.0 if d is None else float(np.exp(-(d / self.com_support_std) ** 2))
            info["r_smooth"] = float(info["r_smooth"]) + self.com_support_weight * info["com_support_match"]
        if getattr(self, "upright_weight", 0.0):
            pitch, roll = self.torso_pitch_roll()
            info["upright_match"] = float(np.exp(-(pitch / self.upright_pitch) ** 2 - (roll / self.upright_roll) ** 2))
            info["r_smooth"] = float(info["r_smooth"]) + self.upright_weight * info["upright_match"]
        if getattr(self, "stance_width_weight", 0.0):
            info["stance_width_m"] = self.stance_width_now()
            if getattr(self, "stance_band", None):
                out = max(self.stance_band[0] - info["stance_width_m"], info["stance_width_m"] - self.stance_band[1], 0.0)
                info["stance_width_out_m"] = float(out)
                info["stance_width_match"] = float(np.exp(-(out / self.stance_width_std) ** 2))
                info["r_smooth"] = float(info["r_smooth"]) - self.stance_band_penalty * min(1.0, out / self.stance_band_span)
            else:
                info["stance_width_match"] = float(np.exp(-((info["stance_width_m"] - self.stance_width_m) / self.stance_width_std) ** 2))
            info["r_smooth"] = float(info["r_smooth"]) + self.stance_width_weight * info["stance_width_match"]
        if getattr(self, "tilt_limit_penalty", 0.0):
            info["tilt_rad"] = float(np.arccos(np.clip(self.data.xmat[self.torso_body_id][8], -1.0, 1.0)))
            info["r_smooth"] = float(info["r_smooth"]) - self.tilt_limit_penalty * float(np.clip((info["tilt_rad"] - self.tilt_limit_rad) / self.tilt_limit_span, 0.0, 1.0))
        if getattr(self, "comfort_gate_m", 0.0):
            info["comfort_gate"] = self.comfort_gate_now()
            comfort = sum(w * info[k] for w, k in ((getattr(self, "upright_weight", 0.0), "upright_match"), (getattr(self, "stance_width_weight", 0.0), "stance_width_match"),
                                                   (getattr(self, "straight_weight", 0.0), "straight_match"), (getattr(self, "hip_yaw_weight", 0.0), "hip_yaw_match")) if w and k in info)
            info["r_smooth"] = float(info["r_smooth"]) - (1.0 - info["comfort_gate"]) * float(comfort)
        if getattr(self, "gait_always", False):
            info["r_gait_still"] = info["r_gait_move"]
        return info

    def com_ahead_of_support(self):
        """Whole-robot centre of mass minus the midpoint of the loaded feet, along the heading, in m; None with no foot loaded."""
        down = self._feet_contact() > 0.5
        if not down.any():
            return None
        d = self.data
        com = (self.model.body_mass[:, None] * d.xipos).sum(axis=0) / self.model.body_mass.sum()
        feet = d.geom_xpos[[g for g, k in zip(self.foot_geom_ids, down) if k]].mean(axis=0)
        w, x, y, z = (float(c) for c in d.qpos[3:7])
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return float((com[0] - feet[0]) * np.cos(yaw) + (com[1] - feet[1]) * np.sin(yaw))

    def comfort_gate_now(self) -> float:
        """0 to 1: how far the clock's swing foot is raised along the torso's axis, against frac of the clock's bump."""
        p = self._phase
        bump = self.comfort_gate_m * float(np.sin(np.pi * (p % 0.5) / 0.5))
        if bump < self.comfort_gate_min:
            return 1.0
        d = self.foot_bend()
        return float(np.clip((d if p >= 0.5 else -d) / (self.comfort_gate_frac * bump), 0.0, 1.0))

    def ref_match(self) -> float:
        """exp(-mean squared error / std^2) of hip pitch, knee and ankle pitch against the reference at the clock's phase."""
        from BipedRobot.fast.gait_ref import lookup
        q = self.data.qpos[[self.joint_qpos_adrs[k] for k in self._imit_idx]]
        e = q - lookup(self._imit_table, self._phase)[self._imit_idx]
        return float(np.exp(-np.mean(e * e) / self.ref_imitate_std ** 2))

    def _draw_vibration(self) -> tuple:
        """One episode's band-limited noise, [gyro (T, 3), joint speed (T, 12)] in rad/s, each channel at its std times one amplitude drawn for the episode and the difficulty ramp."""
        c, r, T = self._vib_cfg, self._noise_rng, int(self.max_episode_steps) + 8
        lo, hi = c.get("hz", (3.0, 20.0))
        amp = float(r.uniform(*c.get("amp", (0.3, 1.0)))) * self._difficulty

        def band(dims: int, std: float) -> np.ndarray:
            x = np.fft.rfft(r.normal(size=(T, dims)), axis=0)
            f = np.fft.rfftfreq(T, self.dt)
            x[(f < lo) | (f > hi)] = 0.0
            y = np.fft.irfft(x, n=T, axis=0)
            return (amp * float(std) * y / np.maximum(y.std(axis=0), 1e-9)).astype(np.float32)
        return band(3, c.get("gyro", 0.0)), band(self.num_joints, c.get("joint_vel", 0.0))

    def ref_full_terms(self) -> dict:
        """The designer's imitation kernels at the clock's phase, each 1 on the reference. exp(-mean(e^2) / std^2) for joints, foot origins in the torso frame, projected gravity, torso
        velocities. Foot height and contact take the worse foot, his `min`, so a limp does not keep half the reward: exp(-((h - ref) / std)^2) on the sole's lowest point, and
        1 - |c - ref| with c = max(loaded, 1 - h / contact_height_m), his soft score, so a dragged foot does not count as lifted."""
        from BipedRobot.fast.gait_ref import lookup
        d, p, s, rf = self.data, self._phase, self.ref_full_std, self._rf
        kern = lambda x, name, std: float(np.exp(-np.mean((x - lookup(rf[name], p)) ** 2) / std ** 2))
        R = d.xmat[self.torso_body_id].reshape(3, 3)
        feet = np.concatenate([R.T @ (d.xpos[b] - d.xpos[self.torso_body_id]) for b in self._rf_feet])
        h = np.maximum(self.foot_clearance(), 0.0)
        c = np.maximum((np.asarray(self._feet_contact()) > 0.5).astype(np.float64), np.clip(1.0 - h / self.ref_full_contact_m, 0.0, 1.0))
        return dict(joint_pos=kern(d.qpos[self.joint_qpos_adrs], "joint_pos", s["joint_pos"]), joint_vel=kern(d.qvel[self.joint_dof_adrs], "joint_vel", s["joint_vel"]),
                    foot_pos=kern(feet, "foot_pos", s["foot_pos"]), orientation=kern(R.T @ np.array([0.0, 0.0, -1.0]), "grav", s["orientation"]),
                    lin_vel=kern(R.T @ d.qvel[:3], "lin_vel", s["lin_vel"]), ang_vel=kern(d.qvel[3:6], "ang_vel", s["ang_vel"]),
                    foot_height=float(np.exp(-(((h - lookup(rf["foot_height"], p)) / s["foot_height"]) ** 2)).min()), contact=float((1.0 - np.abs(c - lookup(rf["contact"], p))).min()))

    def _reference_start(self):
        """Puts the robot on the reference at a random clock phase, feet on the floor, with the action filter and delay line holding that pose, and rebuilds the observation."""
        from BipedRobot.fast.gait_ref import lookup
        self.sole_tilt()  # fixes the sole reference from the standing pose first
        k0 = int(self.np_random.integers(0, int(round(1.0 / (self.dt * self.clock_hz)))))  # a phase the clock actually reads
        phi = (k0 * self.dt * self.clock_hz) % 1.0
        q = lookup(self._imit_table, phi)
        if getattr(self, "rsi_joint_noise", 0.0):
            q = np.clip(q + self.np_random.uniform(-self.rsi_joint_noise, self.rsi_joint_noise, q.shape), self.jnt_min, self.jnt_max)
        eps = 0.01
        qd = (lookup(self._imit_table, phi + eps) - lookup(self._imit_table, phi - eps)) * self.clock_hz / (2.0 * eps)
        d = self.data
        d.qvel[:] = 0.0
        d.qvel[0] = self._imit_speed if abs(float(self._desired_goal[0])) > 0.0 else 0.0
        for k, (a, b) in enumerate(zip(self.joint_qpos_adrs, self.joint_dof_adrs)):
            d.qpos[a], d.qvel[b] = q[k], qd[k]
        d.qpos[2] = 0.0
        mujoco.mj_forward(self.model, d)
        d.qpos[2] = self._root_height_for_floor_contact()
        mujoco.mj_forward(self.model, d)
        pos, neg = self.jnt_max - self.default_qpos, self.default_qpos - self.jnt_min
        a = np.clip(np.where(q >= self.default_qpos, (q - self.default_qpos) / (self.action_range_fraction * pos), (q - self.default_qpos) / (self.action_range_fraction * neg)), -1.0, 1.0).astype(np.float32)
        self._prev_action[:], self._prev_prev_action[:], self._action_filter_state[:] = a, a, a
        q32 = q.astype(np.float32)
        self._filtered_action, self._delayed_action, self._backlash_state = q32.copy(), q32.copy(), q32.copy()
        for i, (t, _) in enumerate(list(self._action_buffer)):
            self._action_buffer[i] = (t, a.copy())
        if self._hist is not None:
            self._hist.clear()
        self._last_angles = None
        if getattr(self, "_slow_yaw", None) is not None:
            self._slow_yaw.clear()
        self._start_pose = None
        self._clock_k = k0
        return self._get_obs_dict()

    def foot_bend(self) -> float:
        """Right foot's height above the left along the torso's vertical axis, less the standing pose's, in metres. A rigid tilt of the whole robot leaves it unchanged."""
        if self._bend0 is None:
            d0 = mujoco.MjData(self.model)
            for adr, q in zip(self.joint_qpos_adrs, self.default_qpos):
                d0.qpos[adr] = q
            mujoco.mj_kinematics(self.model, d0)
            self._bend0 = self._foot_bend_in(d0)
        return self._foot_bend_in(self.data) - self._bend0

    def _foot_bend_in(self, d) -> float:
        r, l = self.foot_geom_ids
        return float(d.xmat[self.torso_body_id].reshape(3, 3)[:, 2] @ (d.geom_xpos[r] - d.geom_xpos[l]))

    def clock_bend_target(self) -> float:
        """The clock's bump for foot_bend: positive while the right foot swings, negative while the left does."""
        h, self.clock_height_m = self.clock_height_m, self.clock_bend_m
        try:
            t = self.clock_swing_target()
        finally:
            self.clock_height_m = h
        return float(t[0] - t[1])

    def stance_width_now(self) -> float:
        """Sideways distance between the two foot centres, across the heading, in metres."""
        a, b = (self.data.geom_xpos[g] for g in self.foot_geom_ids)
        w, x, y, z = (float(c) for c in self.data.qpos[3:7])
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return float(abs(-(a[0] - b[0]) * np.sin(yaw) + (a[1] - b[1]) * np.cos(yaw)))

    def torso_pitch_roll(self) -> tuple:
        """Torso pitch (nose down positive) and roll in radians, from the base quaternion."""
        w, x, y, z = (float(c) for c in self.data.qpos[3:7])
        return float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))), float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))

    def straight_error(self) -> tuple:
        """(heading change since reset, wrapped, in rad; sideways offset from the start position along the start heading, in m, left positive)."""
        q = self.data.qpos
        w, x, y, z = (float(c) for c in q[3:7])
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        if self._start_pose is None:
            self._start_pose = (yaw, float(q[0]), float(q[1]))
        y0, px, py = self._start_pose
        err = float(np.remainder(yaw - y0 + np.pi, 2 * np.pi) - np.pi)
        side = -np.sin(y0) * (float(q[0]) - px) + np.cos(y0) * (float(q[1]) - py)
        return err, float(side)

    def swing_substep(self, fn: np.ndarray) -> None:
        """Called by GaitCostEnv at every physics substep with each foot's normal force: counts feet carrying weight while the clock swings them."""
        if self._weight_n is None:
            self._weight_n = float(self.model.body_mass.sum()) * abs(float(self.model.opt.gravity[2]))
        p, m = self._phase, self.swing_contact_margin
        d = getattr(self, "clock_stance", 0.5) - 0.5
        swing = np.array([0.5 + d + m <= (p + d / 2) % 1.0 < 1.0 - m, 0.5 + d + m <= (p - 0.5 + d / 2) % 1.0 < 1.0 - m])
        self._swing_acc += swing & (np.asarray(fn) > self.swing_contact_load * self._weight_n)
        self._swing_n += 1

    def foot_clearance(self) -> np.ndarray:
        """Height of each foot's lowest mesh point above the floor, [right, left], as evaluation_suite_v2 measures it."""
        d, m = self.data, self.model
        if getattr(self, "_foot_verts", None) is None:
            self._foot_verts = [m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]].astype(np.float64)
                                for g in self.foot_geom_ids]
        return np.array([float(np.min(d.geom_xpos[g, 2] + self._foot_verts[k] @ d.geom_xmat[g].reshape(3, 3)[2, :])) for k, g in enumerate(self.foot_geom_ids)])

    def sole_tilt(self) -> np.ndarray:
        """Angle of each sole from its orientation at reset, [right, left], in radians. The first call after reset fixes that orientation."""
        d = self.data
        R = [d.geom_xmat[g].reshape(3, 3) for g in self.foot_geom_ids]
        if getattr(self, "_sole_normal", None) is None:
            self._sole_normal = [r.T @ np.array([0.0, 0.0, 1.0]) for r in R]
        return np.array([float(np.arccos(np.clip((r @ n)[2], -1.0, 1.0))) for r, n in zip(R, self._sole_normal)])

    def foot_drag_speed(self) -> float:
        """Sum over feet off the floor of (1 - clip(clearance / foot_drag height, 0, 1)) * horizontal speed of the foot, in m/s."""
        clear, down = self.foot_clearance(), self._feet_contact() > 0.5
        out, vel = 0.0, np.zeros(6)
        for k, g in enumerate(self.foot_geom_ids):
            if down[k]:
                continue
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, g, vel, 0)
            out += (1.0 - float(np.clip(clear[k] / self.foot_drag_h, 0.0, 1.0))) * float(np.hypot(vel[3], vel[4]))
        return out

    def sole(self) -> list:
        """Each foot's sole from its mesh, computed once: the flat bottom face (the faces pointing down at the first call, which is a reset, grouped
        within 3 deg of the largest one), as (outward normal pointing down, the face's vertices, length axis pointing to the toe, width axis pointing
        left), all in the foot's own frame."""
        if getattr(self, "_sole", None) is None:
            m, d = self.model, self.data
            heading = d.xmat[self.torso_body_id].reshape(3, 3)[:, 0].copy()
            heading[2] = 0.0
            heading /= max(np.linalg.norm(heading), 1e-9)
            soles = []
            for g in self.foot_geom_ids:
                mid = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
                F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
                area = 0.5 * np.linalg.norm(cr, axis=1)
                n = cr / np.maximum(2.0 * area[:, None], 1e-12)
                R = d.geom_xmat[g].reshape(3, 3)
                down = n @ (R.T @ np.array([0.0, 0.0, -1.0])) > np.cos(np.radians(20.0))
                i0 = int(np.argmax(np.where(down, area, 0.0)))
                same = down & (n @ n[i0] > np.cos(np.radians(3.0)))
                ns = (n[same] * area[same, None]).sum(axis=0)
                ns /= np.linalg.norm(ns)
                h = V @ ns
                on = V[h > h.max() - 0.0005]
                x = on - on.mean(axis=0)
                x -= np.outer(x @ ns, ns)
                fwd = np.linalg.eigh(x.T @ x)[1][:, -1]
                fwd -= (fwd @ ns) * ns
                fwd /= np.linalg.norm(fwd)
                if (R @ fwd) @ heading < 0.0:
                    fwd = -fwd
                soles.append((ns, on, fwd, np.cross(-ns, fwd)))
            self._sole = soles
        return self._sole

    def sole_angle(self) -> np.ndarray:
        """Angle of each sole to the floor, [right, left], in radians: 0 when it lies flat."""
        d = self.data
        return np.array([float(np.arccos(np.clip(-(d.geom_xmat[g].reshape(3, 3) @ ns)[2], -1.0, 1.0))) for g, (ns, _, _, _) in zip(self.foot_geom_ids, self.sole())])

    def sole_gap(self) -> np.ndarray:
        """How far each sole's highest point is above its lowest, [right, left], in m: 0 when it lies flat, about 2 mm for a 1.3 deg tilt along a 9 cm sole."""
        d = self.data
        z = [d.geom_xpos[g, 2] + on @ d.geom_xmat[g].reshape(3, 3)[2, :] for g, (_, on, _, _) in zip(self.foot_geom_ids, self.sole())]
        return np.array([float(x.max() - x.min()) for x in z])

    def sole_slope(self) -> np.ndarray:
        """Each sole's up-normal along its own length and width axes, [[along, across] right, [...] left]. along < 0: toe raised, the heel lower;
        across < 0: left edge raised, the right edge lower."""
        d = self.data
        out = []
        for g, (ns, _, f, l) in zip(self.foot_geom_ids, self.sole()):
            R = d.geom_xmat[g].reshape(3, 3)
            up, fw, lf = -(R @ ns), R @ f, R @ l
            up_h = up - up[2] * np.array([0.0, 0.0, 1.0])
            out.append((float(up_h @ fw), float(up_h @ lf)))
        return np.array(out)

    def gait_ref_pose(self, t: float) -> np.ndarray:
        """The pattern's joint angles for an action issued at time t after reset: standing before 0, ramping in over ref_ramp_s."""
        from BipedRobot.fast.gait_ref import lookup
        ramp = float(np.clip(t / self.ref_ramp_s, 0.0, 1.0)) if self.ref_ramp_s > 0 else float(t >= 0.0)
        base = self.default_qpos.astype(np.float64)
        return base + ramp * (lookup(self._ref_table, t * self.clock_hz) - base)

    def _retrieve_delayed_action(self, current_time: float, delay_sec: float) -> np.ndarray:
        self._ref_t = current_time - delay_sec  # the delayed action was issued then, and the pattern is read at the same time
        return super()._retrieve_delayed_action(current_time, delay_sec)

    def _action_to_target(self, action: np.ndarray) -> np.ndarray:
        if getattr(self, "_ref_table", None) is None:
            return super()._action_to_target(action)
        target = self.gait_ref_pose(self._ref_t) + np.asarray(action, np.float64) * self.ref_residual * self._ref_half
        return np.clip(target, np.maximum(self.jnt_min, self.ctrl_min), np.minimum(self.jnt_max, self.ctrl_max)).astype(np.float32)

    def clock_stance_feet(self, p: float) -> tuple:
        """(right in stance, left in stance) at clock phase p. Right stance starts at -d/2, left at 0.5 - d/2, each lasting clock_stance of the cycle."""
        s = getattr(self, "clock_stance", 0.5)
        d = s - 0.5
        return bool((p + d / 2) % 1.0 < s), bool((p - 0.5 + d / 2) % 1.0 < s)

    def clock_swing_target(self) -> np.ndarray:
        """Foot heights the clock asks for, [right, left]. First half of a cycle: right foot down, left foot on a sine bump; second half the other way round."""
        p, h = self._phase, self.clock_height_m
        st = getattr(self, "clock_stance", 0.5)
        if st != 0.5:  # validity.clock_stance: the bump spans each foot's shorter swing
            d = st - 0.5
            out = np.zeros(2)
            for k, shift in enumerate((d / 2, -0.5 + d / 2)):
                q = (p + shift) % 1.0
                if q >= st:
                    out[k] = h * np.sin(np.pi * (q - st) / (1.0 - st))
            return out
        if p < 0.5:
            return np.array([0.0, h * np.sin(2.0 * np.pi * p)])
        return np.array([h * np.sin(2.0 * np.pi * (p - 0.5)), 0.0])

    def relative_track(self, achieved: np.ndarray, desired: np.ndarray) -> np.ndarray:
        """Tracking kernel whose width follows the command (reward.track_rel), on rows of [vx, vy, wz]."""
        v_rel, v_min, w_rel, w_min = self.track_rel
        a, g = np.asarray(achieved, np.float32)[:, :3], np.asarray(desired, np.float32)[:, :3]
        sv = np.maximum(v_rel * np.linalg.norm(g[:, :2], axis=1), v_min)
        sw = np.maximum(w_rel * np.abs(g[:, 2]), w_min)
        v, w = a[:, :2] - g[:, :2], a[:, 2] - g[:, 2]
        return (np.exp(-np.sum(v * v, axis=1) / sv ** 2) * np.exp(-(w * w) / sw ** 2)).astype(np.float32)

    def _tracking_reward_batch(self, achieved: np.ndarray, desired: np.ndarray) -> np.ndarray:
        if getattr(self, "track_rel", None):
            return self.relative_track(achieved, desired)
        return super()._tracking_reward_batch(achieved, desired)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        res = super().compute_reward(achieved_goal, desired_goal, info)
        if not getattr(self, "track_rel", None) or self.reward_mode != "track" or self.adaptive_reward_fn is not None:
            return res
        batch = np.asarray(achieved_goal).ndim > 1
        a = np.atleast_2d(np.asarray(achieved_goal, np.float32))
        g = np.atleast_2d(np.asarray(desired_goal, np.float32))
        # Swap the base env's Gaussian tracking term for the relative one. Rows that fell keep the fall penalty.
        swapped = np.asarray(res, np.float32) - tracking_reward_batch(a, g, sigma_v=self.sigma_v, sigma_w=self.sigma_w) + self.relative_track(a, g)
        out = np.where(_fell_mask(info, len(a)), np.asarray(res, np.float32), swapped).astype(np.float32)
        return out if batch else float(out[0])

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._noise_rng = np.random.default_rng(seed)
            self._focus_rng = np.random.default_rng(seed + 1)
        if self._hist is not None:
            self._hist.clear()
        self._last_angles = None
        self._clock_k = 0
        self._sole_normal = None  # the reset pose stands on flat feet; the first sole_tilt() after it fixes the reference
        if getattr(self, "_slow_yaw", None) is not None:
            self._slow_yaw.clear()
        self._start_pose = None
        self._weight_n = None
        self._raw_prev = self._raw_now = None
        if getattr(self, "_bias_cfg", None):
            r, c, n = self._noise_rng, self._bias_cfg, self.num_joints
            d = self._difficulty
            self._bias = {"joint_pos": d * r.uniform(-c.get("joint_pos", 0.0), c.get("joint_pos", 0.0) + 1e-12, n).astype(np.float32),
                          "gyro": d * r.uniform(-c.get("gyro", 0.0), c.get("gyro", 0.0) + 1e-12, 3).astype(np.float32),
                          "gravity": d * r.uniform(-c.get("gravity", 0.0), c.get("gravity", 0.0) + 1e-12, 3).astype(np.float32)}
        if getattr(self, "_vib_cfg", None) and self.realism == "full":
            self._vib, self._vib_k = self._draw_vibration(), 0
        if getattr(self, "com_offset", None):
            ox, oy = self.com_offset
            if seed is not None:
                self._com_rng = np.random.default_rng(seed + 7)
            r = self._com_rng
            self.model.body_ipos[self.torso_body_id] = self._torso_ipos0 + self._difficulty * np.array([r.uniform(-ox, ox), r.uniform(-oy, oy), 0.0])
        out = super().reset(seed=seed, options=options)
        if getattr(self, "duty_range", None):
            lo, hi = self.duty_range
            u = float(self.np_random.uniform(lo, hi)) if hi > lo else lo
            self._duty = 1.0 - self._difficulty * (1.0 - u)
            self._stall, self._cap, self._curve_w0 = self._stall0 * self._duty, self._cap0 * self._duty, self._curve_w00 * self._duty
            self._priv[42 + 3] = (float(np.mean(self._torque_scale)) if self._torque_scale is not None else 1.0) * self._duty
        if getattr(self, "_imit_table", None) is not None and self.rsi_prob > 0.0 and not (options and "desired_goal" in options) and self.np_random.random() < self.rsi_prob:
            out = (self._reference_start(), out[1])
        if self._planner_on:
            self._plan_p = self.data.qpos[self.joint_qpos_adrs].astype(np.float64)
            self._plan_v = np.zeros(self.num_joints)
        self._play_reset()
        if getattr(self, "delay_zoh", False):  # validity.delay_mode zoh: empty queue, the actuators hold the home-pose target the delay buffer starts from
            self._zq, self._zoh_new, self._zoh_s = [], False, 0
            self._zoh_prev = np.clip(self.default_qpos, self.ctrl_min, self.ctrl_max).astype(np.float64)
        return out

    def _play_reset(self) -> None:
        """Draws this episode's gear play and starts the motor side where the joint is. With play off, puts back the model's own armature and damping."""
        ids, dofs = self.actuator_ids, self.joint_dof_adrs
        if not getattr(self, "_play_on", False):
            if getattr(self, "_arm0", None) is not None:
                self.model.dof_armature[dofs] = self._arm0
                self.model.actuator_biasprm[ids, 2] = -self._kv0
                if getattr(self, "_arm_range", None) and self.realism == "full" and self.dr_enabled:  # validity.servo_model.armature_range
                    lo, hi = self._arm_range
                    self.model.dof_armature[dofs] = self._arm0 + self._difficulty * (self.np_random.uniform(lo, hi, self.num_joints) - self._arm0)
            return
        c = self._play_cfg
        lo, hi = (float(x) for x in c.get("deg", (1.0, 2.5)))
        self._play_half = 0.5 * self._difficulty * np.radians(self.np_random.uniform(lo, hi, self.num_joints))
        self._play_k, self._play_c = float(c.get("stiffness", 80.0)), float(c.get("damping", 1.0))
        self._play_J = float(c.get("motor_inertia", 0.0234))
        self.model.dof_armature[dofs] = float(c.get("link_armature", 0.005))
        self.model.actuator_biasprm[ids, 2] = 0.0  # back-EMF damping acts on the motor side here, not on the joint
        self._play_m = self.data.qpos[self.joint_qpos_adrs].astype(np.float64).copy()
        self._play_w = np.zeros(self.num_joints)

    def _play_limits(self, w: np.ndarray) -> tuple:
        """Motor torque limits at motor speed w: the torque-speed line when validity.servo_curve is set, else the flat limit."""
        scale = self._torque_scale if self._torque_scale is not None else np.ones(self.num_joints)
        if self._curve_w0 is None:
            lim = self.torque_limit * scale
            return -lim, lim
        limit = self._cap * scale
        head = np.minimum(limit, np.clip(1.0 - np.abs(w) / self._curve_w0, 0.0, 1.0) * self._stall * scale)
        return -np.where(w < 0, head, limit), np.where(w > 0, head, limit)

    def _servo_play(self, m, d) -> None:
        """One physics substep of the servo with gear play. The first substep of a control step takes the target that HERLocoEnv.step (or the move
        planner, every substep) wrote to ctrl; the gear torque reaches the joint through its position actuator, with ctrl set so kp (ctrl - q) equals it."""
        ids = self.actuator_ids
        if self._planner_on or self._play_new:
            self._play_target = d.ctrl[ids].astype(np.float64)
            self._play_new = False
        q, qd = d.qpos[self.joint_qpos_adrs], d.qvel[self.joint_dof_adrs]
        kp = m.actuator_gainprm[ids, 0]
        lo, hi = self._play_limits(self._play_w)
        tau_s = np.clip(kp * (self._play_target - q) - self._kv0 * self._play_w, lo, hi)
        x = self._play_m - q
        over = np.abs(x) - self._play_half
        tau_c = np.where(over > 0.0, self._play_k * np.sign(x) * over + self._play_c * (self._play_w - qd), 0.0)
        h = m.opt.timestep
        self._play_w = self._play_w + h * (tau_s - tau_c) / self._play_J
        self._play_m = self._play_m + h * self._play_w
        m.actuator_forcerange[ids, 0], m.actuator_forcerange[ids, 1] = -20.0, 20.0  # the mesh, not the motor, sets this torque
        d.ctrl[ids] = q + tau_c / kp

    def _servo_zoh(self, m, d) -> None:
        """validity.delay_mode zoh, first at every physics substep: queue the target HERLocoEnv.step wrote for this control step, then put the newest
        queued target whose time has come into ctrl, before the move planner, servo_play or servo_curve read it."""
        ids = self.actuator_ids
        k = self._step_count * self.frame_skip + self._zoh_s
        self._zoh_s += 1
        if self._zoh_new:
            self._zq.append((self._step_count * self.frame_skip + int(np.ceil(self._active_delay / self.sim_dt - 1e-9)), d.ctrl[ids].astype(np.float64).copy()))
            self._zoh_new = False
        changed = False
        while self._zq and self._zq[0][0] <= k:
            self._zoh_prev = self._zq.pop(0)[1]
            changed = True
        d.ctrl[ids] = self._zoh_prev
        if changed:
            self._plan_new = True
            self._play_new = True
        if self._zoh_cb is not None:
            self._zoh_cb(m, d)

    def _servo_all(self, m, d) -> None:
        if self._planner_on:
            self._servo_planner(m, d)
        if getattr(self, "_play_on", False):
            self._servo_play(m, d)
        elif self._curve_w0 is not None:
            self._servo_curve(m, d)

    def step(self, action):
        self._ref_t = self._step_count * self.dt  # when delay is off, the action is used at once
        self._swing_acc[:] = 0.0
        self._swing_n = 0
        if getattr(self, "push_range", None) and self.push_enabled and self._step_count > 0 and self._step_count % self.dr_push_interval == 0:
            self.dr_push_magnitude = self._difficulty * float(self.np_random.uniform(*self.push_range))  # read by HERLocoEnv.step for this push
        if getattr(self, "raw_action_rate", 0.0):
            self._raw_prev = self._raw_now
            self._raw_now = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        play = getattr(self, "_play_on", False)
        zoh = getattr(self, "delay_zoh", False) and self.enable_delay
        if self._curve_w0 is None and not self._planner_on and not play and not zoh:
            return super().step(action)
        self._plan_new = True
        self._play_new = True
        cb = self._servo_all if play else (self._servo_both if self._planner_on else (self._servo_curve if self._curve_w0 is not None else None))
        if zoh:  # validity.delay_mode zoh: HERLocoEnv's chain runs without its interpolating delay; _servo_zoh delays the held target instead
            self._zoh_cb, self._zoh_new, self._zoh_s = cb, True, 0
            self.enable_delay = False
            cb = self._servo_zoh
        mujoco.set_mjcb_control(cb)
        try:
            return super().step(action)
        finally:
            if zoh:
                self.enable_delay = True
            mujoco.set_mjcb_control(None)
            if play and self._play_target is not None:
                lo, hi = self._play_limits(self._play_w)
                self.model.actuator_forcerange[self.actuator_ids, 0], self.model.actuator_forcerange[self.actuator_ids, 1] = lo, hi
                self.data.ctrl[self.actuator_ids] = self._play_target
