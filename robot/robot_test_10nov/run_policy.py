#!/usr/bin/env python3
"""Run a trained policy on the real robot at 50 Hz. Runs on the Pi. No ROS and no torch.

NOT YET RUN ON REAL SERVOS. Before the first run:
  1. Run robot_test.py and send the results. They show whether the directions, timing and servo response match training.
  2. Stop servo_bridge. Only one program can use the servos.
  3. Keep a hand on the power switch. Ctrl-C or any problem turns torque OFF and the legs drop.

Copy run_policy.py, robot_test.py and policy.npz (from export_policy.py) to the Pi. First find how the IMU is mounted. No torque, you tilt the robot by hand when asked:
    python3 run_policy.py policy.npz --stage imu --biped-ws ~/biped_ws

Then start with the robot hanging, using the --imu-mount that step printed:
    python3 run_policy.py policy.npz --stage hang --speed 0.2 --budget 1,1,1 --imu-mount x,y,z --biped-ws ~/biped_ws

Stages, each one only after the last has looked right:
    hang       feet off the ground, torque limit 500, actions scaled to 30%, 20 s
    supported  feet touching, robot held by a harness or a person, limit 700, full actions, 15 s
    ground     harness slack, limit 1000, full actions, 15 s. Asks you to type GO.

It stops with torque off on: a servo or the IMU not answering, heat, low voltage (11.1 V), tilt over 35 degrees, a joint far from where it was told to go,
a late control loop, Ctrl-C, or the SSH link dropping. It logs every loop to a new folder and writes nothing to the servos' permanent memory.
A policy that reads true speed or budget balances is refused, because the robot cannot measure them.
Testing without a robot: add --mock (results are meaningless).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import sys
import tarfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from robot_test import (Abort, Guard, SLOW, FAST, brake, clear_accel_cap, set_servo_gains, RATE_HZ, STEPS_PER_RAD, checked_steps, find_hardware_code, gravity_in_body, open_imu, read_one, safe_print,
                        read_settings, signed_delta, stop_on_signal, torque_off)

STAGES = {"hang": dict(limit=500, seconds=20.0, scale=0.3),
          "supported": dict(limit=700, seconds=15.0, scale=1.0),
          "ground": dict(limit=1000, seconds=15.0, scale=1.0)}
MAX_SPEED = 0.3              # the fastest forward command in training, m/s
TILT_LIMIT_DEG = 35.0        # stop well before a fall (training ended an episode at about 49 degrees); 7 Nov: was 45
MIN_VOLTS = 11.1             # 7 Nov: stop below this battery voltage (hardware.yaml's own floor is lower)
EXPECT_PID = (32, 32, 0)     # the stored P, D, I measured on all 12 servos on 6 Nov
POSE_TOLERANCE_DEG = 5.0
TRAIN_NAMES = ["r_hip_z", "r_hip_x", "r_hip_y", "r_knee", "r_ankle_y", "r_ankle_x", "l_hip_z", "l_hip_x", "l_hip_y", "l_knee", "l_ankle_y", "l_ankle_x"]
# The pose check (stage poses), robot hanging: each pose is the home pose with one joint moved, by training-joint name. What the person should see comes
# from the simulator model: l_knee -20 swings the left foot about 39 mm back and 13 mm up; r_hip_y +10 swings the right foot about 38 mm outward.
POSES = (("home", None, 0.0, "Both legs in the home pose, feet level and side by side?"),
         ("left knee", "l_knee", -20.0, "Only the robot's LEFT knee bent about 20 degrees more, its foot swung back and up?"),
         ("right hip roll", "r_hip_y", 10.0, "Only the robot's RIGHT leg swung sideways about 10 degrees, its foot moved OUTWARD, away from the other leg?"))
DELAY_MOVE_DEG = 0.35        # --stage delay: this much movement counts as the joint having started (about 4 encoder steps)
TRAINED_DELAY_MS = 40.0      # the delay range the newest policies trained over
JOINT_ERROR_RAD = 0.7        # a joint this far from its target ...
JOINT_ERROR_S = 0.5          # ... for this long is treated as stuck or runaway
MAX_MISSED_READS = 3         # consecutive loops one servo may miss
IMU_STALE_LOOPS = 10         # the driver repeats its last reading when a read fails
LATE_LOOP_FACTOR = 1.5
MAX_LATE_LOOPS = 5           # consecutive loops longer than LATE_LOOP_FACTOR control steps
ABSURD_LOOP_S = 0.1          # one loop this long stops at once
HOME_TOLERANCE_DEG = 5.0
LOG_COLUMNS = (["t_s", "dt_ms", "scale"] + [f"action_{i}" for i in range(12)] + [f"target_{i}" for i in range(12)] + [f"q_{i}" for i in range(12)] + [f"qd_{i}" for i in range(12)]
               + ["gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z", "min_volts", "max_temp_c", "work_ms", "missed_reads"])


class PolicyCore:
    """Sensors in, joint targets out, with the training env's order and scaling. No hardware, so tests can compare it with the env number by number."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.layers = [(z["w0"], z["b0"]), (z["w1"], z["b1"])]
        self.mu = (z["mu_w"], z["mu_b"])
        self.keep = z["keep"]
        self.obs_dim, self.history = int(z["obs_dim"]), int(z["history"])
        self.scale = z["scale"]  # joint_pos, joint_vel, gyro, gravity, prev_action
        self.default, self.lo, self.hi = z["default"], z["lo"], z["hi"]
        self.ctrl_min, self.ctrl_max = z["ctrl_min"], z["ctrl_max"]
        self.fraction, self.alpha = float(z["fraction"]), float(z["alpha"])
        self.servo_names = [str(x) for x in z["servo_names"]]
        self.meta = json.loads(str(z["meta"]))
        self.reads_truth = bool(z["reads_truth"])
        self.clock_hz = float(z["clock_hz"]) if "clock_hz" in z.files else 0.0
        # validity.slow_yaw_s in training: one more input after the clock, 5 x the mean sensed yaw rate over the last slow_yaw_n control steps
        self.slow_yaw_n = int(z["slow_yaw_n"]) if "slow_yaw_n" in z.files else 0
        # GAIT REFERENCE policies (validity.gait_ref in training): joint target = walking pattern at the gait phase + action * residual * half the range
        self.ref_table = z["ref_table"].astype(np.float64) if "ref_table" in z.files else None
        if self.ref_table is not None:
            if not self.clock_hz:
                raise ValueError("a gait-reference policy needs its gait clock")
            self.ref_residual, self.ref_ramp_s = float(z["ref_residual"]), float(z["ref_ramp_s"])
            self.ref_half = (self.hi.astype(np.float64) - self.lo) / 2.0
        # Exactly left-right symmetric policies (equivariant_actor in training): the action is the network averaged with its mirror image
        self.mirror = (z["mirror_obs_perm"], z["mirror_obs_sign"], z["mirror_act_perm"]) if "mirror_obs_perm" in z.files else None
        self.dt = 1.0 / RATE_HZ
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.history)
        self.filtered = np.zeros(12, np.float32)
        self.clock_k = 0
        self.ref_k = 0
        self.slow_yaw = deque(maxlen=max(1, self.slow_yaw_n))
        self.ref_scale = 1.0

    def frame(self, q, qd, gyro, gravity):
        s = self.scale
        return np.concatenate(((q - self.default) * s[0], qd * s[1], gyro * s[2], gravity * s[3], self.filtered * s[4])).astype(np.float32)

    def _mean(self, x):
        for w, b in self.layers:
            x = np.maximum(x @ w.T + b, 0.0)
        return x @ self.mu[0].T + self.mu[1]

    def forward(self, x):
        return np.tanh(self._mean(x))

    def policy_action(self, obs):
        """The action for a whole observation. Without a mirror this is forward on the policy's inputs."""
        if self.mirror is None:
            return self.forward(obs[self.keep])
        perm, sign, act_perm = self.mirror
        return np.tanh(0.5 * (self._mean(obs[self.keep]) + self._mean((obs[perm] * sign)[self.keep])[act_perm]))

    def ref_pose(self, t):
        """The pattern's joint angles for an action issued t seconds after the start: standing before 0, ramping in over ref_ramp_s, as in training.
        ref_scale shrinks the pattern's swing about the standing pose along with the actions (the hang stage moves at 30%)."""
        ramp = min(max(t / self.ref_ramp_s, 0.0), 1.0) if self.ref_ramp_s > 0 else float(t >= 0.0)
        n = len(self.ref_table)
        x = ((t * self.clock_hz) % 1.0) * n
        i = int(math.floor(x)) % n
        f = x - math.floor(x)
        base = self.default.astype(np.float64)
        return base + self.ref_scale * ramp * ((1.0 - f) * self.ref_table[i] + f * self.ref_table[(i + 1) % n] - base)

    def to_target(self, filtered):
        if self.ref_table is not None:
            target = self.ref_pose(self.ref_k * self.dt) + filtered.astype(np.float64) * self.ref_residual * self.ref_half
            return np.clip(target, np.maximum(self.lo, self.ctrl_min), np.minimum(self.hi, self.ctrl_max)).astype(np.float32)
        return np.clip(np.where(filtered >= 0.0, self.default + filtered * self.fraction * (self.hi - self.default),
                                self.default + filtered * self.fraction * (self.default - self.lo)), self.ctrl_min, self.ctrl_max).astype(np.float32)

    def step(self, q, qd, gyro, gravity, command, u, scale=1.0, achieved=None, levels=None):
        """One control step: returns (action, joint targets in radians). `achieved` and `levels` exist only for tests of policies that read them."""
        f = self.frame(q, qd, gyro, gravity)
        if not self.frames:
            self.frames.extend([f] * self.history)
        else:
            self.frames.append(f)
        obs = np.zeros(self.obs_dim, np.float32)
        obs[3:6], obs[6:9] = command, u
        obs[9:9 + 42 * self.history] = np.concatenate(self.frames)
        if self.clock_hz:
            angle = 2.0 * math.pi * ((self.clock_k * self.dt * self.clock_hz) % 1.0)
            obs[9 + 42 * self.history:9 + 42 * self.history + 2] = math.sin(angle), math.cos(angle)
            self.clock_k += 1
        if self.slow_yaw_n:
            self.slow_yaw.append(float(gyro[2]))
            obs[9 + 42 * self.history + (2 if self.clock_hz else 0)] = 5.0 * float(np.mean(self.slow_yaw))
        obs[-3:] = 1.0
        if achieved is not None:
            obs[:3] = achieved
        if levels is not None:
            obs[-3:] = levels
        self.last_obs = obs
        action = self.policy_action(obs) * scale
        self.filtered = (self.alpha * action + (1.0 - self.alpha) * self.filtered).astype(np.float32)
        self.ref_scale = scale
        target = self.to_target(self.filtered)
        self.ref_k += 1
        return action, target


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def heading_command(q, yaw0, gain=1.0, max_wz=0.15):
    """Yaw command that holds the starting heading, from the IMU's orientation. 0.15 rad/s is the largest yaw command in training."""
    return max(-max_wz, min(max_wz, -gain * math.remainder(yaw_of(q) - yaw0, 2 * math.pi)))


def imu_mount(text):
    """'x,y,z' says which sensor axis lies along the robot's forward, left and up axes. Example: '-y,x,z'. Returns the matrix from sensor to robot."""
    axes = text.replace(" ", "").split(",")
    m = np.zeros((3, 3))
    if len(axes) != 3:
        raise ValueError(f"--imu-mount needs three axes, got {text!r}")
    for row, a in enumerate(axes):
        sign = -1.0 if a.startswith("-") else 1.0
        name = a.lstrip("+-")
        if name not in ("x", "y", "z"):
            raise ValueError(f"--imu-mount: {a!r} is not an axis")
        m[row, "xyz".index(name)] = sign
    if abs(abs(np.linalg.det(m)) - 1.0) > 1e-9 or np.linalg.det(m) < 0:
        raise ValueError(f"--imu-mount {text!r} is not a rotation")
    return m


def solve_mount(upright, forward, left):
    """--imu-mount from gravity in sensor axes with the robot upright, tilted forward and tilted left. In the robot's axes, gravity reads (0, 0, -1) upright and gains +x tilted
    forward and +y tilted left. Returns the text, or raises ValueError if the tilts do not give a clear answer."""
    up = -np.asarray(upright, float)
    fwd, lft = np.asarray(forward, float) - upright, np.asarray(left, float) - upright
    if min(np.linalg.norm(fwd), np.linalg.norm(lft)) < 0.15:
        raise ValueError("the tilts were too small to read. Tilt about 20 degrees and hold still")
    names = []
    for v in (fwd, lft, up):
        i = int(np.abs(v).argmax())
        if abs(v[i]) < 0.7 * np.linalg.norm(v):
            raise ValueError(f"direction {np.round(v, 2).tolist()} is not close to a sensor axis. Tilt straight, or the IMU is mounted at an angle, which --imu-mount cannot describe")
        names.append(("-" if v[i] < 0 else "") + "xyz"[i])
    text = ",".join(names)
    imu_mount(text)
    return text


def imu_stage(imu, ask, say, wait=2.0, sleep=time.sleep):
    """Guided check, no torque. Prints the --imu-mount to use, then checks the gyro signs with it."""
    def hold(text):
        ask(f"{text} Press Enter and hold still. ")
        sleep(wait)
        rows = []
        for _ in range(25):
            rows.append(gravity_in_body(imu.read().quaternion))
            sleep(0.02)
        return np.mean(rows, axis=0)

    def spin(text, mount, axis, sign):
        ask(f"{text} Press Enter, then do it during the next 2 seconds. ")
        rows = []
        for _ in range(100):
            rows.append(mount @ np.asarray(imu.read().gyro, float))
            sleep(0.02)
        total = np.asarray(rows).sum(axis=0) * 0.02
        ok = abs(total[axis]) > 0.15 and np.sign(total[axis]) == sign and abs(total[axis]) >= np.abs(total).max() - 1e-9
        say(f"  turned {np.round(total, 2).tolist()} rad about the robot's x, y, z: " + ("OK" if ok else "NOT what was expected"))
        return ok

    say("Hold the robot in the air. Forward is the way its toes point.")
    upright = hold("Hold the robot upright.")
    forward = hold("Tilt it forward about 20 degrees, nose down.")
    left = hold("Back upright, then tilt it to ITS left about 20 degrees, left side down.")
    try:
        text = solve_mount(upright, forward, left)
    except ValueError as exc:
        raise Abort(str(exc)) from exc
    mount = imu_mount(text)
    say(f"IMU mount: --imu-mount {text}")
    checks = [spin("Tilt it forward about 30 degrees, nose down.", mount, 1, 1.0),
              spin("Tilt it to ITS left about 30 degrees, left side down.", mount, 0, -1.0),
              spin("Turn it to ITS left about 45 degrees, staying upright.", mount, 2, 1.0)]
    if not all(checks):
        raise Abort("the gyro does not agree with the tilt readings. Repeat this step; if it still fails, send the output")
    say("Gravity and gyro agree.")
    return text


class Robot:
    """Servos and IMU, in the training env's joint order and the robot's axes."""

    def __init__(self, arr, cfg, imu, names, mount, speed, accel):
        self.arr, self.cfg, self.imu, self.mount, self.speed, self.accel = arr, cfg, imu, mount, int(speed), int(accel)
        self.joints = [cfg.joint(n) for n in names]
        self.guard = Guard(cfg)
        self.missed = [0] * 12
        self.q = None
        self.same_imu = 0
        self.last_imu = None
        self.volts = self.temp = float("nan")

    def read_joints(self):
        """Joint angles from the servos. A single missed reply keeps the last value; signed speed comes from the caller's differences, because Status.speed drops its sign."""
        status = self.arr.read_status()
        q = np.zeros(12, np.float64) if self.q is None else self.q.copy()
        volts, temps = [], []
        for i, j in enumerate(self.joints):
            s = status.get(j.name)
            if s is None:
                self.missed[i] += 1
                if self.q is None or self.missed[i] > MAX_MISSED_READS:
                    raise Abort(f"{j.name} stopped answering")
                continue
            self.missed[i] = 0
            q[i] = j.steps_to_rad(s.steps)
            volts.append(s.voltage_v)
            temps.append(s.temp_c)
            if self.guard.confirmed((j.name, "temp"), s.temp_c >= self.guard.max_temp):
                raise Abort(f"{j.name} is too hot: {s.temp_c} C (limit {self.guard.max_temp:.0f} C)")
            if self.guard.confirmed((j.name, "volt"), s.voltage_v < self.guard.min_volt):
                raise Abort(f"{j.name} voltage is too low: {s.voltage_v:.1f} V (limit {self.guard.min_volt:.1f} V)")
        self.volts, self.temp = min(volts) if volts else float("nan"), max(temps) if temps else float("nan")
        self.q = q
        return q

    def read_imu(self):
        """Gyro in rad/s and unit gravity, in the robot's axes. Stops if the driver keeps repeating one reading."""
        r = self.imu.read()
        raw = (tuple(r.quaternion), tuple(r.gyro))
        self.same_imu = self.same_imu + 1 if raw == self.last_imu else 0
        self.last_imu = raw
        if self.same_imu >= IMU_STALE_LOOPS and not getattr(self.imu, "constant_ok", False):
            raise Abort("the IMU has stopped updating")
        gravity = self.mount @ np.asarray(gravity_in_body(r.quaternion))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -gravity[2]))))
        if tilt > TILT_LIMIT_DEG:
            raise Abort(f"tilted {tilt:.0f} degrees (limit {TILT_LIMIT_DEG:.0f})")
        return (self.mount @ np.asarray(r.gyro)).astype(np.float32), gravity.astype(np.float32)

    def write(self, target):
        self.arr.write_positions({j.name: checked_steps(j, float(t), self.guard) for j, t in zip(self.joints, target)}, self.speed, self.accel)


class Clock:
    now = staticmethod(time.perf_counter)

    @staticmethod
    def sleep_until(t):
        time.sleep(max(0.0, t - time.perf_counter()))


def steady_timing(say):
    """Best effort for a steady 50 Hz: no garbage collection during the run, and real-time priority when the system allows it."""
    gc.collect()
    gc.disable()
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(10))
        say("Real-time priority on.")
    except (OSError, AttributeError):
        say("Real-time priority not available (it needs sudo). Check the late-loop count in summary.json.")


def enable_torque(arr, cfg, limit, say):
    """Hold every joint where it is, then enable torque, so nothing jumps toward an old command."""
    start = dict(arr.hold_current(*SLOW))
    arr.set_torque_limit(limit)
    arr.set_torque(True)
    time.sleep(0.2)
    for j in cfg.joints:
        moved = math.degrees(abs(signed_delta(read_one(arr, j), start[j.name])) / STEPS_PER_RAD)
        if moved > 8.0:
            raise Abort(f"{j.name} jumped {moved:.1f} deg when torque came on")
    say(f"Torque ON, limit {limit}.")


def go_home(robot, core, say, timeout=6.0, clock=Clock):
    """Slowly to the training home pose."""
    return go_to(robot, core.default, say, "the home pose", timeout, clock)


def go_to(robot, target, say, what, timeout=6.0, clock=Clock):
    """Slowly to `target` (training joint order, radians)."""
    robot.arr.write_positions({j.name: checked_steps(j, float(t), robot.guard) for j, t in zip(robot.joints, target)}, *SLOW)
    t_end = clock.now() + timeout
    worst = None
    while clock.now() < t_end:
        q = robot.read_joints()
        worst = float(np.abs(q - target).max())
        if math.degrees(worst) < HOME_TOLERANCE_DEG:
            say(f"At {what}.")
            return q
        clock.sleep_until(clock.now() + 0.05)
    raise Abort(f"could not reach {what} in {timeout:.0f} s: a joint is still {math.degrees(worst):.1f} deg away")


def file_sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def servo_setting_problems(rows, expect_pid=EXPECT_PID, need_cap_cleared=True):
    """From robot_test.read_settings: what differs from the servos the policy was trained for. Empty means all 12 match."""
    out = []
    for name, r in rows.items():
        pid = (r.get("PID_P"), r.get("PID_D"), r.get("PID_I"))
        if pid != tuple(expect_pid):
            out.append(f"{name} gains P {pid[0]} D {pid[1]} I {pid[2]}, expected P {expect_pid[0]} D {expect_pid[1]} I {expect_pid[2]}")
        if need_cap_cleared and r.get("MAX_ACC") != 0:
            out.append(f"{name} acceleration cap reads {r.get('MAX_ACC')}, expected 0 (add --clear-accel-cap)")
    return out


def pose_target(core, joint, degrees, names):
    """The home pose with one training joint moved by `degrees`."""
    t = core.default.astype(np.float64).copy()
    if joint is not None:
        t[names.index(joint)] += math.radians(degrees)
    return t


def pose_errors(target, q):
    """Per-joint error in degrees, measured minus commanded."""
    return np.degrees(np.asarray(q, np.float64) - np.asarray(target, np.float64))


def pose_check(robot, core, names, say, ask, hold_s=2.0, clock=Clock):
    """Robot hanging: home, then one joint at a time. Stops if a joint misses its target; asks the person whether the right part moved the right way."""
    results = []
    for label, joint, deg, question in POSES:
        target = pose_target(core, joint, deg, names)
        go_to(robot, target, say, label, clock=clock)
        clock.sleep_until(clock.now() + hold_s)
        err = pose_errors(target, robot.read_joints())
        worst = int(np.abs(err).argmax())
        say(f"{label}: worst joint {names[worst]} {err[worst]:+.1f} deg from its target")
        answer = ask(question + " (y/n) ").strip().lower()
        results.append({"pose": label, "joint": joint, "degrees": deg, "errors_deg": np.round(err, 2).tolist(), "seen_right": answer in ("y", "yes")})
        if abs(err[worst]) > POSE_TOLERANCE_DEG:
            raise Abort(f"{label}: {names[worst]} is {err[worst]:+.1f} deg from its target")
        if answer not in ("y", "yes"):
            raise Abort(f"{label}: the robot did not move as expected. Check the joint map, signs and calibration before any walking")
    go_home(robot, core, say, clock=clock)
    return results


def delay_test(robot, core, names, say, joint="r_knee", step_deg=6.0, repeats=6, clock=Clock):
    """Robot hanging, at the policy's own loop: all 12 servos and the IMU read, all 12 written, 50 Hz. Steps one joint and times command to first movement. The result is rounded up to the 20 ms loop, so it is an upper end."""
    dt = 1.0 / RATE_HZ
    i = names.index(joint)
    home = core.default.astype(np.float64).copy()
    moved = home.copy()
    moved[i] += math.radians(-abs(step_deg) if home[i] < 0 else abs(step_deg))
    target, delays, periods = home.copy(), [], []
    t_prev = clock.now()
    t_next = t_prev + dt

    def tick():
        nonlocal t_prev, t_next
        clock.sleep_until(t_next)
        t0 = clock.now()
        periods.append(t0 - t_prev)
        t_prev, t_next = t0, t_next + dt
        q = robot.read_joints().copy()
        robot.read_imu()
        return q

    for r in range(repeats):
        for _ in range(35):  # 0.7 s to settle
            q0 = tick()
            robot.write(target)
        target = moved.copy() if r % 2 == 0 else home.copy()
        t_cmd = clock.now()
        robot.write(target)
        for _ in range(25):
            q = tick()
            robot.write(target)
            if abs(math.degrees(q[i] - q0[i])) >= DELAY_MOVE_DEG:
                delays.append(1000.0 * (clock.now() - t_cmd))
                break
        else:
            raise Abort(f"{joint} did not move within 0.5 s of a {step_deg:.0f} deg step")
    p = np.array(periods[1:]) * 1000.0
    out = {"joint": joint, "step_deg": step_deg, "move_threshold_deg": DELAY_MOVE_DEG, "loop_ms_median": float(np.median(p)), "loop_ms_p99": float(np.percentile(p, 99)), "delay_ms": [round(x, 1) for x in delays], "delay_ms_median": float(np.median(delays))}
    say(f"Loop period: median {out['loop_ms_median']:.1f} ms, 99th percentile {out['loop_ms_p99']:.1f} ms. Command to first movement of {joint}: median {out['delay_ms_median']:.0f} ms, all {out['delay_ms']}. Measured in 20 ms loops, so it can read up to 20 ms high.")
    say(f"Training covers 0-{TRAINED_DELAY_MS:.0f} ms. " + (f"OVER THAT: no current policy has trained at this delay. Say so before any walking." if out["delay_ms_median"] > TRAINED_DELAY_MS + 1000.0 * dt else "Within or near that range, allowing for the 20 ms rounding."))
    return out


def load_replay(path, names):
    z = np.load(path, allow_pickle=False)
    target, q_sim = z["target"].astype(np.float64), z["q_sim"].astype(np.float64)
    if target.ndim != 2 or target.shape[1] != 12 or target.shape != q_sim.shape:
        raise Abort(f"{path}: target and q_sim must be N x 12 with the same shape")
    if abs(float(z["dt"]) - 1.0 / RATE_HZ) > 1e-6:
        raise Abort(f"{path}: made at {float(z['dt'])} s a step, the loop runs at {1.0 / RATE_HZ} s")
    if [str(x) for x in z["servo_names"]] != list(names):
        raise Abort(f"{path}: its joint order differs from this policy's")
    return target, q_sim


def replay_loop(robot, target, say, clock=Clock):
    """Sends a recorded list of joint targets at 50 Hz, open loop, under every safety check. Returns the measured angles, one row per loop."""
    dt = 1.0 / RATE_HZ
    late = stuck = 0
    t_prev = clock.now()
    t_next = t_prev + dt
    q_all = np.zeros_like(target)
    for k in range(len(target)):
        clock.sleep_until(t_next)
        t0 = clock.now()
        elapsed = t0 - t_prev
        if elapsed > ABSURD_LOOP_S:
            raise Abort(f"a control loop took {1000 * elapsed:.0f} ms")
        late = late + 1 if elapsed > LATE_LOOP_FACTOR * dt else 0
        if late >= MAX_LATE_LOOPS:
            raise Abort(f"{late} control loops in a row ran late")
        t_prev = t0
        q = robot.read_joints()
        robot.read_imu()
        q_all[k] = q
        was = target[max(0, k - 1)]
        stuck = stuck + 1 if float(np.abs(q - was).max()) > JOINT_ERROR_RAD else 0
        if stuck * dt >= JOINT_ERROR_S:
            i = int(np.abs(q - was).argmax())
            raise Abort(f"{robot.joints[i].name} is {math.degrees(abs(q[i] - was[i])):.0f} deg from its target and not catching up")
        robot.write(target[k])
        t_next += dt
    return q_all


def replay_report(target, q_sim, q_real, names, dt=1.0 / RATE_HZ, skip=50):
    """Joint by joint, after the first second: how far the measured angle is from the simulated one, how far it lags, and how much of the motion it makes."""
    rows = {}
    for i, n in enumerate(names):
        a, b = q_sim[skip:, i], q_real[skip:, i]
        x, y = a - a.mean(), b - b.mean()
        lags = range(-15, 16)
        score = [float(np.dot(x[max(0, -L):len(x) - max(0, L)], y[max(0, L):len(y) - max(0, -L)])) for L in lags]
        rows[n] = {"rms_deg": float(np.degrees(np.sqrt(np.mean((a - b) ** 2)))), "lag_ms": 1000.0 * dt * lags[int(np.argmax(score))], "motion_ratio": float(np.ptp(b) / max(np.ptp(a), 1e-6)),
                   "target_motion_deg": float(np.degrees(np.ptp(target[skip:, i])))}
    return rows


def control_loop(core, robot, command, u, seconds, scale, ramp_s, say, clock=Clock, log=None, truth=None, hold_heading=False):
    """Runs until `seconds` have passed. Any Abort leaves the caller to turn torque off. `truth` feeds tests with values the robot cannot measure."""
    command = np.array(command, np.float32)
    yaw0 = None
    dt = 1.0 / RATE_HZ
    steps = int(round(seconds * RATE_HZ))
    core.reset()
    q_prev = robot.read_joints().copy()
    t_prev = clock.now()
    t_next = t_prev + dt
    t_start = t_prev
    late = stuck = 0
    target = core.default
    for k in range(steps):
        clock.sleep_until(t_next)
        t0 = clock.now()
        elapsed = t0 - t_prev
        if elapsed > ABSURD_LOOP_S:
            raise Abort(f"a control loop took {1000 * elapsed:.0f} ms")
        late = late + 1 if elapsed > LATE_LOOP_FACTOR * dt else 0
        if late >= MAX_LATE_LOOPS:
            raise Abort(f"{late} control loops in a row ran late")
        q = robot.read_joints()
        qd = ((q - q_prev) / max(elapsed, 1e-3)).astype(np.float32)
        q_prev, t_prev = q.copy(), t0
        gyro, gravity = robot.read_imu()
        if hold_heading:
            q_imu = robot.last_imu[0]
            yaw0 = yaw_of(q_imu) if yaw0 is None else yaw0
            command[2] = heading_command(q_imu, yaw0)
        stuck = stuck + 1 if float(np.abs(q - target).max()) > JOINT_ERROR_RAD else 0
        if stuck * dt >= JOINT_ERROR_S:
            i = int(np.abs(q - target).argmax())
            raise Abort(f"{robot.joints[i].name} is {math.degrees(abs(q[i] - target[i])):.0f} deg from its target and not catching up")
        ramp = scale * min(1.0, k * dt / ramp_s) if ramp_s > 0 else scale
        extra = truth() if truth else {}
        qd = extra.pop("qd", qd)
        action, target = core.step(q.astype(np.float32), qd, gyro, gravity, command, u, ramp, **extra)
        robot.write(target)
        if log is not None:
            log.append([t0 - t_start, 1000 * elapsed, ramp, *action, *target, *q, *qd, *gyro, *gravity, robot.volts, robot.temp, 1000 * (clock.now() - t0), max(robot.missed)])
        t_next += dt
    say(f"Ran {steps} control steps.")


def save_results(out, log, meta, say):
    data = np.array(log, dtype=np.float64) if log else np.zeros((0, len(LOG_COLUMNS)))
    np.savetxt(out / "log.csv", data, delimiter=",", header=",".join(LOG_COLUMNS), comments="", fmt="%.5f")
    if len(data):
        dt, work = data[:, LOG_COLUMNS.index("dt_ms")], data[:, LOG_COLUMNS.index("work_ms")]
        meta["loops"] = len(data)
        meta["loop_ms"] = {"median": float(np.median(dt)), "p99": float(np.percentile(dt, 99)), "max": float(dt.max())}
        meta["work_ms"] = {"median": float(np.median(work)), "p99": float(np.percentile(work, 99)), "max": float(work.max())}
        meta["late_loops"] = int((dt > 1000 * LATE_LOOP_FACTOR / RATE_HZ).sum())
    (out / "summary.json").write_text(json.dumps(meta, indent=1, default=str))
    tar = out.with_suffix(".tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(out, arcname=out.name)
    say(f"Results: {out}\nSend this file: {tar}")


# --stage readback, added 8 Nov 2026. Read only, torque stays off. (label, address, bytes). Addresses are the ST3215 map in biped_hardware/st3215/registers.py, with 0x55 the undocumented cap.
READBACK = (("P", 0x15, 1), ("D", 0x16, 1), ("I", 0x17, 1), ("cap_0x55", 0x55, 1), ("accel_0x29", 0x29, 1), ("max_torque", 0x10, 2), ("torque_limit", 0x30, 2),
            ("min_pos", 0x09, 2), ("max_pos", 0x0B, 2), ("pos", 0x38, 2), ("deadband_cw", 0x1A, 1), ("deadband_ccw", 0x1B, 1), ("reply_delay", 0x07, 1),
            ("prot_current", 0x1C, 2), ("prot_torque", 0x22, 1), ("prot_time", 0x23, 1), ("overload_torque", 0x24, 1), ("volts_x10", 0x3E, 1), ("temp_c", 0x3F, 1))


def read_servo_settings(arr, joints, encode_read):
    """Every READBACK value for each joint. One that cannot be read is None, never 0, so it can't pass for a real reading."""
    rows = {}
    for j in joints:
        row = {"id": j.servo_id}
        for label, address, size in READBACK:
            try:
                data = arr._bus_for_id(j.servo_id).transfer(encode_read(j.servo_id, address, size), response_data_len=size)
                row[label] = int(data[0]) if size == 1 else int(data[0]) | int(data[1]) << 8
            except Exception:
                row[label] = None
        rows[j.name] = row
    return rows


def readback_problems(rows, expect_pid=EXPECT_PID, want_torque=1000):
    """What differs from what the policies trained for, one line each. Empty means all match."""
    out = []
    for name, r in rows.items():
        unread = [k for k, _, _ in READBACK if r.get(k) is None]
        if unread:
            out.append(f"{name} could not read: {', '.join(unread)}")
        if None not in (r.get("P"), r.get("D"), r.get("I")) and (r["P"], r["D"], r["I"]) != tuple(expect_pid):
            out.append(f"{name} gains P {r['P']} D {r['D']} I {r['I']}, expected P {expect_pid[0]} D {expect_pid[1]} I {expect_pid[2]}")
        if r.get("cap_0x55") not in (0, None):
            out.append(f"{name} acceleration cap reads {r['cap_0x55']}, expected 0 (add --clear-accel-cap)")
        for key in ("max_torque", "torque_limit"):
            if r.get(key) is not None and r[key] < want_torque:
                out.append(f"{name} {key} reads {r[key]}, below the {want_torque} the policies trained with")
        lo, hi, pos = r.get("min_pos"), r.get("max_pos"), r.get("pos")
        if None not in (lo, hi, pos) and hi > lo and not lo <= pos <= hi:
            out.append(f"{name} sits at {pos}, outside its stored limits {lo} to {hi}")
    return out


def readback_stage(arr, joints, say, encode_read, expect_pid=EXPECT_PID):
    """Prints one line a servo and returns the rows and every difference from training. Torque stays off."""
    rows = read_servo_settings(arr, joints, encode_read)
    for name, r in rows.items():
        say(f"{name:10s} " + " ".join(f"{k} {'?' if r[k] is None else r[k]}" for k in ("id", "P", "D", "I", "cap_0x55", "accel_0x29", "max_torque", "torque_limit", "min_pos", "pos", "max_pos")))
    say("Overload settings. These may cut torque under load; the simulator has none of this, so send me these numbers:")
    for name, r in rows.items():
        say(f"{name:10s} " + " ".join(f"{k} {'?' if r[k] is None else r[k]}" for k in ("prot_current", "prot_torque", "prot_time", "overload_torque", "deadband_cw", "deadband_ccw", "reply_delay", "volts_x10", "temp_c")))
    problems = readback_problems(rows, expect_pid)
    if problems:
        say("DIFFERENT FROM TRAINING:")
        for line in problems:
            say("  " + line)
    else:
        say("All 12 servos match training on gains, acceleration cap and torque limits, and each sits inside its stored limits.")
    return {"rows": rows, "problems": problems}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy", help="policy.npz from export_policy.py")
    p.add_argument("--replay-file", default=None, help="--stage replay: targets and simulated angles from export_replay.py")
    p.add_argument("--stage", choices=["readback", "imu", "poses", "delay", "replay"] + list(STAGES), default="hang",
                   help="readback prints every servo's settings with no torque; imu finds --imu-mount with no torque; poses checks joint directions with the robot hanging; the others run the policy")
    p.add_argument("--speed", type=float, default=0.2, help=f"forward command in m/s, at most {MAX_SPEED}")
    p.add_argument("--budget", default="1,1,1", help="u for flight, slip and torque: 0 is the tightest and 1 the loosest")
    p.add_argument("--seconds", type=float, default=None, help="run time, at most the stage's")
    p.add_argument("--action-scale", type=float, default=None, help="scales the policy's actions, at most 1")
    p.add_argument("--ramp-s", type=float, default=2.0, help="actions grow from zero over this time")
    p.add_argument("--servo-speed", type=int, default=FAST[0], help="servo speed setting. The default is the servo's fastest, which is what training assumes")
    p.add_argument("--servo-accel", type=int, default=FAST[1], help="servo acceleration setting, 0 to 254, 0 = no ramp")
    p.add_argument("--clear-accel-cap", action="store_true", help="clear the servos' acceleration cap before torque comes on (RAM only, back after a power cycle)")
    p.add_argument("--imu-mount", default="x,y,z", help="which sensor axes lie along the robot's forward, left and up axes")
    p.add_argument("--hold-heading", action="store_true", help="steer back to the starting heading using the IMU. Assumes the IMU's z axis is close to vertical")
    p.add_argument("--biped-ws", default=None, help="folder of your biped_ws code")
    p.add_argument("--out", default=None, help="results folder (default: a new folder in your home)")
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--hardware-config", default=None)
    p.add_argument("--servos-config", default=None)
    p.add_argument("--mock", action="store_true", help="pretend servos, for testing this script")
    p.add_argument("--yes", action="store_true", help="answer yes to every question. Only allowed with --mock")
    p.add_argument("--torque-limit", type=int, default=None, help="servo torque limit, 100 to 1000, in place of the stage's (hardware.yaml still caps it)")
    p.add_argument("--min-volts", type=float, default=MIN_VOLTS, help=f"stop below this servo supply voltage (default {MIN_VOLTS})")
    p.add_argument("--servo-p", type=int, default=None, help="set the servos' P gain for this power-up only (RAM; stored gains unchanged), e.g. 24 for a softer servo; the check then expects it")
    p.add_argument("--servo-d", type=int, default=None, help="as --servo-p, for the D gain")
    p.add_argument("--expect-pid", default=",".join(map(str, EXPECT_PID)), help="stored servo gains P,D,I the policy was trained with; any servo that differs stops the run")
    p.add_argument("--allow-accel-cap", action="store_true", help="run even though the servos' acceleration cap is on (only for policies trained on the capped servo)")
    args = p.parse_args(argv)
    if args.yes and not args.mock:
        sys.exit("--yes only works with --mock. On the real robot, a person must answer.")
    stage = STAGES.get(args.stage, STAGES["supported"] if args.stage == "replay" else STAGES["hang"])
    if args.stage == "replay" and not args.replay_file:
        sys.exit("--stage replay needs --replay-file")
    if args.torque_limit is not None and not 100 <= args.torque_limit <= 1000:
        sys.exit("--torque-limit must be from 100 to 1000")
    try:
        expect_pid = tuple(int(x) for x in args.expect_pid.split(","))
        assert len(expect_pid) == 3
    except (ValueError, AssertionError):
        sys.exit("--expect-pid needs three whole numbers, P,D,I")
    seconds = min(args.seconds, stage["seconds"]) if args.seconds else stage["seconds"]
    scale = args.action_scale if args.action_scale is not None else stage["scale"]
    if not 0 < scale <= 1:
        sys.exit("--action-scale must be above 0 and at most 1")
    if not 0 <= args.speed <= MAX_SPEED:
        sys.exit(f"--speed must be from 0 to {MAX_SPEED}")
    u = np.array([float(x) for x in args.budget.split(",")], np.float32)
    if u.shape != (3,) or u.min() < 0 or u.max() > 1:
        sys.exit("--budget needs three numbers from 0 to 1")
    command = np.array([args.speed, 0.0, 0.0], np.float32)
    try:
        mount = imu_mount(args.imu_mount)
    except ValueError as exc:
        sys.exit(str(exc))
    core = PolicyCore(args.policy)
    if core.reads_truth and not args.mock:
        sys.exit("This policy reads true speed or budget balances, which the robot cannot measure. Train with --actor_hide achieved,levels.")

    find_hardware_code(args.biped_ws)
    try:
        from biped_hardware.tools.common import setup
    except ImportError as exc:
        sys.exit(f"Missing a Python package ({exc}). Install with: pip install pyserial pyyaml")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_on_signal)
    say = safe_print
    ask = (lambda prompt: "GO") if args.yes else input
    try:
        cfg, bus, arr = setup(args)
    except Exception as exc:  # usually unplugged boards or servo power off; the bus code's own message says which
        sys.exit(f"Could not reach the servos: {exc}")
    code = 0
    log = []
    out = Path(args.out).expanduser() if args.out else Path.home() / f"policy_run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    meta = {"started": datetime.now().isoformat(), "stage": args.stage, "mock": args.mock, "speed": args.speed, "budget": u.tolist(), "seconds": seconds, "action_scale": scale,
            "servo_speed": args.servo_speed, "servo_accel": args.servo_accel, "imu_mount": args.imu_mount, "policy": core.meta, "columns": LOG_COLUMNS, "stopped": None,
            "policy_file": str(Path(args.policy).resolve()), "policy_file_sha256": file_sha256(args.policy), "torque_limit_asked": args.torque_limit,
            "min_volts": args.min_volts, "tilt_limit_deg": TILT_LIMIT_DEG}
    imu = None
    try:
        if not args.mock and not cfg.calibration_loaded:
            raise Abort("no calibration file. Run biped_calibrate first")
        imu = open_imu(cfg, args.mock, say)
        if imu is None:
            raise Abort("the policy needs the IMU, and it did not start")
        if args.stage == "imu":
            arr.set_torque(False)
            meta["imu_mount_found"] = imu_stage(imu, ask, say)
            return code
        if args.mock:
            imu.constant_ok = True
        wanted = args.torque_limit if args.torque_limit is not None else stage["limit"]
        limit = min(wanted, int(cfg.safety.get("torque_limit", wanted)))
        if limit < wanted:
            say(f"hardware.yaml caps the torque limit at {limit}, below the {wanted} asked for. Using {limit}. The policy was trained with the full limit (1000).")
        meta["torque_limit"] = limit
        robot = Robot(arr, cfg, imu, core.servo_names, mount, args.servo_speed, args.servo_accel)
        if not args.mock:
            robot.guard.min_volt = max(robot.guard.min_volt, args.min_volts)
        missing = [n for n, ok in arr.ping_all().items() if not ok]
        if missing:
            raise Abort("no answer from: " + ", ".join(missing))
        if args.clear_accel_cap:
            meta["accel_cap"] = clear_accel_cap(arr, cfg, say, mock=args.mock)
        if args.servo_p is not None or args.servo_d is not None:
            meta["servo_gains"] = set_servo_gains(arr, cfg, say, p=args.servo_p, d=args.servo_d, mock=args.mock)
            expect_pid = (args.servo_p if args.servo_p is not None else expect_pid[0], args.servo_d if args.servo_d is not None else expect_pid[1], expect_pid[2])
        if args.stage == "readback":
            from biped_hardware.st3215.protocol import encode_read
            meta["readback"] = readback_stage(arr, cfg.joints, say, encode_read, expect_pid)
            return code
        if not args.mock:
            from types import SimpleNamespace
            rows = read_settings(SimpleNamespace(cfg=cfg, arr=arr))
            meta["servo_settings"] = rows
            problems = servo_setting_problems(rows, expect_pid, need_cap_cleared=not args.allow_accel_cap)
            if problems:
                raise Abort("servo settings differ from training: " + "; ".join(problems))
            say("Servo settings match training: gains P{} D{} I{}, acceleration cap {} on all 12.".format(*expect_pid, "cleared" if not args.allow_accel_cap else "not checked"))
        say(f"Policy file {Path(args.policy).name}, sha256 {meta['policy_file_sha256'][:16]}...")
        robot.read_joints()
        gyro, gravity = robot.read_imu()
        say(f"Policy: {core.meta.get('checkpoint')} at {core.meta.get('steps')} steps. Stage {args.stage}: torque limit {limit}, actions x{scale}, {seconds:.0f} s, {args.speed} m/s, budget {args.budget}.")
        say(f"Gravity reads {np.round(gravity, 2).tolist()}, and should be close to [0, 0, -1] when the robot is upright. If it is not, fix --imu-mount.")
        if args.stage in ("hang", "poses", "delay"):
            confirmed = ask("The robot is hanging with both feet off the ground, and your hand is on the power switch? (y/n) ").strip().lower() in ("y", "yes", "go")
        else:
            confirmed = ask(f"Stage {args.stage}. robot_test.py has passed, the hang stage looked right, the harness is on, and your hand is on the power switch. Type GO to continue: ").strip() == "GO"
        if not confirmed:
            raise Abort("not confirmed")
        enable_torque(arr, cfg, limit, say)
        go_home(robot, core, say)
        if args.stage == "poses":
            names = list(core.meta.get("joint_train_names") or TRAIN_NAMES)
            meta["poses"] = pose_check(robot, core, names, say, ask)
            say("Pose check passed: every joint reached its target and moved the way the simulator does.")
            return code
        if args.stage == "delay":
            steady_timing(say)
            meta["delay"] = delay_test(robot, core, list(core.meta.get("joint_train_names") or TRAIN_NAMES), say)
            go_home(robot, core, say)
            return code
        if args.stage == "replay":
            names = list(core.meta.get("joint_train_names") or TRAIN_NAMES)
            target, q_sim = load_replay(args.replay_file, core.servo_names)
            meta["replay_file"], meta["replay_file_sha256"] = str(Path(args.replay_file).resolve()), file_sha256(args.replay_file)
            go_to(robot, target[0], say, "the first pose of the replay")
            if ask(f"Feet on the floor, harness holding the robot. Replay {len(target) / RATE_HZ:.0f} s of recorded joint targets, open loop? (y/n) ").strip().lower() not in ("y", "yes", "go"):
                raise Abort("not started")
            steady_timing(say)
            q_real = replay_loop(robot, target, say)
            brake(arr, robot.joints, robot.speed, robot.accel)
            go_home(robot, core, say)
            np.savez(out / "replay_measured.npz", target=target, q_sim=q_sim, q_real=q_real, names=np.array(names))
            meta["replay"] = replay_report(target, q_sim, q_real, names)
            for n, r in meta["replay"].items():
                say(f"{n:10s} off the simulator by {r['rms_deg']:5.1f} deg rms, lag {r['lag_ms']:4.0f} ms, makes {100 * r['motion_ratio']:3.0f}% of the simulated motion ({r['target_motion_deg']:.0f} deg asked)")
            return code
        if ask("Start the policy? (y/n) ").strip().lower() not in ("y", "yes", "go"):
            raise Abort("not started")
        steady_timing(say)
        control_loop(core, robot, command, u, seconds, scale, args.ramp_s, say, log=log, hold_heading=args.hold_heading)
        brake(arr, robot.joints, robot.speed, robot.accel)  # stop at the policy's setting before the gentle move home
        go_home(robot, core, say)
        if not args.yes:
            ask("Done, holding the home pose. Support the robot, then press Enter to turn torque off. ")
    except Abort as exc:
        meta["stopped"] = str(exc)
        say(f"\nSTOPPED: {exc}")
        code = 1
    except KeyboardInterrupt:
        meta["stopped"] = "interrupted"
        say("\nInterrupted.")
        code = 1
    except Exception as exc:  # report it plainly; the raw traceback still goes to the screen below
        meta["stopped"] = repr(exc)
        say(f"\nSTOPPED by an unexpected error: {exc!r}")
        code = 1
        import traceback
        traceback.print_exc()
    finally:
        torque_off(arr, say)
        gc.enable()
        meta["clamped_writes"] = getattr(arr, "clamped_writes", None)
        save_results(out, log, meta, say)
        if imu is not None:
            imu.close()
        bus.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
