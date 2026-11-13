#!/usr/bin/env python3
"""Measure the real robot so the simulator can use real numbers instead of guesses. Runs on the Pi. No ROS needed.

Before you start:
  1. Stop servo_bridge if it is running. Only one program can use the servos.
  2. HANG the robot so both feet are off the ground and the legs can swing freely.
  3. Keep fingers clear of the joints, and keep a hand near the power switch.

Run everything (about 15 minutes):
    python3 robot_test.py all --biped-ws ~/biped_ws

Or one test at a time: check, timing, noise, directions, step, track.
    check       reads servos and settings. No torque.
    timing      how long one control loop takes. No torque.
    noise       sensor noise while holding still, 60 s. Torque on, low limit.
    directions  moves each joint 5 deg and asks you if it went the right way. Low limit.
    step        small quick moves on all 12 joints, to measure servo delay. Limit 500.
    track       small sine waves on the right leg, to measure lag while moving. Limit 500.

Ctrl-C, an SSH disconnect or any problem turns torque OFF, so the legs will drop. That is why the robot must hang.
Results go in a new folder, packed into one .tar.gz file. Nothing is written to the servos' permanent memory.

Needs: pyserial, pyyaml, and your biped_hardware code. Testing without a robot: add --mock (results are meaningless).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import signal
import statistics
import sys
import tarfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

STEPS_PER_RAD = 4096 / (2 * math.pi)
SLOW = (300, 10)            # speed steps/s and accel, the same gentle values as biped_jog
LOW_LIMIT = 300             # torque limit 0..1000, biped_jog's value
MOTION_LIMIT = 500          # never above this here, even if hardware.yaml says more
FAST = (3400, 254)          # highest speed and accel values the servo takes, so the servo's own response is measured
RATE_HZ = 50                # the control rate used in training
REG_ACC = 0x29              # goal acceleration, unit 100 steps/s^2, 0 = no ramp. biped_hardware writes it with every position command
REG_MAX_ACC = 0x55          # undocumented acceleration cap, factory 50, held in RAM so every power-up restores it (the V2 designer's notes, 2026)
FREE = (3400, 0)            # fastest speed with no acceleration ramp. Only measured once the cap is cleared
MAX_STEP_DEG = 10.0
RUNAWAY_MARGIN_DEG = 10.0   # stop if a joint goes this far beyond where it was told to go

# What a small POSITIVE joint angle does. joint_map.py checks this against the simulator's kinematics.
MOTION = {"hip_yaw": "toes_outward", "hip_pitch": "leg_backward", "hip_roll": "leg_outward",
          "knee": "shin_forward", "ankle_pitch": "toes_down", "ankle_roll": "outer_edge_up"}
OPPOSITE = {"toes_outward": "toes_inward", "leg_backward": "leg_forward", "leg_outward": "leg_inward",
            "shin_forward": "shin_backward", "toes_down": "toes_up", "outer_edge_up": "outer_edge_down"}
OPPOSITE.update({v: k for k, v in list(OPPOSITE.items())})
WORDS = {
    "toes_outward": "the toes turn OUTWARD, away from the other foot",
    "toes_inward": "the toes turn INWARD, toward the other foot",
    "leg_backward": "the whole leg swings BACKWARD (front is where the toes point)",
    "leg_forward": "the whole leg swings FORWARD (front is where the toes point)",
    "leg_outward": "the whole leg swings OUTWARD, away from the other leg",
    "leg_inward": "the whole leg swings INWARD, toward the other leg",
    "shin_forward": "the shin swings FORWARD, straightening the knee",
    "shin_backward": "the shin swings BACKWARD, bending the knee",
    "toes_down": "the toes point DOWN",
    "toes_up": "the toes point UP",
    "outer_edge_up": "the OUTER edge of the foot lifts up",
    "outer_edge_down": "the OUTER edge of the foot drops down",
}
# The designer's safe range for each joint, in radians. servos.yaml allows more, including bending the knee the wrong
# way, so tests stay inside both. These are the training model's joint ranges; joint_map.py checks that.
DEPLOY_LIMITS = {"hip_yaw": (-0.785398, 0.785398), "hip_roll": (-0.436332, 0.785398), "hip_pitch": (-1.570796, 0.523599),
                 "knee": (-2.094395, 0.087266), "ankle_pitch": (-0.610865, 0.872665), "ankle_roll": (-1.047198, 1.047198)}
# What the simulator assumes now (config/v2_budget_survive.yaml). joint_map.py checks these too.
SIM = {"delay_range_ms": [0, 50], "servo_lag_tau_ms": 20, "volts": 12.0,
       "noise": {"joint_pos": 0.002, "joint_vel": 0.25, "gyro": 0.02, "gravity": 0.02},
       "bias": {"joint_pos": 0.0175, "gyro": 0.02, "gravity": 0.05},
       # The track test run in the simulator by sim_track.py, at delay 0 and 50 ms. Includes the 20 ms control step.
       "track_lag_ms": {"1.0": [127, 177], "3.0": [89, 140]}, "track_gain": {"1.0": [0.73, 0.74], "3.0": [0.60, 0.62]}}
ND = "NOT DETERMINED"


class Abort(RuntimeError):
    """Stop now: torque goes off on the way out."""


# ---------------------------------------------------------------------------------------------------------------------
# Number crunching. No hardware here, so tests can check it against known answers.
# ---------------------------------------------------------------------------------------------------------------------

def signed_delta(a, b):
    """a - b in encoder steps, safe across the 4095/0 seam."""
    return ((a - b + 2048) % 4096) - 2048


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(math.ceil(q / 100 * len(s))) - 1)]


def spread(values):
    """Median, 95%, 99% and worst of a list, or None for an empty list."""
    if not values:
        return None
    return {"median": statistics.median(values), "p95": percentile(values, 95), "p99": percentile(values, 99), "max": max(values)}


def median_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def analyze_step(t, y, y0, target, min_move=3.0):
    """One step. t = seconds after the command, y = encoder steps, y0 = resting value before, target = commanded steps.

    dead_ms is the first sample that has clearly left y0. Anything never reached stays None, which means not determined.
    """
    span = target - y0
    out = {"dead_ms": None, "t63_ms": None, "t90_ms": None, "overshoot_pct": None, "final_error_steps": None}
    if not t or abs(span) < 2 * min_move:
        return out
    frac = [(v - y0) / span for v in y]
    threshold = max(min_move, 0.05 * abs(span)) / abs(span)
    for key, level in (("dead_ms", threshold), ("t63_ms", 0.63), ("t90_ms", 0.9)):
        out[key] = next((1000 * a for a, f in zip(t, frac) if f >= level), None)
    out["overshoot_pct"] = 100 * max(0.0, max(frac) - 1.0)
    tail = y[-max(1, len(y) // 5):]
    out["final_error_steps"] = sum(tail) / len(tail) - target
    return out


def fit_sine(t, y, freq):
    """Least squares y = A sin(wt + phase) + c. Returns A, phase, c. Handles uneven sample times."""
    w = 2 * math.pi * freq
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for ti, yi in zip(t, y):
        row = (math.sin(w * ti), math.cos(w * ti), 1.0)
        for i in range(3):
            atb[i] += row[i] * yi
            for j in range(3):
                ata[i][j] += row[i] * row[j]
    for i in range(3):                  # Gaussian elimination with pivoting
        p = max(range(i, 3), key=lambda r: abs(ata[r][i]))
        ata[i], ata[p] = ata[p], ata[i]
        atb[i], atb[p] = atb[p], atb[i]
        for r in range(i + 1, 3):
            f = ata[r][i] / ata[i][i]
            atb[r] -= f * atb[i]
            for c in range(i, 3):
                ata[r][c] -= f * ata[i][c]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (atb[i] - sum(ata[i][c] * x[c] for c in range(i + 1, 3))) / ata[i][i]
    return math.hypot(x[0], x[1]), math.atan2(x[1], x[0]), x[2]


def analyze_track(t_cmd, cmd, t_pos, pos, freq):
    """Gain and time lag of the joint behind its command."""
    a_c, ph_c, _ = fit_sine(t_cmd, cmd, freq)
    a_p, ph_p, _ = fit_sine(t_pos, pos, freq)
    lag = (ph_c - ph_p + math.pi) % (2 * math.pi) - math.pi
    return {"gain": a_p / a_c if a_c > 0 else None, "lag_ms": 1000 * lag / (2 * math.pi * freq)}


def gravity_in_body(q):
    """Unit gravity in the robot frame from quaternion w,x,y,z, the same quantity the policy sees in simulation."""
    w, x, y, z = q
    return (-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y)))


def std3(rows):
    return [statistics.pstdev(c) for c in zip(*rows)]


def mean3(rows):
    return [statistics.fmean(c) for c in zip(*rows)]


def kind_of(name):
    """r_hip_roll_joint -> hip_roll"""
    return name.split("_", 1)[1].replace("_joint", "")


def joint_limits(joint):
    lo, hi = DEPLOY_LIMITS[kind_of(joint.name)]
    return max(lo, joint.lower), min(hi, joint.upper)


def expected_words(name, sign):
    motion = MOTION[kind_of(name)]
    return WORDS[motion if sign > 0 else OPPOSITE[motion]]


def centre_for(lower, upper, margin, now, amp):
    """Where to centre a +-amp move so it stays inside the soft limits, or None if it cannot fit."""
    lo, hi = lower + margin + amp, upper - margin - amp
    return None if lo > hi else min(hi, max(lo, now))


# ---------------------------------------------------------------------------------------------------------------------
# Safety and small helpers
# ---------------------------------------------------------------------------------------------------------------------

class Guard:
    """Stops on heat, low voltage or a silent servo. The servo bus sometimes returns one corrupted temperature or voltage, so heat and
    voltage must be bad for CONFIRM_LOOPS reads in a row before it stops. Real heat and real voltage sag last far longer than that."""
    CONFIRM_LOOPS = 5

    def __init__(self, cfg):
        self.max_temp = float(cfg.safety.get("max_temperature_c", 65))
        self.min_volt = float(cfg.safety.get("min_voltage_v", 6.0))
        self.margin = float(cfg.safety.get("limit_margin_rad", 0.02))
        self.streak = {}
        self.glitches = 0

    def confirmed(self, key, bad):
        """True once `key` has been bad CONFIRM_LOOPS reads in a row. A single bad read is counted in self.glitches and ignored."""
        if not bad:
            self.streak[key] = 0
            return False
        self.glitches += 1
        self.streak[key] = self.streak.get(key, 0) + 1
        return self.streak[key] >= self.CONFIRM_LOOPS

    def status(self, status):
        for name, st in status.items():
            if st is None:
                raise Abort(f"{name} stopped answering")
            if self.confirmed((name, "temp"), st.temp_c >= self.max_temp):
                raise Abort(f"{name} is too hot: {st.temp_c} C (limit {self.max_temp:.0f} C). Let it cool down")
            if self.confirmed((name, "volt"), st.voltage_v < self.min_volt):
                raise Abort(f"{name} voltage is too low: {st.voltage_v:.1f} V (limit {self.min_volt:.1f} V)")


def torque_off(arr, say):
    last = None
    for _ in range(3):
        try:
            arr.set_torque(False)
            say("Torque OFF.")
            return True
        except Exception as exc:  # keep trying: this is the one write that must get through
            last = exc
            time.sleep(0.05)
    say(f"!!! COULD NOT TURN TORQUE OFF ({last}). CUT THE POWER NOW.")
    return False


@contextmanager
def torque_on(ctx, limit):
    """Hold every joint where it is, then enable torque. Torque always goes off on the way out."""
    arr = ctx.arr
    limit = min(int(limit), int(ctx.cfg.safety.get("torque_limit", MOTION_LIMIT)), MOTION_LIMIT)
    start = dict(arr.hold_current(*SLOW))
    try:
        arr.set_torque_limit(limit)
        arr.set_torque(True)
        time.sleep(0.2)
        for j in ctx.cfg.joints:            # catches a joint that jumps as torque comes on
            moved = abs(signed_delta(read_one(arr, j), start[j.name])) / STEPS_PER_RAD
            if math.degrees(moved) > 8.0:
                raise Abort(f"{j.name} jumped {math.degrees(moved):.1f} deg when torque came on")
        yield start
    finally:
        torque_off(arr, ctx.say)


def read_one(arr, joint, tries=5):
    last = None
    for _ in range(tries):
        try:
            return arr.read_steps(joint)
        except Exception as exc:  # a single missed reply is normal on a serial bus; five in a row is not
            last = exc
    raise Abort(f"{joint.name} stopped answering: {last}")


def checked_steps(joint, rad, guard):
    """Radians to servo steps, refusing anything outside the soft limits or the encoder range."""
    lo, hi = joint_limits(joint)
    if not lo + guard.margin - 1e-9 <= rad <= hi - guard.margin + 1e-9:
        raise Abort(f"{joint.name}: target {rad:.3f} rad is outside the limits {lo:.3f}..{hi:.3f}")
    steps = joint.rad_to_steps(rad)
    if not 0 <= steps <= 4095:
        raise Abort(f"{joint.name}: target crosses the encoder seam. Check calibration")
    return steps


def move_slow(ctx, joint, steps, timeout=3.0):
    """Gentle move. Warns if the joint stops a little short and stops the test if it stops far short."""
    ctx.arr.write_positions({joint.name: steps}, *SLOW)
    t0 = time.perf_counter()
    here = read_one(ctx.arr, joint)
    while time.perf_counter() - t0 < timeout and abs(signed_delta(here, round(steps))) > 6:
        time.sleep(0.01)
        here = read_one(ctx.arr, joint)
    off_deg = math.degrees(abs(signed_delta(here, round(steps))) / STEPS_PER_RAD)
    if off_deg > 3.0:
        raise Abort(f"{joint.name} did not reach its target: {off_deg:.1f} deg short. Something may be blocking it")
    return here


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def need_calibration(ctx):
    if not ctx.cfg.calibration_loaded and not ctx.mock:
        raise Abort(f"No calibration file at {ctx.cfg.calibration_path}. Run biped_calibrate first, then run this again")


class Ctx:
    def __init__(self, cfg, bus, arr, imu, out_dir, ask, say, mock=False, quick=False, step_deg=5.0):
        self.cfg, self.bus, self.arr, self.imu, self.out, self.ask, self.say = cfg, bus, arr, imu, Path(out_dir), ask, say
        self.mock, self.quick, self.step_deg, self.guard, self.results = mock, quick, step_deg, Guard(cfg), {}
        self.cap_cleared = False
        self.modes = None

    def save(self):
        (self.out / "summary.json").write_text(json.dumps(self.results, indent=1))


# ---------------------------------------------------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------------------------------------------------

def read_settings(ctx):
    """Read-only look at each servo's stored settings. The sim's fit assumes specific gains, so these matter."""
    from biped_hardware.st3215.protocol import encode_read
    from biped_hardware.st3215.registers import Reg
    wanted = ("PID_P", "PID_D", "PID_I", "CW_DEADBAND", "CCW_DEADBAND", "RETURN_DELAY", "LOCK_FLAG", "ACC", "MAX_ACC")
    raw = {"ACC": REG_ACC, "MAX_ACC": REG_MAX_ACC}  # by address: Reg may not name the undocumented one
    rows = {}
    for j in ctx.cfg.joints:
        row = {}
        for key in wanted:
            try:
                data = ctx.arr._bus_for_id(j.servo_id).transfer(encode_read(j.servo_id, raw[key] if key in raw else getattr(Reg, key), 1), response_data_len=1)
                row[key] = int(data[0])
            except Exception:  # an unsupported read must show up as unknown, never as 0
                row[key] = ND
        rows[j.name] = row
    return rows


def write_packet(servo_id, address, data):
    """Feetech WRITE instruction (0x03): FF FF id length 03 address data... checksum, with length = len(data) + 3."""
    body = [servo_id, len(data) + 3, 0x03, address, *data]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def read_byte(arr, servo_id, address):
    from biped_hardware.st3215.protocol import encode_read
    return int(arr._bus_for_id(servo_id).transfer(encode_read(servo_id, address, 1), response_data_len=1)[0])


def clear_accel_cap(arr, cfg, say, mock=False):
    """Clear every servo's acceleration cap, torque off: 0x55 = 0, then 0x29 = 0 (the cap gates 0x29, so 0x29 goes after it).

    Both are RAM, so switching the servos off and on restores the factory cap. Nothing is written to permanent memory: a servo whose permanent
    memory is unlocked (LOCK_FLAG 0) stops the run before anything is written. Each value is read back, and any servo that does not read 0 stops the
    run, so a result is never labelled as uncapped when it is not.
    """
    if mock:
        say("MOCK: the acceleration cap is not touched.")
        return {"mock": True}
    try:
        from biped_hardware.st3215.protocol import encode_write
        packet = lambda sid, address: encode_write(sid, address, bytes([0]))
    except ImportError:
        packet = lambda sid, address: write_packet(sid, address, [0])
    from biped_hardware.st3215.registers import Reg
    arr.set_torque(False)
    out = {}
    for j in cfg.joints:
        sid = j.servo_id
        try:
            lock = read_byte(arr, sid, Reg.LOCK_FLAG)
        except Exception as exc:
            raise Abort(f"could not read {j.name}'s memory lock: {exc!r}")
        if lock == 0:
            raise Abort(f"{j.name}'s permanent memory is unlocked, so a write could stick. Nothing was written. Power-cycle the servos and try again")
        try:
            before = (read_byte(arr, sid, REG_MAX_ACC), read_byte(arr, sid, REG_ACC))
            for address in (REG_MAX_ACC, REG_ACC):
                arr._bus_for_id(sid).transfer(packet(sid, address), response_data_len=0)
                time.sleep(0.002)
            after = (read_byte(arr, sid, REG_MAX_ACC), read_byte(arr, sid, REG_ACC))
        except Exception as exc:
            raise Abort(f"could not clear the acceleration cap on {j.name}: {exc!r}")
        out[j.name] = {"lock": lock, "max_acc_before": before[0], "acc_before": before[1], "max_acc_after": after[0], "acc_after": after[1]}
        if after[0] != 0:
            raise Abort(f"{j.name} still reads acceleration cap {after[0]} after clearing")
    caps = sorted({r["max_acc_before"] for r in out.values()})
    say(f"Acceleration cap was {caps} on the 12 servos; now 0 on all. Switching the servos off and on brings the factory cap back.")
    return out


REG_PID = {"P": 0x15, "D": 0x16}


def set_servo_gains(arr, cfg, say, p=None, d=None, mock=False, encode_read=None, encode_write=None):
    """Set every servo's position-loop P and/or D gain for this power-up only, torque off (12 Nov 2026, the owner: the robot shivers).

    The gains sit in the servo's EEPROM area, and the servo keeps writes there only when its memory lock is open (LOCK_FLAG 0). As in
    clear_accel_cap, a servo whose lock is open stops the run before anything is written, so nothing can stick: switching the servos off and on
    brings back the stored gains (P 32, D 32 on this robot). Each value is read back; a servo that does not read the new value stops the run.
    Check after a power cycle with --stage readback that the stored gains are unchanged.
    """
    want = {k: v for k, v in (("P", p), ("D", d)) if v is not None}
    if not want:
        return {}
    for k, v in want.items():
        if not (0 <= int(v) <= 254):
            raise Abort(f"servo {k} gain {v} is outside 0-254")
    if mock:
        say(f"MOCK: servo gains not touched (asked {want}).")
        return {"mock": True, "asked": want}
    if encode_read is None or encode_write is None:
        from biped_hardware.st3215.protocol import encode_read as er, encode_write as ew
        encode_read = encode_read or er
        encode_write = encode_write or (lambda sid, address, value: ew(sid, address, bytes([value])))
    from biped_hardware.st3215.registers import Reg
    rd = lambda sid, address: int(arr._bus_for_id(sid).transfer(encode_read(sid, address, 1), response_data_len=1)[0])
    arr.set_torque(False)
    out = {}
    for j in cfg.joints:
        sid = j.servo_id
        try:
            lock = rd(sid, Reg.LOCK_FLAG)
        except Exception as exc:
            raise Abort(f"could not read {j.name}'s memory lock: {exc!r}")
        if lock == 0:
            raise Abort(f"{j.name}'s permanent memory is unlocked, so a gain write could stick. Nothing was written. Power-cycle the servos and try again")
        row = {"lock": lock}
        for k, v in want.items():
            try:
                row[f"{k}_before"] = rd(sid, REG_PID[k])
                arr._bus_for_id(sid).transfer(encode_write(sid, REG_PID[k], int(v)), response_data_len=0)
                time.sleep(0.002)
                row[f"{k}_after"] = rd(sid, REG_PID[k])
            except Exception as exc:
                raise Abort(f"could not set {k} on {j.name}: {exc!r}")
            if row[f"{k}_after"] != int(v):
                raise Abort(f"{j.name} reads {k} {row[k + '_after']} after writing {v}; this servo may not take gain changes while locked. Nothing else was changed")
        out[j.name] = row
    say("Servo gains for this power-up: " + ", ".join(f"{k} {v}" for k, v in want.items()) + " on all 12 (stored gains unchanged; switching off and on restores them).")
    return out

def test_check(ctx):
    arr, cfg, say = ctx.arr, ctx.cfg, ctx.say
    say("\n--- CHECK: reading servos, no torque ---")
    missing = [n for n, ok in arr.ping_all().items() if not ok]
    if missing:
        ctx.results["check"] = {"missing": missing}
        ctx.save()
        raise Abort("These servos do not answer: " + ", ".join(missing))
    status = arr.read_status()
    ctx.guard.status(status)
    say(f"{'joint':22}{'ID':>4}{'angle deg':>11}{'volts':>7}{'temp C':>8}  note")
    rows, off = [], []
    for j in cfg.joints:
        st = status[j.name]
        rad = j.steps_to_rad(st.steps)
        lo, hi = joint_limits(j)
        note = "" if lo <= rad <= hi else "OUTSIDE LIMITS"
        if note:
            off.append(j.name)
        say(f"{j.name:22}{j.servo_id:>4}{math.degrees(rad):>11.1f}{st.voltage_v:>7.1f}{st.temp_c:>8}  {note}")
        rows.append([j.name, j.servo_id, j.direction, j.zero_steps, st.steps, rad, st.voltage_v, st.temp_c])
    write_csv(ctx.out / "check.csv", ["joint", "id", "direction", "zero_steps", "steps", "rad", "volts", "temp_c"], rows)
    volts = [status[j.name].voltage_v for j in cfg.joints]
    settings = read_settings(ctx)
    ctx.results["check"] = {"volts_min": min(volts), "volts_max": max(volts), "temps_max_c": max(status[j.name].temp_c for j in cfg.joints),
                            "outside_limits": off, "calibration_loaded": cfg.calibration_loaded,
                            "servo_settings": settings, "imu_started": ctx.imu is not None}
    ctx.save()
    gains = {(s["PID_P"], s["PID_D"], s["PID_I"]) for s in settings.values()}
    say(f"Stored servo gains (P, D, I): {sorted(gains, key=str)}")
    say(f"Acceleration cap (register 0x55): {sorted({s['MAX_ACC'] for s in settings.values()}, key=str)}. Factory value is 50; 0 means no cap.")
    if min(volts) < SIM["volts"] - 1.0:
        say(f"Warning: lowest voltage is {min(volts):.1f} V. The simulator assumes {SIM['volts']:.0f} V, so the robot will be weaker than simulated.")
    if not cfg.calibration_loaded:
        say(f"Warning: no calibration file at {cfg.calibration_path}. The tests that move joints will refuse to run.")
    if off:
        say("Warning: joints outside their limits: " + ", ".join(off) + ". Check calibration before moving.")
    say("IMU: started." if ctx.imu is not None else f"IMU: did not start, so IMU numbers are {ND}.")


def test_timing(ctx):
    arr, cfg, say = ctx.arr, ctx.cfg, ctx.say
    cycles = 100 if ctx.quick else 30 * RATE_HZ
    say(f"\n--- TIMING: {cycles} control loops at {RATE_HZ} Hz, no torque, nothing moves ---")
    arr.set_torque(False)
    speed, accel = int(cfg.safety.get("servo_speed_cap_steps_s", 1500)), int(cfg.safety.get("servo_accel", 30))
    dt = 1.0 / RATE_HZ
    rows, failed = [], 0
    t_next = time.perf_counter()
    for i in range(cycles):
        t0 = time.perf_counter()
        status = arr.read_status()
        t1 = time.perf_counter()
        missing = [n for n, s in status.items() if s is None]
        failed += len(missing)
        if len(missing) == len(status):
            raise Abort("All servos stopped answering")
        # Same bus traffic as a real loop. The target equals the present position, so nothing moves even with torque on.
        arr.write_positions({n: s.steps for n, s in status.items() if s is not None}, speed, accel)
        t2 = time.perf_counter()
        if ctx.imu is not None:
            ctx.imu.read()
        t3 = time.perf_counter()
        rows.append([i, t0, 1000 * (t1 - t0), 1000 * (t2 - t1), 1000 * (t3 - t2), 1000 * (t3 - t0)])
        t_next += dt
        time.sleep(max(0.0, t_next - time.perf_counter()))
    write_csv(ctx.out / "timing.csv", ["i", "t_s", "read_ms", "write_ms", "imu_ms", "total_ms"], rows)
    totals = [r[5] for r in rows]
    gaps = [1000 * (b[1] - a[1]) for a, b in zip(rows, rows[1:])]
    res = {"read_ms": spread([r[2] for r in rows]), "write_ms": spread([r[3] for r in rows]),
           "imu_ms": spread([r[4] for r in rows]) if ctx.imu is not None else ND, "total_ms": spread(totals),
           "loops_over_20ms": sum(t > 1000 * dt for t in totals), "loops": cycles, "failed_servo_reads": failed,
           "loop_start_gap_ms": spread(gaps), "clamped_writes": getattr(arr, "clamped_writes", ND)}
    ctx.results["timing"] = res
    ctx.save()
    s = res["total_ms"]
    say(f"One loop: typical {s['median']:.1f} ms, worst {s['max']:.1f} ms. Time allowed: {1000 * dt:.0f} ms, and the policy needs some too.")
    say(f"Loops over the limit: {res['loops_over_20ms']} of {cycles}. Failed servo reads: {failed}.")


def test_noise(ctx):
    arr, cfg, say, imu = ctx.arr, ctx.cfg, ctx.say, ctx.imu
    seconds = 2 if ctx.quick else 60
    need_calibration(ctx)
    say(f"\n--- NOISE: holding still for {seconds} s. Wait until the robot stops swinging, then do not touch it ---")
    if ctx.ask("Is the robot hanging still and is everyone clear? [y/n] ").strip().lower() not in YES:
        raise Abort("Stopped at your request")
    dt = 1.0 / RATE_HZ
    servo_rows, imu_rows, frozen, prev = [], [], 0, None
    with torque_on(ctx, LOW_LIMIT):
        time.sleep(0.5 if ctx.quick else 3.0)    # let the hold settle
        t0 = t_next = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            status = arr.read_status()
            now = time.perf_counter() - t0
            ctx.guard.status(status)
            servo_rows.append([now] + [status[j.name].steps for j in cfg.joints] + [status[j.name].speed for j in cfg.joints]
                              + [status[j.name].load for j in cfg.joints] + [status[j.name].voltage_v for j in cfg.joints])
            if imu is not None:
                r = imu.read()
                sample = tuple(r.quaternion) + tuple(r.gyro)
                frozen += sample == prev
                prev = sample
                imu_rows.append([now] + list(r.gyro) + list(r.accel) + list(r.quaternion))
            t_next += dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
    names = [j.name for j in cfg.joints]
    write_csv(ctx.out / "noise_servos.csv", ["t_s"] + [f"steps_{n}" for n in names] + [f"speed_{n}" for n in names]
              + [f"load_{n}" for n in names] + [f"volts_{n}" for n in names], servo_rows)
    res = {"seconds": seconds, "joints": {}}
    for k, n in enumerate(names):
        col = [r[1 + k] for r in servo_rows]
        ref = col[0]
        rad = [signed_delta(v, ref) / STEPS_PER_RAD for v in col]
        vel = [(b - a) / (rt[0] - lt[0]) for a, b, lt, rt in zip(rad, rad[1:], servo_rows, servo_rows[1:])]
        res["joints"][n] = {"pos_std_rad": statistics.pstdev(rad), "pos_range_rad": max(rad) - min(rad),
                            "diff_vel_std_rad_s": statistics.pstdev(vel) if vel else ND}
    pos = [v["pos_std_rad"] for v in res["joints"].values()]
    res["pos_std_rad_median"] = statistics.median(pos)
    res["diff_vel_std_rad_s_median"] = median_or_none([v["diff_vel_std_rad_s"] for v in res["joints"].values() if v["diff_vel_std_rad_s"] != ND])
    res["imu"] = ND
    if imu is not None and imu_rows:
        write_csv(ctx.out / "noise_imu.csv", ["t_s", "gx", "gy", "gz", "ax", "ay", "az", "qw", "qx", "qy", "qz"], imu_rows)
        frac = frozen / len(imu_rows)
        if frac < 0.5:
            gyro = [r[1:4] for r in imu_rows]
            grav = [gravity_in_body(r[7:11]) for r in imu_rows]
            res["imu"] = {"gyro_std": std3(gyro), "gyro_mean": mean3(gyro), "gravity_std": std3(grav), "gravity_mean": mean3(grav), "frozen_fraction": frac}
        else:
            res["imu_note"] = f"IMU repeated the same value {100 * frac:.0f}% of the time, so its numbers are {ND}"
            say(res["imu_note"])
    ctx.results["noise"] = res
    ctx.save()
    say(f"Joint position noise: {res['pos_std_rad_median']:.5f} rad. The simulator uses {SIM['noise']['joint_pos']}.")


def test_directions(ctx):
    arr, cfg, say = ctx.arr, ctx.cfg, ctx.say
    need_calibration(ctx)
    amp = math.radians(5.0)
    say("\n--- DIRECTIONS: each joint moves 5 deg, twice. Watch it, then answer. ---")
    say("'Right leg' means the robot's own right, as you stand behind it looking the way the toes point.")
    if ctx.ask("Is the robot HANGING with both feet off the ground and fingers clear? Torque turns on. [y/n] ").strip().lower() not in YES:
        raise Abort("Stopped at your request")
    rows, wrong, unknown = [], [], []
    joints = cfg.joints[:1] if ctx.quick else cfg.joints
    with torque_on(ctx, LOW_LIMIT):
        for j in joints:
            ctx.guard.status(arr.read_status())
            now_steps = read_one(arr, j)
            now = j.steps_to_rad(now_steps)
            lo, hi = joint_limits(j)
            sign = 1 if now + amp <= hi - ctx.guard.margin else -1
            if not lo + ctx.guard.margin <= now + sign * amp <= hi - ctx.guard.margin:
                unknown.append(j.name)
                say(f"{j.name}: no room to move, skipped. Its direction is {ND}.")
                rows.append([j.name, j.servo_id, sign, ND, ND, ND])
                continue
            target = checked_steps(j, now + sign * amp, ctx.guard)
            leg = "RIGHT" if j.name.startswith("r_") else "LEFT"
            words = expected_words(j.name, sign)
            while True:
                say(f"\n{leg} leg, {kind_of(j.name)} (servo ID {j.servo_id}). Expected: {words}.")
                time.sleep(0 if ctx.quick else 1.0)
                moved = 0
                for _ in range(2):
                    here = move_slow(ctx, j, target)
                    moved = signed_delta(here, now_steps)
                    time.sleep(0 if ctx.quick else 0.5)
                    move_slow(ctx, j, now_steps)
                    time.sleep(0 if ctx.quick else 0.5)
                answer = ctx.ask("Did it move like that? [y = yes, n = no, r = show again] ").strip().lower()
                if answer != "r":
                    break
            # Moves were commanded through the same sign the encoder is read with, so only a person can confirm the true direction.
            followed = moved * j.direction * sign > 0 and abs(moved) >= 0.5 * amp * STEPS_PER_RAD
            ok = answer in YES and followed
            if not ok:
                wrong.append(j.name)
            rows.append([j.name, j.servo_id, sign, moved, answer, ok])
    write_csv(ctx.out / "directions.csv", ["joint", "id", "sign", "moved_steps", "answer", "ok"], rows)
    ctx.results["directions"] = {"wrong": wrong, "not_determined": unknown, "checked": len(rows)}
    ctx.save()
    if wrong:
        raise Abort("These joints are wrong: " + ", ".join(wrong) + ". Fix direction or wiring in servos.yaml or calibration first")
    say(f"\nDirections: {len(rows) - len(unknown)} joints correct, {len(unknown)} not determined.")


def sample_step(ctx, j, target, seconds, centre_steps, limit_steps):
    """Send one command, then read this joint as fast as possible."""
    arr = ctx.arr
    t, y = [], []
    t0 = time.perf_counter()
    arr.write_positions({j.name: target[0]}, target[1], target[2])
    while time.perf_counter() - t0 < seconds:
        a = time.perf_counter()
        raw = read_one(arr, j)
        b = time.perf_counter()
        v = centre_steps + signed_delta(raw, round(centre_steps))
        if abs(v - centre_steps) > limit_steps:
            raise Abort(f"{j.name} moved too far ({abs(v - centre_steps) / STEPS_PER_RAD:.2f} rad from centre). Stopping")
        t.append(0.5 * (a + b) - t0)
        y.append(v)
        time.sleep(0.0005)      # about 1 kHz is plenty, and it keeps the bus and the result file reasonable
    return t, y


def brake(arr, joints, speed, accel, seconds=0.15):
    """Hold each joint where it is now, at the setting it was moving with, so it stops quickly. On 6 Nov, with the acceleration cap cleared, a gentle
    move (accel 10) sent to r_hip_yaw while it still moved at about 1 rad/s after the fast setting ran 26 deg past its target before stopping."""
    arr.write_positions({j.name: read_one(arr, j) for j in joints}, speed, accel)
    time.sleep(seconds)


def speed_modes(ctx):
    s = ctx.cfg.safety
    modes = {"bridge": (int(s.get("servo_speed_cap_steps_s", 1500)), int(s.get("servo_accel", 30))), "fast": FAST}
    if getattr(ctx, "cap_cleared", False):
        modes["free"] = FREE
    wanted = getattr(ctx, "modes", None)
    if wanted:
        missing = [m for m in wanted if m not in modes]
        if missing:
            raise Abort(f"unknown or unavailable settings: {', '.join(missing)} ('free' needs --clear-accel-cap)")
        modes = {m: modes[m] for m in wanted}  # in the order asked
    return modes


def enter_centre(ctx, j, amp):
    """Move to a centre with room on both sides. Returns centre radians and where the joint started, or None if it will not fit."""
    start = read_one(ctx.arr, j)
    now = j.steps_to_rad(start)
    centre = centre_for(*joint_limits(j), ctx.guard.margin, now, amp)
    if centre is None or abs(centre - now) > math.radians(15.0):
        ctx.say(f"{j.name}: too close to its limit, or calibration is off. Skipped, result is {ND}.")
        return None
    move_slow(ctx, j, checked_steps(j, centre, ctx.guard))
    return centre, start


def test_step(ctx):
    arr, cfg, say = ctx.arr, ctx.cfg, ctx.say
    need_calibration(ctx)
    amp = math.radians(ctx.step_deg)
    dwell = 0.6
    say(f"\n--- STEP: quick {ctx.step_deg:.0f} deg moves on every joint, with your bridge's servo settings and with the fastest settings ---")
    if ctx.ask("Is the robot HANGING with both feet off the ground and fingers clear? [y/n] ").strip().lower() not in YES:
        raise Abort("Stopped at your request")
    modes = speed_modes(ctx)
    joints = cfg.joints[:1] if ctx.quick else cfg.joints
    rows, per = [], {}
    with torque_on(ctx, MOTION_LIMIT):
        for j in joints:
            ctx.guard.status(arr.read_status())
            entered = enter_centre(ctx, j, amp)
            per[j.name] = {m: ND for m in modes}
            if entered is None:
                continue
            centre, start = entered
            c_steps = j.rad_to_steps(centre)
            limit = (amp + math.radians(RUNAWAY_MARGIN_DEG)) * STEPS_PER_RAD
            for mode, (speed, accel) in modes.items():
                time.sleep(0.2)
                rest = c_steps + signed_delta(read_one(arr, j), round(c_steps))
                stats = []
                for seg, off in enumerate((amp, 0.0, -amp, 0.0) * (1 if ctx.quick else 2)):
                    target = checked_steps(j, centre + off, ctx.guard)
                    t, y = sample_step(ctx, j, (target, speed, accel), dwell, c_steps, limit)
                    rows.extend([j.name, mode, seg, round(target, 1), f"{a:.5f}", b] for a, b in zip(t, y))
                    stats.append(analyze_step(t, y, rest, target))
                    tail = y[-max(1, len(y) // 5):]
                    rest = sum(tail) / len(tail)
                    ctx.guard.status(arr.read_status())
                per[j.name][mode] = {k: median_or_none([s[k] for s in stats]) for k in stats[0]}
                per[j.name][mode]["segments"] = len(stats)
                per[j.name][mode]["reached"] = sum(s["t90_ms"] is not None for s in stats)
                per[j.name][mode]["read_hz"] = len(t) / dwell
            arr.write_positions({j.name: c_steps}, *SLOW)
            time.sleep(0.3)
            move_slow(ctx, j, start)
            d = per[j.name].get("fast")
            say(f"{j.name:22} fastest settings: starts moving in {_ms(d, 'dead_ms')}, 90% there in {_ms(d, 't90_ms')}")
    write_csv(ctx.out / "step.csv", ["joint", "mode", "segment", "target_steps", "t_s", "pos_steps"], rows)
    ctx.results["step"] = {"step_deg": ctx.step_deg, "modes": {k: list(v) for k, v in modes.items()}, "joints": per,
                           "summary": {m: {k: median_or_none([p[m][k] for p in per.values() if isinstance(p[m], dict)])
                                           for k in ("dead_ms", "t63_ms", "t90_ms", "overshoot_pct")} for m in modes}}
    ctx.save()


def _ms(d, key):
    v = d.get(key) if isinstance(d, dict) else None
    return ND if v is None else f"{v:.0f} ms"


def test_track(ctx):
    arr, cfg, say = ctx.arr, ctx.cfg, ctx.say
    need_calibration(ctx)
    amp = math.radians(ctx.step_deg)
    freqs = (1.0,) if ctx.quick else (1.0, 2.0, 3.0)
    dt = 1.0 / RATE_HZ
    say(f"\n--- TRACK: {ctx.step_deg:.0f} deg sine waves on the right leg at {RATE_HZ} Hz commands, like a walking policy ---")
    if ctx.ask("Is the robot HANGING with both feet off the ground and fingers clear? [y/n] ").strip().lower() not in YES:
        raise Abort("Stopped at your request")
    modes = speed_modes(ctx)
    right = [j for j in cfg.joints if j.name.startswith("r_")]
    joints = right[:1] if ctx.quick else right
    rows, per = [], {}
    try:
        _track_joints(ctx, joints, modes, freqs, amp, dt, rows, per)
    finally:  # a run that stops still leaves its samples, so the cause can be seen
        write_csv(ctx.out / "track.csv", ["joint", "mode", "freq_hz", "t_cmd_s", "cmd_steps", "t_pos_s", "pos_steps", "load"], rows)
        ctx.results["track_partial"] = per
        ctx.save()
    ctx.results["track"] = {"amp_deg": ctx.step_deg, "rate_hz": RATE_HZ, "joints": per,
                            "summary": {m: {str(f): {k: median_or_none([p[m][str(f)][k] for p in per.values() if m in p and str(f) in p[m]]) for k in ("gain", "lag_ms")}
                                            for f in freqs} for m in modes}}
    ctx.save()


def _track_joints(ctx, joints, modes, freqs, amp, dt, rows, per):
    arr, say = ctx.arr, ctx.say
    with torque_on(ctx, MOTION_LIMIT):
        for j in joints:
            entered = enter_centre(ctx, j, amp)
            per[j.name] = {}
            if entered is None:
                continue
            centre, start = entered
            c_steps = j.rad_to_steps(centre)
            limit = (amp + math.radians(RUNAWAY_MARGIN_DEG)) * STEPS_PER_RAD
            for mode, (speed, accel) in modes.items():
                per[j.name][mode] = {}
                for f in freqs:
                    settle = 1.0 / f                 # the first cycle ramps up gently and is not analysed
                    cycles = max(4, int(math.ceil(3.0 * f)))
                    total = settle + cycles / f
                    t_cmd, cmd, t_pos, pos = [], [], [], []
                    t0 = t_next = time.perf_counter()
                    while True:
                        a = time.perf_counter()
                        if a - t0 >= total:
                            break
                        status = arr.read_status()
                        b = time.perf_counter()
                        ctx.guard.status(status)
                        p = c_steps + signed_delta(status[j.name].steps, round(c_steps))
                        if abs(p - c_steps) > limit:
                            raise Abort(f"{j.name} moved too far ({(p - c_steps) / STEPS_PER_RAD * 57.2958:+.1f} deg from centre, last command "
                                        f"{((cmd[-1] if cmd else c_steps) - c_steps) / STEPS_PER_RAD * 57.2958:+.1f} deg) at {mode} {f:.0f} Hz, {now if t_cmd else 0.0:.2f} s in. Stopping")
                        now = time.perf_counter() - t0
                        ramp = min(1.0, now / settle)
                        rad = centre + ramp * amp * math.sin(2 * math.pi * f * now)
                        target = checked_steps(j, rad, ctx.guard)
                        arr.write_positions({j.name: target}, speed, accel)
                        t_pos.append(0.5 * (a + b) - t0)
                        pos.append(p)
                        t_cmd.append(now)
                        cmd.append(target)
                        rows.append([j.name, mode, f, f"{now:.5f}", round(target, 1), f"{0.5 * (a + b) - t0:.5f}", p, status[j.name].load])
                        t_next += dt
                        time.sleep(max(0.0, t_next - time.perf_counter()))
                    keep_c = [i for i, x in enumerate(t_cmd) if x >= settle]
                    keep_p = [i for i, x in enumerate(t_pos) if x >= settle]
                    per[j.name][mode][str(f)] = analyze_track([t_cmd[i] for i in keep_c], [cmd[i] for i in keep_c],
                                                              [t_pos[i] for i in keep_p], [pos[i] for i in keep_p], f)
                    brake(arr, [j], speed, accel)  # stop first at this setting: a gentle move sent while still moving brakes slowly and overshoots
                    arr.write_positions({j.name: c_steps}, *SLOW)
                    time.sleep(0.4)
            move_slow(ctx, j, start)
            for mode, r in per[j.name].items():
                say(f"{j.name:22} {mode:6} " + ", ".join(f"gain {v['gain']:.2f}, lag {v['lag_ms']:.0f} ms at {k} Hz" for k, v in r.items()))


TESTS = {"check": test_check, "timing": test_timing, "noise": test_noise, "directions": test_directions, "step": test_step, "track": test_track}


def comparison(results):
    """Simulator assumption next to the measurement. Anything not measured stays NOT DETERMINED."""
    def fmt(v, form="{:.1f}"):
        return ND if v is None or v == ND else form.format(v)

    def span(pair):
        return f"{pair[0]}-{pair[1]}"

    timing = results.get("timing", {})
    noise = results.get("noise", {})
    imu = noise.get("imu") if isinstance(noise.get("imu"), dict) else {}
    step = results.get("step", {}).get("summary", {}).get("fast", {})
    track = results.get("track", {}).get("summary", {}).get("fast", {})
    total = timing.get("total_ms") or {}
    rows = [
        ("Control loop, typical ms", "must fit in 20", fmt(total.get("median"))),
        ("Control loop, worst ms", "must fit in 20", fmt(total.get("max"))),
        ("Servo starts moving, ms", f"delay {SIM['delay_range_ms'][0]}-{SIM['delay_range_ms'][1]}", fmt(step.get("dead_ms"), "{:.0f}")),
        ("Lag behind command, 1 Hz, ms", span(SIM["track_lag_ms"]["1.0"]), fmt((track.get("1.0") or {}).get("lag_ms"), "{:.0f}")),
        ("Lag behind command, 3 Hz, ms", span(SIM["track_lag_ms"]["3.0"]), fmt((track.get("3.0") or {}).get("lag_ms"), "{:.0f}")),
        ("Motion size / command, 1 Hz", span(SIM["track_gain"]["1.0"]), fmt((track.get("1.0") or {}).get("gain"), "{:.2f}")),
        ("Motion size / command, 3 Hz", span(SIM["track_gain"]["3.0"]), fmt((track.get("3.0") or {}).get("gain"), "{:.2f}")),
        ("Joint position noise, rad", f"{SIM['noise']['joint_pos']}", fmt(noise.get("pos_std_rad_median"), "{:.5f}")),
        ("Joint speed noise, rad/s", f"{SIM['noise']['joint_vel']}", fmt(noise.get("diff_vel_std_rad_s_median"), "{:.3f}")),
        ("Gyro noise, rad/s", f"{SIM['noise']['gyro']}", fmt(max(imu["gyro_std"]) if imu else None, "{:.4f}")),
        ("Gyro bias, rad/s", f"{SIM['bias']['gyro']}", fmt(max(abs(v) for v in imu["gyro_mean"]) if imu else None, "{:.4f}")),
        ("Gravity noise", f"{SIM['noise']['gravity']}", fmt(max(imu["gravity_std"]) if imu else None, "{:.4f}")),
        ("Voltage, V", f"{SIM['volts']:.0f}", fmt(results.get("check", {}).get("volts_min"))),
    ]
    return rows


def finish(ctx):
    rows = comparison(ctx.results)
    ctx.say(f"\n{'What':32}{'Simulator':>18}{'Your robot':>18}")
    for what, sim, real in rows:
        ctx.say(f"{what:32}{sim:>18}{real:>18}")
    ctx.results["comparison"] = [{"what": a, "simulator": b, "robot": c} for a, b, c in rows]
    ctx.save()
    archive = ctx.out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(ctx.out, arcname=ctx.out.name)
    ctx.say(f"\nDone. Results: {ctx.out}\nSend me this one file: {archive}")


def open_imu(cfg, mock, say):
    if not cfg.imu.get("enabled", True):
        return None
    from biped_hardware.bno085 import BNO085, MockBNO085
    imu = MockBNO085() if mock else BNO085(cfg.imu.get("i2c_bus", 1), cfg.imu.get("address", 0x4A), cfg.imu.get("mode", "imuplus"))
    try:
        imu.initialize()
        return imu
    except Exception as exc:  # the servo tests are still worth running without the IMU
        say(f"IMU could not start: {exc}")
        return None


YES = ("y", "yes")  # 4 Nov 2026: "yes" was read as no, which marked all six right-leg joints wrong although they moved the right way


def find_hardware_code(path):
    """Use biped_hardware if it is already installed, otherwise look in the biped_ws folder."""
    try:
        import biped_hardware  # noqa: F401
        return
    except ImportError:
        pass
    places = [Path(path).expanduser()] if path else [Path.home() / "biped_ws", Path.home() / "ros2_ws/src/biped_ws", Path.cwd() / "biped_ws"]
    for ws in places:
        if (ws / "biped_hardware/biped_hardware/config.py").exists():
            sys.path.insert(0, str(ws / "biped_hardware"))
            return
    sys.exit("Could not find your biped_hardware code. Say where it is with: --biped-ws /path/to/biped_ws")


def safe_print(text=""):
    try:
        print(text, flush=True)
    except OSError:  # the terminal went away (SSH dropped). Shutting down torque must carry on regardless.
        pass


def stop_on_signal(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", choices=["all"] + list(TESTS))
    p.add_argument("--biped-ws", default=None, help="folder of your biped_ws code")
    p.add_argument("--out", default=None, help="results folder (default: a new folder in your home)")
    p.add_argument("--step-deg", type=float, default=5.0, help=f"size of test moves, at most {MAX_STEP_DEG:.0f}")
    p.add_argument("--no-imu", action="store_true", help="skip the IMU")
    p.add_argument("--modes", default=None, help="servo settings for the step and track tests, in order, e.g. free,fast (default: bridge, fast, and free once cleared)")
    p.add_argument("--clear-accel-cap", action="store_true", help="clear the servos' acceleration cap (RAM only, back after a power cycle) and add a 'free' setting, no ramp, to the step and track tests")
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=None)
    p.add_argument("--calibration", default=None)
    p.add_argument("--hardware-config", default=None)
    p.add_argument("--servos-config", default=None)
    p.add_argument("--mock", action="store_true", help="pretend servos, for testing this script")
    p.add_argument("--yes", action="store_true", help="answer yes to every question. Only allowed with --mock")
    p.add_argument("--quick", action="store_true", help="shortened run. Only allowed with --mock")
    args = p.parse_args(argv)
    if (args.yes or args.quick) and not args.mock:
        sys.exit("--yes and --quick only work with --mock. On the real robot, a person must answer.")
    if not 0 < args.step_deg <= MAX_STEP_DEG:
        sys.exit(f"--step-deg must be above 0 and at most {MAX_STEP_DEG:.0f}")

    find_hardware_code(args.biped_ws)
    try:
        from biped_hardware.tools.common import setup
    except ImportError as exc:
        sys.exit(f"Missing a Python package ({exc}). Install with: pip install pyserial pyyaml")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_on_signal)

    say = safe_print
    ask = (lambda prompt: "y") if args.yes else input
    try:
        cfg, bus, arr = setup(args)
    except Exception as exc:  # usually unplugged boards or servo power off; the bus code's own message says which
        sys.exit(f"Could not reach the servos: {exc}")
    out = Path(args.out).expanduser() if args.out else Path.home() / f"robot_test_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    imu = None if args.no_imu else open_imu(cfg, args.mock, say)
    ctx = Ctx(cfg, bus, arr, imu, out, ask, say, mock=args.mock, quick=args.quick, step_deg=args.step_deg)
    ctx.modes = [m.strip() for m in args.modes.split(",")] if args.modes else None
    meta = {"started": datetime.now().isoformat(), "host": platform.node(), "python": platform.python_version(), "mock": args.mock,
            "test": args.test, "script": sys.argv[0], "joints": [vars(j) if hasattr(j, "__dict__") else str(j) for j in cfg.joints],
            "bus": cfg.bus, "safety": cfg.safety, "pid_config": cfg.pid, "imu_config": cfg.imu, "calibration_loaded": cfg.calibration_loaded}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if cfg.calibration_loaded:
        shutil.copy(cfg.calibration_path, out / "calibration.yaml")

    say(f"Results folder: {out}" + ("\nMOCK SERVOS: these numbers mean nothing about a real robot." if args.mock else ""))
    code = 0
    try:
        if args.clear_accel_cap:
            ctx.results["accel_cap"] = clear_accel_cap(arr, cfg, say, mock=args.mock)
            ctx.cap_cleared = True  # with --mock nothing is cleared, but the 'free' mode still runs so the script can be tried
            ctx.save()
        for name in (list(TESTS) if args.test == "all" else [args.test]):
            ctx.results.setdefault("completed", [])
            TESTS[name](ctx)
            ctx.results["completed"].append(name)
            ctx.save()
        if args.test == "all" and not args.yes:
            grams = ask("\nLast thing: put the whole robot on a kitchen scale. Weight in grams? (Enter to skip) ").strip()
            ctx.results["mass_g"] = float(grams) if grams.replace(".", "", 1).isdigit() else ND
    except Abort as exc:
        ctx.results["stopped"] = str(exc)
        say(f"\nSTOPPED: {exc}")
        code = 1
    except KeyboardInterrupt:
        ctx.results["stopped"] = "interrupted"
        say("\nInterrupted.")
        code = 1
    except Exception as exc:  # report it plainly; the raw traceback still goes to the screen below
        ctx.results["stopped"] = repr(exc)
        say(f"\nSTOPPED by an unexpected error: {exc!r}")
        code = 1
        import traceback
        traceback.print_exc()
    finally:
        torque_off(arr, say)
        finish(ctx)
        if imu is not None:
            imu.close()
        bus.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
