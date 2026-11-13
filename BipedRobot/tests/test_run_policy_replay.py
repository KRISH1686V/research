"""run_policy.py stages added 8 Nov 2026: --stage delay and --stage replay, and export_replay.py's file format. No hardware: a fake robot, and on the mock path pretend servos. CPU only."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2/hardware_tests"
sys.path.insert(0, str(HW))
import run_policy  # noqa: E402
from robot_test import Abort  # noqa: E402

NAMES = list(run_policy.TRAIN_NAMES)


class Core:
    default = np.array([0.0, -0.17, 0.0, -0.45, -0.28, 0.0] * 2)


class FakeRobot:
    """Moves `steps_late` control loops after it is told."""

    def __init__(self, steps_late=1, stuck=False):
        self.joints = [type("J", (), {"name": n})() for n in NAMES]
        self.q = Core.default.copy()
        self.pending = []
        self.late, self.stuck = steps_late, stuck
        self.imu_reads = 0

    def read_joints(self):
        if len(self.pending) >= self.late:
            self.q = self.pending[-self.late]
        return self.q.copy()

    def read_imu(self):
        self.imu_reads += 1
        return np.zeros(3, np.float32), np.array([0, 0, -1], np.float32)

    def write(self, target):
        if not self.stuck:
            self.pending.append(np.array(target, float))


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep_until(self, t):
        self.t = max(self.t, t)


@pytest.mark.parametrize("late,want", [(1, 20.0), (2, 40.0), (3, 60.0)])
def test_delay_test_reads_the_robots_delay_in_loops(late, want):
    said, robot, clock = [], FakeRobot(steps_late=late), Clock()
    out = run_policy.delay_test(robot, Core, NAMES, said.append, clock=clock)
    assert out["delay_ms_median"] == pytest.approx(want, abs=1e-6), out
    assert out["loop_ms_median"] == pytest.approx(20.0, abs=1e-6)
    assert len(out["delay_ms"]) == 6 and out["joint"] == "r_knee"
    assert robot.imu_reads > 100, "it reads the IMU like the real loop does"
    assert any("Training covers" in s for s in said)


def test_delay_test_says_so_when_over_the_trained_range():
    said = []
    run_policy.delay_test(FakeRobot(steps_late=4), Core, NAMES, said.append, clock=Clock())  # 80 ms reads as past 40 + the 20 ms rounding
    assert any("OVER THAT" in s for s in said)
    said = []
    run_policy.delay_test(FakeRobot(steps_late=2), Core, NAMES, said.append, clock=Clock())
    assert not any("OVER THAT" in s for s in said)


def test_delay_test_stops_on_a_joint_that_does_not_move():
    with pytest.raises(Abort):
        run_policy.delay_test(FakeRobot(stuck=True), Core, NAMES, lambda s: None, clock=Clock())


def sine_targets(n=500):
    t = np.arange(n) * 0.02
    return np.tile(Core.default, (n, 1)) + 0.2 * np.sin(2 * np.pi * t)[:, None]


def test_replay_sends_every_target_in_order_and_measures():
    robot, target = FakeRobot(steps_late=1), sine_targets()
    q = run_policy.replay_loop(robot, target, lambda s: None, clock=Clock())
    assert q.shape == target.shape and len(robot.pending) == len(target)
    assert np.allclose(robot.pending[10], target[10])
    assert robot.imu_reads == len(target)


def test_replay_stops_on_a_stuck_joint():
    with pytest.raises(Abort):
        run_policy.replay_loop(FakeRobot(stuck=True), Core.default + 2.0 * np.ones((200, 12)), lambda s: None, clock=Clock())


def test_replay_report_finds_lag_and_lost_motion():
    target = sine_targets()
    q_sim = target.copy()
    q_real = np.roll(target, 3, axis=0)  # 60 ms late
    q_real[:, 3] = Core.default[3] + 0.5 * (q_real[:, 3] - Core.default[3])  # the knee makes half the motion
    r = run_policy.replay_report(target, q_sim, q_real, NAMES)
    assert r["r_hip_z"]["lag_ms"] == pytest.approx(60.0) and r["r_hip_z"]["motion_ratio"] == pytest.approx(1.0, abs=0.01)
    assert r[NAMES[3]]["motion_ratio"] == pytest.approx(0.5, abs=0.01) and r[NAMES[3]]["rms_deg"] > 1.0
    assert run_policy.replay_report(target, q_sim, q_sim, NAMES)["r_hip_z"]["rms_deg"] == pytest.approx(0.0)


def test_load_replay_refuses_a_wrong_file(tmp_path):
    good = dict(target=sine_targets(), q_sim=sine_targets(), dt=np.float64(0.02), servo_names=np.array(NAMES))
    f = tmp_path / "ok.npz"
    np.savez(f, **good)
    target, q_sim = run_policy.load_replay(f, NAMES)
    assert target.shape == (500, 12)
    with pytest.raises(Abort):
        run_policy.load_replay(f, NAMES[::-1])
    bad = tmp_path / "bad.npz"
    np.savez(bad, **{**good, "dt": np.float64(0.01)})
    with pytest.raises(Abort):
        run_policy.load_replay(bad, NAMES)


class FakeBus:
    """Answers register reads from a table. A register not in the table raises, like a servo that does not reply."""

    def __init__(self, regs):
        self.regs = regs

    def transfer(self, packet, response_data_len=0):
        _sid, address, size = packet
        assert size == response_data_len
        value = self.regs[address]
        return bytes([value & 0xFF, value >> 8][:size])


class FakeArr:
    def __init__(self, regs):
        self.bus = FakeBus(regs)

    def _bus_for_id(self, _sid):
        return self.bus


GOOD_REGS = {0x15: 32, 0x16: 32, 0x17: 0, 0x55: 0, 0x29: 0, 0x10: 1000, 0x30: 1000, 0x09: 0, 0x0B: 4095, 0x38: 2048, 0x1A: 1, 0x1B: 1, 0x07: 0,
             0x1C: 500, 0x22: 20, 0x23: 200, 0x24: 80, 0x3E: 125, 0x3F: 34}
JOINTS = [type("J", (), {"name": n, "servo_id": i + 1})() for i, n in enumerate(NAMES)]


def encode_for_test(sid, address, size):
    return (sid, address, size)


def test_readback_reads_two_byte_registers_and_passes_good_servos():
    said = []
    r = run_policy.readback_stage(FakeArr(GOOD_REGS), JOINTS, said.append, encode_for_test)
    assert r["problems"] == [] and r["rows"]["r_knee"]["torque_limit"] == 1000 and r["rows"]["l_knee"]["max_pos"] == 4095 and r["rows"]["l_knee"]["id"] == 10
    assert any("match training" in s for s in said) and any("prot_torque 20" in s for s in said)


@pytest.mark.parametrize("change,want", [({0x55: 50}, "acceleration cap reads 50"), ({0x30: 500}, "torque_limit reads 500"), ({0x10: 700}, "max_torque reads 700"),
                                         ({0x15: 16}, "gains P 16"), ({0x38: 3000, 0x0B: 2500}, "sits at 3000")])
def test_readback_names_each_difference_from_training(change, want):
    r = run_policy.readback_stage(FakeArr({**GOOD_REGS, **change}), JOINTS, lambda s: None, encode_for_test)
    assert len(r["problems"]) == 12 and all(want in line for line in r["problems"])


def test_readback_shows_an_unread_register_as_unknown_not_zero():
    regs = {k: v for k, v in GOOD_REGS.items() if k != 0x55}
    said = []
    r = run_policy.readback_stage(FakeArr(regs), JOINTS, said.append, encode_for_test)
    assert r["rows"]["r_knee"]["cap_0x55"] is None and any("could not read: cap_0x55" in line for line in r["problems"]) and any("cap_0x55 ?" in s for s in said)
    assert not any("acceleration cap reads" in line for line in r["problems"])
