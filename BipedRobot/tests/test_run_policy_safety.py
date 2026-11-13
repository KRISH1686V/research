"""run_policy.py safety additions of 7 Nov 2026: tilt and voltage limits, the servo-settings check, the policy file hash and the pose check (stage poses).
No hardware: a fake robot answers with whatever it was last told, optionally with one joint wrong."""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "BipedRobot/diagnostics/audit_2026-10-31/audit/v2/hardware_tests"
sys.path.insert(0, str(HW))
import run_policy  # noqa: E402
from robot_test import Abort  # noqa: E402

NAMES = run_policy.TRAIN_NAMES


def test_limits():
    assert run_policy.TILT_LIMIT_DEG == 35.0
    assert run_policy.MIN_VOLTS == 11.1
    assert run_policy.EXPECT_PID == (32, 32, 0)


def test_servo_settings_check():
    good = {f"j{i}": {"PID_P": 32, "PID_D": 32, "PID_I": 0, "MAX_ACC": 0} for i in range(12)}
    assert run_policy.servo_setting_problems(good) == []
    bad = {k: dict(v) for k, v in good.items()}
    bad["j3"]["PID_D"] = 0
    bad["j7"]["MAX_ACC"] = 50
    msgs = run_policy.servo_setting_problems(bad)
    assert len(msgs) == 2 and "j3 gains P 32 D 0 I 0" in msgs[0] and "j7 acceleration cap reads 50" in msgs[1]
    assert run_policy.servo_setting_problems(bad, need_cap_cleared=False) == [msgs[0]]
    unknown = {"j0": {"PID_P": "ND", "PID_D": 32, "PID_I": 0, "MAX_ACC": 0}}
    assert run_policy.servo_setting_problems(unknown), "a setting that could not be read must not pass"


def test_policy_file_hash(tmp_path):
    f = tmp_path / "p.npz"
    f.write_bytes(b"abc")
    assert run_policy.file_sha256(f) == hashlib.sha256(b"abc").hexdigest()


class Core:
    default = np.array([0.0, -0.17, 0.0, -0.45, -0.28, 0.0] * 2)


class FakeRobot:
    """Holds whatever it is told. `wrong` = (joint index, degrees added to what it reports), like a reversed or offset joint."""

    def __init__(self, wrong=None):
        self.joints = [type("J", (), {"name": n})() for n in NAMES]
        self.guard = None
        self.q = Core.default.copy()
        self.wrong = wrong
        arr = type("A", (), {})()
        arr.write_positions = lambda d, speed, accel: setattr(self, "q", np.array([d[n] for n in NAMES], float))
        self.arr = arr

    def read_joints(self):
        q = self.q.copy()
        if self.wrong:
            q[self.wrong[0]] += math.radians(self.wrong[1])
        return q


class NoWait:
    """Simulated time: sleeping jumps the clock forward at once, so timeouts still expire."""
    t = 0.0

    @classmethod
    def now(cls):
        return cls.t

    @classmethod
    def sleep_until(cls, t):
        cls.t = max(cls.t, t) + 1e-3


@pytest.fixture(autouse=True)
def plain_steps(monkeypatch):
    monkeypatch.setattr(run_policy, "checked_steps", lambda j, t, guard: t)


def test_pose_targets_move_one_joint():
    t = run_policy.pose_target(Core, "l_knee", -20.0, NAMES)
    d = np.degrees(t - Core.default)
    assert abs(d[NAMES.index("l_knee")] + 20.0) < 1e-9 and np.count_nonzero(np.abs(d) > 1e-9) == 1
    assert np.array_equal(run_policy.pose_target(Core, None, 0.0, NAMES), Core.default)


def test_pose_check_passes_on_a_correct_robot():
    said = []
    out = run_policy.pose_check(FakeRobot(), Core, NAMES, said.append, lambda q: "y", clock=NoWait)
    assert [r["pose"] for r in out] == ["home", "left knee", "right hip roll"]
    assert all(r["seen_right"] for r in out) and max(max(abs(e) for e in r["errors_deg"]) for r in out) < 1e-9


def test_pose_check_stops_on_a_joint_that_misses():
    """A joint 8 deg off where it is told to be (a wrong offset or sign) stops the check at the first pose, before anything else moves."""
    with pytest.raises(Abort, match="could not reach home"):
        run_policy.pose_check(FakeRobot(wrong=(NAMES.index("r_hip_y"), 8.0)), Core, NAMES, lambda s: None, lambda q: "y", clock=NoWait)


def test_pose_check_stops_when_the_person_sees_something_else():
    answers = iter(["y", "n"])
    with pytest.raises(Abort, match="left knee: the robot did not move as expected"):
        run_policy.pose_check(FakeRobot(), Core, NAMES, lambda s: None, lambda q: next(answers), clock=NoWait)
