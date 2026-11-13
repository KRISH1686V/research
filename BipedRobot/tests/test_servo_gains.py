"""set_servo_gains (12 Nov 2026): P/D gains for one power-up only, written only when the servo's memory lock is closed, read back. CPU only, no robot."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "BipedRobot/diagnostics/audit_2026-10-31/audit/v2/hardware_tests"))
import robot_test  # noqa: E402

LOCK = 0x37


@pytest.fixture(autouse=True)
def fake_registers(monkeypatch):
    try:
        import biped_hardware.st3215.registers  # noqa: F401
    except Exception:
        pkg = types.ModuleType("biped_hardware"); st = types.ModuleType("biped_hardware.st3215"); regs = types.ModuleType("biped_hardware.st3215.registers")
        regs.Reg = type("Reg", (), {"LOCK_FLAG": LOCK})
        monkeypatch.setitem(sys.modules, "biped_hardware", pkg); monkeypatch.setitem(sys.modules, "biped_hardware.st3215", st); monkeypatch.setitem(sys.modules, "biped_hardware.st3215.registers", regs)


class Bus:
    def __init__(self, regs, sticky=False):
        self.regs = {sid: dict(regs) for sid in range(1, 13)}  # each servo has its own registers
        self.sticky, self.writes = sticky, []

    def transfer(self, packet, response_data_len=0):
        if packet[0] == "r":
            return bytes([self.regs[packet[1]][packet[2]]])
        self.writes.append(packet)
        if not self.sticky:
            self.regs[packet[1]][packet[2]] = packet[3]
        return b""


class Arr:
    def __init__(self, bus):
        self.bus, self.torque = bus, True

    def _bus_for_id(self, _sid):
        return self.bus

    def set_torque(self, on):
        self.torque = on


CFG = types.SimpleNamespace(joints=[types.SimpleNamespace(name=f"j{i}", servo_id=i + 1) for i in range(12)])
ER = lambda sid, address, size: ("r", sid, address, size)
EW = lambda sid, address, value: ("w", sid, address, value)


def run(regs, sticky=False, **kw):
    bus = Bus(dict(regs), sticky)
    arr = Arr(bus)
    out = robot_test.set_servo_gains(arr, CFG, lambda s: None, encode_read=ER, encode_write=EW, **kw)
    return out, bus, arr


def test_locked_servos_take_the_new_gain_and_torque_is_off():
    out, bus, arr = run({LOCK: 1, 0x15: 32, 0x16: 32}, p=24)
    assert not arr.torque and len(out) == 12 and all(r["P_before"] == 32 and r["P_after"] == 24 for r in out.values())
    assert all(w[2] == 0x15 for w in bus.writes) and len(bus.writes) == 12  # D untouched


def test_unlocked_memory_stops_before_any_write():
    with pytest.raises(robot_test.Abort):
        run({LOCK: 0, 0x15: 32, 0x16: 32}, p=24)


def test_servo_that_ignores_the_write_stops_the_run():
    with pytest.raises(robot_test.Abort):
        run({LOCK: 1, 0x15: 32, 0x16: 32}, sticky=True, p=24)


def test_range_mock_and_nothing_asked():
    with pytest.raises(robot_test.Abort):
        run({LOCK: 1, 0x15: 32, 0x16: 32}, p=300)
    out, bus, _ = run({LOCK: 1, 0x15: 32, 0x16: 32}, p=24, mock=True)
    assert out["mock"] and not bus.writes
    out, bus, _ = run({LOCK: 1, 0x15: 32, 0x16: 32})
    assert out == {} and not bus.writes
