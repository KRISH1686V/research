"""Clearing the servos' acceleration cap (robot_test.clear_accel_cap), on a fake bus that keeps registers.

The real biped_hardware package is not needed: a stand-in encode_read is installed, and the fake bus answers reads and writes as an STS3215 does.
"""
import sys
import types
from pathlib import Path

import pytest

HW = Path(__file__).resolve().parents[1] / "diagnostics/audit_2026-10-31/claude_audit/v2/hardware_tests"
sys.path.insert(0, str(HW))


def _fake_protocol(monkeypatch):
    def encode_read(sid, address, n):
        body = [sid, 4, 0x02, address, n]
        return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])
    pkg = types.ModuleType("biped_hardware")
    st = types.ModuleType("biped_hardware.st3215")
    proto = types.ModuleType("biped_hardware.st3215.protocol")
    def encode_write(sid, address, data):
        body = [sid, len(data) + 3, 0x03, address, *data]
        return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])
    proto.encode_read, proto.encode_write = encode_read, encode_write
    regs = types.ModuleType("biped_hardware.st3215.registers")
    regs.Reg = types.SimpleNamespace(LOCK_FLAG=0x37)
    monkeypatch.setitem(sys.modules, "biped_hardware.st3215.registers", regs)
    monkeypatch.setitem(sys.modules, "biped_hardware", pkg)
    monkeypatch.setitem(sys.modules, "biped_hardware.st3215", st)
    monkeypatch.setitem(sys.modules, "biped_hardware.st3215.protocol", proto)


class Bus:
    def __init__(self, sid, stuck=False):
        self.reg = {0x55: 50, 0x29: 254, 0x37: 1}
        self.sid, self.stuck, self.writes = sid, stuck, []

    def transfer(self, packet, response_data_len):
        assert packet[:2] == b"\xff\xff" and packet[2] == self.sid
        body = list(packet[2:-1])
        assert packet[-1] == (~sum(body)) & 0xFF, "bad checksum"
        assert packet[3] == len(packet) - 4, "length must count instruction, parameters and checksum"
        instr, address = packet[4], packet[5]
        if instr == 0x02:
            assert response_data_len == packet[6]
            return bytes([self.reg[address]])
        assert instr == 0x03 and response_data_len == 0
        self.writes.append((address, list(packet[6:-1])))
        if not (self.stuck and address == 0x55):
            self.reg[address] = packet[6]
        return b""


class Arr:
    def __init__(self, n=12, stuck_id=None):
        self.buses = {i: Bus(i, stuck=(i == stuck_id)) for i in range(1, n + 1)}
        self.torque = True
        self.torque_when_written = []

    def set_torque(self, on):
        self.torque = on

    def _bus_for_id(self, sid):
        bus = self.buses[sid]
        self.torque_when_written.append(self.torque)
        return bus


class Cfg:
    def __init__(self, n=12):
        self.joints = [types.SimpleNamespace(name=f"j{i}", servo_id=i) for i in range(1, n + 1)]


def test_write_packet_layout():
    from robot_test import write_packet
    assert write_packet(1, 0x55, [0]).hex() == "ffff0104035500a2"
    assert write_packet(1, 0x29, [0]).hex() == "ffff01040329" + "00" + "ce"


def test_clear_writes_cap_then_acc_with_torque_off(monkeypatch):
    _fake_protocol(monkeypatch)
    from robot_test import clear_accel_cap
    arr, said = Arr(), []
    out = clear_accel_cap(arr, Cfg(), said.append)
    assert not any(arr.torque_when_written), "registers must only be touched with torque off"
    for sid, bus in arr.buses.items():
        assert [a for a, _ in bus.writes] == [0x55, 0x29], "0x55 gates 0x29, so 0x55 goes first"
        assert bus.reg == {0x55: 0, 0x29: 0, 0x37: 1}
        assert out[f"j{sid}"] == {"lock": 1, "max_acc_before": 50, "acc_before": 254, "max_acc_after": 0, "acc_after": 0}
    assert "[50]" in said[-1]


def test_clear_stops_when_a_servo_keeps_its_cap(monkeypatch):
    _fake_protocol(monkeypatch)
    from robot_test import Abort, clear_accel_cap
    with pytest.raises(Abort, match="j5 still reads acceleration cap 50"):
        clear_accel_cap(Arr(stuck_id=5), Cfg(), lambda s: None)


def test_mock_leaves_registers_alone():
    from robot_test import clear_accel_cap
    arr = Arr()
    assert clear_accel_cap(arr, Cfg(), lambda s: None, mock=True) == {"mock": True}
    assert all(b.writes == [] for b in arr.buses.values())


def test_free_mode_only_after_clearing():
    from robot_test import FREE, speed_modes
    ctx = types.SimpleNamespace(cfg=types.SimpleNamespace(safety={}), cap_cleared=False)
    assert "free" not in speed_modes(ctx)
    ctx.cap_cleared = True
    assert speed_modes(ctx)["free"] == FREE == (3400, 0)


def test_unlocked_memory_stops_before_any_write(monkeypatch):
    _fake_protocol(monkeypatch)
    from robot_test import Abort, clear_accel_cap
    arr = Arr()
    arr.buses[3].reg[0x37] = 0
    with pytest.raises(Abort, match="j3's permanent memory is unlocked"):
        clear_accel_cap(arr, Cfg(), lambda s: None)
    assert arr.buses[3].writes == [] and arr.buses[3].reg[0x55] == 50
