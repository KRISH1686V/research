"""The Pi policy runner against the training env: same numbers, same joint wiring, whole stack closed through MuJoCo, and every safety stop.

Run by path from the workspace root. Needs BIPED_WS pointing at a biped_ws copy for the wiring and mock tests.
"""
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "diagnostics/audit_2026-10-31/claude_audit/v2/hardware_tests"
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(TOOLS))
import export_policy  # noqa: E402
import robot_test as rt  # noqa: E402
import run_policy as rp  # noqa: E402
from BipedRobot.fast.budget_env import BudgetLocoEnv  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.nets import Actor, actor_from_checkpoint, actor_inputs  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402

WS = Path(os.environ["BIPED_WS"]).expanduser() if os.environ.get("BIPED_WS") else None
needs_ws = pytest.mark.skipif(WS is None, reason="set BIPED_WS to a biped_ws copy")
GOAL = np.array([0.2, 0.0, 0.0, 1.0, 1.0, 1.0], np.float32)
HAS_SERIAL = importlib.util.find_spec("serial") is not None


def env_config():
    """Easy realism with no noise: the runner has no delay, lag, backlash or noise of its own, so everything should match exactly."""
    c = yaml.safe_load(open(REPO / "config/v2_budget_window.yaml"))
    ec = build_env_config(c, "easy", False)
    ec["validity"] = {"friction": True, "motor_torque": True, "history": 5}
    ec["token_budget"] = c["token_budget"]
    ec.setdefault("shaping", {})["reset_velocity"] = False
    return ec


def resolved(tmp, ec):
    path = tmp / "resolved.yaml"
    path.write_text(yaml.safe_dump({"resolved_env_config": ec}))
    return str(path)


def make_env(ec=None):
    env = BudgetLocoEnv(config=ec or env_config())
    env.reward_mode = "track"
    return env


def exported(tmp, hide=("achieved", "levels"), seed=0):
    torch.manual_seed(seed)
    keep = actor_inputs(222, 6, 3, set(hide)) if hide else None
    actor = Actor(222, 12, keep=keep).eval()
    ckpt = tmp / f"policy_{seed}.pt"
    torch.save({"actor": actor.state_dict(), "obs_dim": 222, "act_dim": 12, "steps": 7}, ckpt)
    out = tmp / f"policy_{seed}.npz"
    export_policy.main([str(ckpt), resolved(tmp, env_config()), str(out)] + ([] if hide else ["--allow-truth-inputs"]))
    return actor, str(out)


def sensors(env):
    d = env.data
    return (np.array([d.qpos[a] for a in env.joint_qpos_adrs], np.float32), np.array([d.qvel[a] for a in env.joint_dof_adrs], np.float32),
            d.qvel[3:6].astype(np.float32), env._get_projected_gravity())


def test_export_refuses_a_policy_that_reads_truth(tmp_path):
    actor = Actor(222, 12).eval()
    ckpt = tmp_path / "p.pt"
    torch.save({"actor": actor.state_dict(), "obs_dim": 222, "act_dim": 12, "steps": 1}, ckpt)
    with pytest.raises(SystemExit):
        export_policy.main([str(ckpt), resolved(tmp_path, env_config()), str(tmp_path / "p.npz")])


def test_core_matches_the_env_and_the_network_number_by_number(tmp_path):
    actor, path = exported(tmp_path)
    core = rp.PolicyCore(path)
    env = make_env()
    assert not core.reads_truth and len(core.keep) == 216
    checked = 0
    for seed in range(4):
        o, _ = env.reset(seed=seed, options={"desired_goal": GOAL})
        core.reset()
        for _ in range(120):
            action, target = core.step(*sensors(env), GOAL[:3], GOAL[3:])
            flat = flatten_obs(o)
            assert np.allclose(core.frames[-1], o["observation"][168:210], atol=1e-6), "the newest frame must equal the env's"
            assert np.allclose(np.concatenate(core.frames), o["observation"][:210], atol=1e-6), "the whole history must equal the env's"
            want = actor.deterministic(torch.as_tensor(flat)[None])[0].detach().numpy()
            assert np.allclose(action, want, atol=2e-5), "the numpy network must equal the torch network"
            o, _, term, trunc, _ = env.step(action)
            assert np.allclose(core.filtered, env._action_filter_state, atol=1e-6)
            assert np.allclose(target, env.data.ctrl[env.actuator_ids], atol=1e-5), "joint targets must equal what the env sends"
            checked += 1
            if term or trunc:
                break
    assert checked > 100
    env.close()


def test_action_scale_shrinks_targets_towards_home(tmp_path):
    _, path = exported(tmp_path)
    core = rp.PolicyCore(path)
    env = make_env()
    env.reset(seed=1, options={"desired_goal": GOAL})
    s = sensors(env)
    a1, t1 = core.step(*s, GOAL[:3], GOAL[3:], scale=1.0)
    core.reset()
    a2, t2 = core.step(*s, GOAL[:3], GOAL[3:], scale=0.3)
    assert np.allclose(a2, 0.3 * a1, atol=1e-6)
    assert np.abs(t2 - core.default).max() < np.abs(t1 - core.default).max()
    core.reset()
    _, t3 = core.step(*s, GOAL[:3], GOAL[3:], scale=0.0)
    assert np.allclose(t3, core.default, atol=1e-6), "scale 0 holds the home pose"
    env.close()


def test_gravity_formula_equals_the_simulator():
    env = make_env()
    env.reset(seed=1, options={"desired_goal": GOAL})
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        env.data.qpos[3:7] = q
        assert np.allclose(rt.gravity_in_body(q), env._get_projected_gravity(), atol=1e-6)
    env.close()


def test_imu_mount():
    assert np.array_equal(rp.imu_mount("x,y,z"), np.eye(3))
    m = rp.imu_mount("-y,x,z")
    assert np.array_equal(m @ np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
    for bad in ("x,y", "x,x,z", "x,y,-z", "a,b,c"):
        with pytest.raises(ValueError):
            rp.imu_mount(bad)


def hardware():
    if not HAS_SERIAL and "serial" not in sys.modules:
        stub = types.ModuleType("serial")
        stub.EIGHTBITS, stub.PARITY_NONE, stub.STOPBITS_ONE = 8, "N", 1
        stub.Serial = None
        sys.modules["serial"] = stub
    sys.path.insert(0, str(WS / "biped_hardware"))
    from biped_hardware.config import load_config
    from biped_hardware.servo_array import Status
    return load_config, Status


class Reading:
    def __init__(self, q=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)):
        self.quaternion, self.gyro = q, gyro


class FakeArr:
    """ServoArray's public calls, over a table of joint angles in radians."""

    def __init__(self, cfg, status_cls, q_by_name):
        self.cfg, self.status_cls, self.q = cfg, status_cls, q_by_name
        self.written = None
        self.temp, self.volts = 30, 12.0
        self.silent = set()

    def read_status(self):
        out = {}
        for j in self.cfg.joints:
            out[j.name] = None if j.name in self.silent else self.status_cls(int(round(j.rad_to_steps(self.q[j.name]))) % 4096, 0, 0, self.volts, self.temp)
        return out

    def write_positions(self, steps, speed, accel):
        self.written = dict(steps)


class FakeImu:
    def __init__(self):
        self.reading = Reading()
        self.n = 0

    def read(self):
        self.n += 1
        self.reading = Reading(self.reading.quaternion, (self.n * 1e-3, 0.0, 0.0))
        return self.reading


def config():
    load_config, Status = hardware()
    return load_config(str(WS / "biped_hardware/config/hardware.yaml"), str(WS / "biped_hardware/config/servos.yaml"), None), Status


@needs_ws
def test_wiring_from_training_order_to_servos(tmp_path):
    _, path = exported(tmp_path)
    core = rp.PolicyCore(path)
    cfg, Status = config()
    assert core.servo_names[1] == "r_hip_pitch_joint" and core.servo_names[2] == "r_hip_roll_joint", "hip pitch and roll swap between training and servo order"
    q_train = np.array([0.11, -0.3, 0.05, -0.7, -0.2, 0.09, -0.12, -0.25, 0.07, -0.6, -0.15, -0.04])
    arr = FakeArr(cfg, Status, dict(zip(core.servo_names, q_train)))
    robot = rp.Robot(arr, cfg, FakeImu(), core.servo_names, np.eye(3), 3400, 254)
    assert np.allclose(robot.read_joints(), q_train, atol=2 * np.pi / 4096), "reads must arrive in training order"
    assert [j.name for j in robot.joints] == core.servo_names
    target = np.array([0.1, -0.25, 0.02, -0.6, -0.1, 0.05, -0.1, -0.3, 0.03, -0.55, -0.12, -0.02], np.float32)
    robot.write(target)
    assert set(arr.written) == {j.name for j in cfg.joints}
    for name, t in zip(core.servo_names, target):
        j = cfg.joint(name)
        assert abs(j.steps_to_rad(int(round(arr.written[name]))) - float(t)) < 2 * np.pi / 4096
    env = make_env()
    for i, name in enumerate(core.servo_names):
        assert env.model.joint(env.joint_ids[i]).name == core.meta["joint_train_names"][i]
    env.close()


class SimClock:
    def __init__(self, sim):
        self.sim = sim

    def now(self):
        return self.sim.t

    def sleep_until(self, t):
        while self.sim.t + 1e-9 < t:
            self.sim.tick()


class SimRobotHardware:
    """The robot as the Pi sees it: encoder steps and IMU readings from MuJoCo, servo steps back into MuJoCo. No shortcuts past the runner's own code."""

    def __init__(self, env, cfg, status_cls, names):
        self.env, self.cfg, self.status_cls, self.names = env, cfg, status_cls, names
        self.t = 0.0
        self.by_name = {n: i for i, n in enumerate(names)}
        self.fell = False
        self.clamped_writes = 0

    def tick(self):
        e = self.env
        for _ in range(e.frame_skip):
            mujoco.mj_step(e.model, e.data)
        e.velocity_window.append(e._get_body_frame_velocity())
        self.t += e.dt
        self.fell = self.fell or bool(e._is_terminated())

    def read_status(self):
        e = self.env
        out = {}
        for j in self.cfg.joints:
            rad = float(e.data.qpos[e.joint_qpos_adrs[self.by_name[j.name]]])
            out[j.name] = self.status_cls(int(round(j.rad_to_steps(rad))) % 4096, 0, 0, 12.0, 30)
        return out

    def write_positions(self, steps, speed, accel):
        e = self.env
        for name, s in steps.items():
            e.data.ctrl[e.actuator_ids[self.by_name[name]]] = self.cfg.joint(name).steps_to_rad(int(round(s)))

    def read(self):
        e = self.env
        return Reading(tuple(float(x) for x in e.data.qpos[3:7]), tuple(float(x) for x in e.data.qvel[3:6]))


CKPT = REPO / "checkpoints/fast_budgetsurvivegait_20261102_seed4/seed_4/sb3_her_final.pt"


@needs_ws
@pytest.mark.skipif(not CKPT.exists(), reason="needs the trained gait policy")
def test_whole_stack_walks_in_mujoco_like_the_env_does(tmp_path):
    """Trained policy through export, PolicyCore, Robot, servo steps and IMU readings, closed through MuJoCo. True speed and joint speed are fed in only because this policy was trained with them."""
    cfg, Status = config()
    npz = str(tmp_path / "walker.npz")
    export_policy.main([str(CKPT), resolved(tmp_path, env_config()), npz, "--allow-truth-inputs"])
    actor = actor_from_checkpoint(torch.load(CKPT, map_location="cpu"))
    core = rp.PolicyCore(npz)
    stack, direct = [], []
    for seed in range(4):
        env = make_env()
        o, _ = env.reset(seed=seed, options={"desired_goal": GOAL})
        start = env.data.qpos[0]
        steps = 0
        for _ in range(500):
            a = actor.deterministic(torch.as_tensor(flatten_obs(o))[None])[0].detach().numpy()
            o, _, term, trunc, _ = env.step(a)
            steps += 1
            if term or trunc:
                break
        direct.append((steps, float(env.data.qpos[0] - start)))

        env.reset(seed=seed, options={"desired_goal": GOAL})
        start = env.data.qpos[0]
        sim = SimRobotHardware(env, cfg, Status, core.servo_names)
        robot = rp.Robot(sim, cfg, sim, core.servo_names, np.eye(3), 3400, 254)
        truth = lambda: {"achieved": env._get_achieved_goal(), "levels": np.ones(3, np.float32), "qd": np.array([env.data.qvel[a] for a in env.joint_dof_adrs], np.float32)}
        log = []
        stopped = None
        try:
            rp.control_loop(core, robot, GOAL[:3], GOAL[3:], 10.0, 1.0, 0.0, lambda s: None, clock=SimClock(sim), log=log, truth=truth)
        except rt.Abort as exc:
            stopped = str(exc)
        stack.append((stopped, float(env.data.qpos[0] - start), len(log)))
        env.close()
    print("direct (steps, metres):", direct, "stack (stopped, metres, loops):", stack)
    assert all(s == 500 for s, _ in direct), "the check needs a policy that stays up in the env"
    assert all(n == 500 and stopped is None for stopped, _, n in stack), f"with true joint speed, the stack should stay up like the env: {stack}"
    d, s = np.mean([m for _, m in direct]), np.mean([m for _, m, _ in stack])
    assert d > 0.3 and 0.5 * d < s < 1.5 * d, f"distance through the stack {s:.2f} m against {d:.2f} m direct"


class ScriptedRobot:
    """Lets each safety test break one thing at a known moment."""

    def __init__(self, core, status_cls, cfg):
        self.names = core.servo_names
        self.cfg, self.status_cls = cfg, status_cls
        self.q = dict(zip(self.names, core.default.astype(float)))
        self.temp, self.volts = 30, 12.0
        self.silent = set()
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.n = 0
        self.frozen_imu = False
        self.on_read = None

    def read_status(self):
        if self.on_read:
            self.on_read(self)
        return {j.name: None if j.name in self.silent else self.status_cls(int(round(j.rad_to_steps(self.q[j.name]))) % 4096, 0, 0, self.volts, self.temp) for j in self.cfg.joints}

    def write_positions(self, steps, speed, accel):
        for name, s in steps.items():
            self.q[name] = self.cfg.joint(name).steps_to_rad(int(round(s)))

    def read(self):
        self.n += 0 if self.frozen_imu else 1
        return Reading(self.quat, (self.n * 1e-3, 0.0, 0.0))


class FakeClock:
    def __init__(self, step=0.02):
        self.t, self.step = 0.0, step

    def now(self):
        return self.t

    def sleep_until(self, t):
        self.t = max(self.t + self.step, t) if self.step > 0.02 else max(self.t, t)


def setup_safety(tmp_path):
    cfg, Status = config()
    _, path = exported(tmp_path)
    core = rp.PolicyCore(path)
    hw = ScriptedRobot(core, Status, cfg)
    robot = rp.Robot(hw, cfg, hw, core.servo_names, np.eye(3), 3400, 254)
    return core, hw, robot


def run_until(core, robot, clock=None, seconds=3.0):
    rp.control_loop(core, robot, GOAL[:3], GOAL[3:], seconds, 0.0, 0.0, lambda s: None, clock=clock or FakeClock())


@needs_ws
def test_healthy_robot_runs_to_the_end(tmp_path):
    core, hw, robot = setup_safety(tmp_path)
    run_until(core, robot)


@needs_ws
def test_heading_hold_runs_and_only_changes_the_yaw_command(tmp_path):
    core, hw, robot = setup_safety(tmp_path)
    hw.quat = (math.cos(0.05), 0.0, 0.0, math.sin(0.05))
    seen = []
    step = core.step
    core.step = lambda q, qd, gyro, gravity, command, u, ramp, **kw: seen.append(np.array(command)) or step(q, qd, gyro, gravity, command, u, ramp, **kw)
    command = np.array([0.2, 0.0, 0.0], np.float32)
    rp.control_loop(core, robot, command, GOAL[3:], 0.5, 0.0, 0.0, lambda s: None, clock=FakeClock(), hold_heading=True)
    assert all(c[0] == pytest.approx(0.2) and c[1] == 0.0 for c in seen) and abs(seen[0][2]) < 1e-6, "the first reading sets the heading to hold"
    hw.quat = (math.cos(0.1), 0.0, 0.0, math.sin(0.1))
    seen.clear()
    assert command[2] == 0.0, "the caller's command is left alone"


@needs_ws
@pytest.mark.parametrize("fault,text", [("heat", "too hot"), ("volts", "too low"), ("silent", "stopped answering"), ("tilt", "tilted"), ("imu", "stopped updating")])
def test_each_fault_stops_the_run(tmp_path, fault, text):
    core, hw, robot = setup_safety(tmp_path)
    calls = []

    def breaker(h):
        calls.append(1)
        if len(calls) > 10:
            if fault == "heat":
                h.temp = 70
            elif fault == "volts":
                h.volts = 5.0
            elif fault == "silent":
                h.silent = {h.names[3]}
            elif fault == "tilt":
                h.quat = (np.cos(0.5), np.sin(0.5), 0.0, 0.0)
            elif fault == "imu":
                h.frozen_imu = True
    hw.on_read = breaker
    with pytest.raises(rt.Abort, match=text):
        run_until(core, robot)


@needs_ws
def test_a_single_missed_reply_is_tolerated(tmp_path):
    core, hw, robot = setup_safety(tmp_path)
    calls = []

    def hiccup(h):
        calls.append(1)
        h.silent = {h.names[0]} if len(calls) == 10 else set()
    hw.on_read = hiccup
    run_until(core, robot)


@needs_ws
def test_a_stuck_joint_stops_the_run(tmp_path):
    core, hw, robot = setup_safety(tmp_path)
    name = core.servo_names[3]
    hw.write_positions = lambda steps, speed, accel: None
    hw.q[name] = float(core.default[3]) + 1.2
    with pytest.raises(rt.Abort, match="catching up"):
        run_until(core, robot)


@needs_ws
def test_late_loops_stop_the_run(tmp_path):
    core, hw, robot = setup_safety(tmp_path)
    with pytest.raises(rt.Abort, match="ran late"):
        run_until(core, robot, clock=FakeClock(step=0.04))
    core, hw, robot = setup_safety(tmp_path)
    with pytest.raises(rt.Abort, match="took"):
        run_until(core, robot, clock=FakeClock(step=0.2))


@needs_ws
def test_targets_outside_the_limits_are_refused(tmp_path):
    core, hw, robot = setup_safety(tmp_path)
    target = core.default.copy()
    target[3] = 0.5  # a knee bending the wrong way
    with pytest.raises(rt.Abort, match="outside the limits"):
        robot.write(target)


def mock_main(tmp_path, *extra):
    hardware()
    _, path = exported(tmp_path)
    out = tmp_path / "run"
    code = rp.main([path, "--mock", "--yes", "--seconds", "1", "--biped-ws", str(WS), "--out", str(out), *extra])
    return code, out


@needs_ws
def test_full_run_on_pretend_servos_logs_and_ends_with_torque_off(tmp_path, monkeypatch):
    seen = []
    real = rt.torque_off

    def spy(arr, say):
        seen.append(True)
        return real(arr, say)
    monkeypatch.setattr(rp, "torque_off", spy)
    code, out = mock_main(tmp_path)
    assert code == 0 and seen, "torque must go off at the end"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["stopped"] is None and summary["mock"] is True and summary["loops"] == 50
    data = np.loadtxt(out / "log.csv", delimiter=",", skiprows=1)
    assert data.shape == (50, len(rp.LOG_COLUMNS))
    assert out.with_suffix(".tar.gz").exists()


@needs_ws
def test_truth_reading_policy_is_refused_on_real_servos(tmp_path):
    _, path = exported(tmp_path, hide=())
    with pytest.raises(SystemExit, match="cannot measure"):
        rp.main([path, "--biped-ws", str(WS), "--out", str(tmp_path / "x")])


def test_real_robot_needs_a_person():
    with pytest.raises(SystemExit, match="person"):
        rp.main(["nothing.npz", "--yes"])


@needs_ws
def test_nothing_is_written_to_permanent_memory(tmp_path, monkeypatch):
    hardware()
    from biped_hardware.mock_bus import MockBus
    sent = []
    for method in ("send_no_reply", "transfer", "sync_read"):
        orig = getattr(MockBus, method)

        def recorder(self, packet, *a, orig=orig, **k):
            sent.append(bytes(packet))
            return orig(self, packet, *a, **k)
        monkeypatch.setattr(MockBus, method, recorder)
    code, _ = mock_main(tmp_path)
    assert code == 0
    writes = [p for p in sent if p[4] in (0x03, 0x04, 0x83)]  # WRITE, REG_WRITE, SYNC_WRITE
    assert writes, "no write packets seen, so the test proves nothing"
    assert {p[5] for p in writes} <= {0x28, 0x29, 0x30}, sorted({hex(p[5]) for p in writes})  # torque, position block, limit; EEPROM is below 0x28


def quat_for(mount, body_quat):
    """Sensor orientation from the robot's: the sensor frame is rotated from the robot frame by `mount` (sensor to robot)."""
    import mujoco as mj
    m = np.zeros(4)
    mj.mju_mat2Quat(m, mount.flatten())
    out = np.zeros(4)
    mj.mju_mulQuat(out, body_quat, m)
    return out


@pytest.mark.parametrize("text", ["x,y,z", "-y,x,z", "y,-x,z", "-x,-y,z", "z,y,-x", "x,-z,y"])
def test_guided_imu_check_finds_the_mount_and_accepts_the_right_gyro(text):
    import mujoco as mj
    mount = rp.imu_mount(text)
    rng = np.random.default_rng(0)

    def axis_quat(axis, angle):
        q = np.zeros(4)
        mj.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
        return q

    poses = {"upright": axis_quat([0, 1, 0], 0.0), "forward": axis_quat([0, 1, 0], 0.35), "left": axis_quat([1, 0, 0], -0.35)}
    spins = [np.array([0.0, 0.3, 0.0]), np.array([-0.3, 0.0, 0.0]), np.array([0.0, 0.0, 0.4])]

    class Imu:
        pose = "upright"
        rate = np.zeros(3)

        def read(self):
            q = quat_for(mount, poses[self.pose])
            return Reading(tuple(q + rng.normal(0, 1e-3, 4)), tuple(mount.T @ self.rate + rng.normal(0, 1e-3, 3)))

    imu = Imu()
    script = iter([("pose", "upright"), ("pose", "forward"), ("pose", "left"), ("rate", spins[0]), ("rate", spins[1]), ("rate", spins[2])])

    def ask(_):
        kind, value = next(script)
        if kind == "pose":
            imu.pose = value
        else:
            imu.rate = value
        return ""
    found = rp.imu_stage(imu, ask, lambda s: None, wait=0.0, sleep=lambda s: None)
    assert np.array_equal(rp.imu_mount(found), mount), (found, text)


def test_guided_imu_check_refuses_small_tilts_and_a_wrong_gyro():
    with pytest.raises(ValueError, match="too small"):
        rp.solve_mount(np.array([0, 0, -1.0]), np.array([0.05, 0, -1.0]), np.array([0, 0.05, -1.0]))
    with pytest.raises(ValueError, match="not close"):
        rp.solve_mount(np.array([0, 0, -1.0]), np.array([0.3, 0.3, -0.9]), np.array([0, 0.3, -0.95]))

    class Imu:
        pose = np.array([1.0, 0.0, 0.0, 0.0])
        rate = np.zeros(3)

        def read(self):
            return Reading(tuple(self.pose), tuple(self.rate))
    imu = Imu()
    import mujoco as mj

    def axis_quat(axis, angle):
        q = np.zeros(4)
        mj.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
        return q
    script = iter([axis_quat([0, 1, 0], 0.0), axis_quat([0, 1, 0], 0.35), axis_quat([1, 0, 0], -0.35), np.array([0.0, -0.3, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])])

    def ask(_):
        v = next(script)
        if v.shape == (4,):
            imu.pose = v
        else:
            imu.rate = v
        return ""
    with pytest.raises(rt.Abort, match="does not agree"):
        rp.imu_stage(imu, ask, lambda s: None, wait=0.0, sleep=lambda s: None)


@needs_ws
def test_imu_stage_on_pretend_servos_stops_cleanly_with_torque_off(tmp_path, monkeypatch):
    """The pretend IMU never moves, so the tilts read as too small."""
    seen = []
    real = rt.torque_off

    def spy(arr, say):
        seen.append(True)
        return real(arr, say)
    monkeypatch.setattr(rp, "torque_off", spy)
    code, out = mock_main(tmp_path, "--stage", "imu")
    assert code == 1 and seen
    assert "too small" in json.loads((out / "summary.json").read_text())["stopped"]
