"""The robot script's sine test, run in the simulator, so robot and simulator lag are measured the same way.

Same procedure as test_track in robot_test.py: right leg, one joint at a time, 5 deg, 50 Hz commands, position read just
before each new command, first cycle ramped and not analysed. Uses the training env's own actuator chain (delay, lag,
backlash, servo curve). The robot is held in the air by giving the base a huge inertia. Domain randomisation is off and
the delay is pinned at each end of its training range. Starts at the home pose; the real test starts where the leg hangs.

Usage: sim_track.py OUT_JSON
"""
import copy
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from BipedRobot.fast.collector import env_class  # noqa: E402
from BipedRobot.train_her_biped import build_env_config  # noqa: E402
from robot_test import analyze_track  # noqa: E402

CONFIG = ROOT / "BipedRobot/config/v2_budget_survive.yaml"
JOINTS = ["r_hip_z", "r_hip_x", "r_hip_y", "r_knee", "r_ankle_y", "r_ankle_x"]
DELAYS_MS = (0.0, 50.0)
FREQS = (1.0, 2.0, 3.0)
AMP = math.radians(5.0)


def run(env, index, freq, delay_s):
    env.reset(seed=1)
    env._active_delay = delay_s
    env._motor_strength_scale = np.ones(env.num_joints, dtype=np.float32)
    env.model.dof_armature[:6] = 1e6
    env.data.qpos[2] = 0.40
    env.data.qvel[:] = 0.0
    for i, adr in enumerate(env.joint_qpos_adrs):
        env.data.qpos[adr] = env.default_qpos[i]
    settle = 1.0 / freq
    cycles = max(4, int(math.ceil(3.0 * freq)))
    n = int(round((settle + cycles / freq) / env.dt))
    span_hi = env.jnt_max[index] - env.default_qpos[index]
    span_lo = env.default_qpos[index] - env.jnt_min[index]
    t_all, cmd, pos = [], [], []
    action = np.zeros(env.num_joints, dtype=np.float32)
    for k in range(n):
        now = k * env.dt
        offset = min(1.0, now / settle) * AMP * math.sin(2 * math.pi * freq * now)
        action[index] = offset / (env.action_range_fraction * (span_hi if offset >= 0 else span_lo))
        t_all.append(now)
        cmd.append(env.default_qpos[index] + offset)
        pos.append(float(env.data.qpos[env.joint_qpos_adrs[index]]))
        env.step(action)
    keep = [i for i, x in enumerate(t_all) if x >= settle]
    t = [t_all[i] for i in keep]
    return analyze_track(t, [cmd[i] for i in keep], t, [pos[i] for i in keep], freq)


def main():
    config = yaml.safe_load(CONFIG.read_text())
    env_config = build_env_config(config, "full", False)
    env_config["push_enabled"] = False
    env_config["domain_randomization"]["enabled"] = False
    env_config["shaping"]["action_filter"] = False       # the robot test sends targets straight to the servo
    env = env_class(env_config)(config=copy.deepcopy(env_config), render_mode=None)
    per = {}
    for name in JOINTS:
        index = JOINTS.index(name)
        for d in DELAYS_MS:
            for f in FREQS:
                per.setdefault(name, {}).setdefault(str(d), {})[str(f)] = run(env, index, f, d / 1000)
    summary = {str(d): {str(f): {k: statistics.median(per[n][str(d)][str(f)][k] for n in JOINTS) for k in ("gain", "lag_ms")} for f in FREQS} for d in DELAYS_MS}
    Path(sys.argv[1]).write_text(json.dumps({"kp": float(env.model.actuator_gainprm[env.actuator_ids[0], 0]), "summary": summary, "joints": per}, indent=1))
    print("kp", float(env.model.actuator_gainprm[env.actuator_ids[0], 0]), "backlash", env.enable_backlash, "lag", env.enable_lag, "delay", env.enable_delay)
    for d in DELAYS_MS:
        for f in FREQS:
            s = summary[str(d)][str(f)]
            print(f"delay {d:>4.0f} ms, {f:.0f} Hz: lag {s['lag_ms']:6.1f} ms, gain {s['gain']:.3f}")
    print("per joint at delay 0, 1 Hz:", {n: round(per[n]["0.0"]["1.0"]["lag_ms"], 1) for n in JOINTS})
    print("per joint at delay 0, 3 Hz:", {n: round(per[n]["0.0"]["3.0"]["lag_ms"], 1) for n in JOINTS})


if __name__ == "__main__":
    main()
