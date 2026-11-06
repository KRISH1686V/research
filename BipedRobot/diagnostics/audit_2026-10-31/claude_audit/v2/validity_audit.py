"""How much do existing policies depend on the simulator's known gaps? No training: each condition changes one thing at test time.

Uses evaluation_suite_v2.run_episode unchanged, on ValidLocoEnv. Seeds 6000-6029 have not been used for any earlier decision.

Usage: validity_audit.py LABEL CHECKPOINT RESOLVED_CONFIG CONDITION OUT_JSON
"""
import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import EvalSettings, Thresholds, aggregate, load_env_config, load_policy, run_episode  # noqa: E402
from BipedRobot.fast.collector import flatten_obs  # noqa: E402
from BipedRobot.fast.nets import Actor  # noqa: E402
from BipedRobot.fast.valid_env import ValidLocoEnv  # noqa: E402

SERVO = {"stall_nm": 2.94, "noload_rad_s": 4.72}
NOISE = {"joint_pos": 0.002, "joint_vel": 0.1, "gyro": 0.05, "gravity": 0.02}
VALID = dict(friction=True, motor_torque=True, obs_noise=NOISE)
KEEP = "from_config"  # V2_* conditions keep the run's own corrections from its resolved config
WIDE = dict(delay_range=(0.01, 0.05))
# name: realism, pushes, validity, env attributes, model options, frame_skip, velocity-input change
CONDITIONS = {
    "V2_train": dict(realism="full", pushes=True, validity=KEEP),
    "V2_no_push": dict(realism="full", validity=KEEP),
    "V2_medium": dict(validity=KEEP),
    "V2_push_6N": dict(realism="full", pushes=True, validity=KEEP, attrs=dict(dr_push_magnitude=6.0)),
    "V2_delay_60": dict(realism="full", validity=KEEP, attrs=dict(delay_range=(0.06, 0.06))),
    "V2_friction_0.2": dict(realism="full", validity=KEEP, attrs=dict(dr_friction_range=(0.2, 0.2))),
    "V2_torque_x0.5": dict(realism="full", validity=KEEP, attrs=dict(dr_motor_strength_range=(0.5, 0.5))),
    "V_medium": dict(validity=VALID),
    "V_no_push": dict(realism="full", validity=VALID, attrs=WIDE),
    "V_push25": dict(realism="full", pushes=True, validity=VALID, attrs=WIDE),
    "V_push100": dict(realism="full", pushes=True, validity=dict(VALID, push_s=0.1), attrs=WIDE),
    "V_delay_0": dict(realism="full", validity=VALID, attrs=dict(delay_range=(0.0, 0.0))),
    "V_delay_60": dict(realism="full", validity=VALID, attrs=dict(delay_range=(0.06, 0.06))),
    "V_servo_curve": dict(realism="full", validity=dict(VALID, servo_curve=SERVO), attrs=WIDE),
    "V_friction_0.4": dict(realism="full", validity=VALID, attrs=dict(WIDE, dr_friction_range=(0.4, 0.4))),
    "V_torque_x0.8": dict(realism="full", validity=VALID, attrs=dict(WIDE, dr_motor_strength_range=(0.8, 0.8))),
    "base_medium": dict(),
    "dt_2.5ms": dict(frame_skip=8),
    "implicitfast": dict(opt=("integrator", mujoco.mjtIntegrator.mjINT_IMPLICITFAST)),
    "elliptic_cone": dict(opt=("cone", mujoco.mjtCone.mjCONE_ELLIPTIC)),
    "friction_0.6-1.4_real": dict(validity=dict(friction=True)),
    "friction_0.6": dict(validity=dict(friction=True), attrs=dict(dr_friction_range=(0.6, 0.6))),
    "friction_0.4": dict(validity=dict(friction=True), attrs=dict(dr_friction_range=(0.4, 0.4))),
    "delay_40ms": dict(attrs=dict(delay_range=(0.04, 0.04))),
    "torso_mass_x1.2": dict(attrs=dict(dr_mass_scale_range=(1.2, 1.2))),
    "motor_torque_0.85-1.15": dict(validity=dict(motor_torque=True), attrs=dict(dr_motor_strength_range=(0.85, 1.15))),
    "motor_torque_x0.8": dict(validity=dict(motor_torque=True), attrs=dict(dr_motor_strength_range=(0.8, 0.8))),
    "servo_curve": dict(validity=dict(servo_curve=SERVO)),
    "sensor_noise": dict(validity=dict(obs_noise=NOISE)),
    "velocity_input_noise_0.05": dict(velocity="noise"),
    "velocity_input_zero": dict(velocity="zero"),
    "pushes_25ms": dict(realism="full", pushes=True),
    "pushes_100ms": dict(realism="full", pushes=True, validity=dict(push_s=0.1)),
    "all_corrected_no_push": dict(realism="full", validity=dict(friction=True, motor_torque=True, servo_curve=SERVO, obs_noise=NOISE)),
    "all_corrected": dict(realism="full", pushes=True, validity=dict(friction=True, push_s=0.1, motor_torque=True, servo_curve=SERVO, obs_noise=NOISE)),
}


def main():
    label, ckpt, rc, name, out = sys.argv[1:6]
    c = CONDITIONS[name]
    torch.set_num_threads(1)
    realism = c.get("realism", "medium")
    settings = EvalSettings(realism=realism, pushes=c.get("pushes", False))
    base = load_env_config(rc)
    if ckpt.endswith(".pt"):
        state = torch.load(ckpt, map_location="cpu")
        actor = Actor(state["obs_dim"], state["act_dim"])
        actor.load_state_dict(state["actor"])
        actor.eval()

        @torch.no_grad()
        def policy(obs):
            return actor.deterministic(torch.as_tensor(flatten_obs(obs))[None])[0].numpy()
    else:
        policy = load_policy(ckpt, base)
    cfg = copy.deepcopy(base)
    cfg["realism"] = realism
    cfg["push_enabled"] = False
    cfg.setdefault("shaping", {})["reset_velocity"] = False
    if c.get("validity") != KEEP:
        cfg["validity"] = c.get("validity", {})
    if "frame_skip" in c:
        cfg["frame_skip"] = c["frame_skip"]
    env = ValidLocoEnv(config=cfg, render_mode=None)
    env.push_enabled = bool(settings.pushes)
    for k, v in c.get("attrs", {}).items():
        setattr(env, k, v)
    if "opt" in c:
        setattr(env.model.opt, c["opt"][0], c["opt"][1])
    rng = np.random.default_rng(6000)
    velocity = c.get("velocity")

    def policy_fn(obs):
        if velocity is None:
            return policy(obs)
        o = dict(obs)
        o["achieved_goal"] = np.zeros(3, np.float32) if velocity == "zero" else (obs["achieved_goal"] + rng.normal(0, 0.05, 3)).astype(np.float32)
        return policy(o)

    command = np.array([0.2, 0.0, 0.0], np.float32)
    episodes, peak = [], []
    for seed in range(6000, 6030):
        episodes.append(run_episode(policy_fn, env, command, seed, settings, Thresholds()))
    env.close()
    Path(out).write_text(json.dumps({"label": label, "condition": name, "checkpoint": ckpt, "command": command.tolist(), "settings": {"realism": realism, "pushes": settings.pushes},
                                     "validity": cfg.get("validity", {}), "aggregate": aggregate(episodes), "episodes": episodes}, default=str))


if __name__ == "__main__":
    main()
