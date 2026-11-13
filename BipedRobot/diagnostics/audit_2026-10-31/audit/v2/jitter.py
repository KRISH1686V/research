"""Jitter / smoothness metrics at the 50 Hz control rate, with the v2 evaluator's env and policy loaders.

Per episode (first 1 s skipped): raw action rate RMS, joint-velocity power fraction above 5 Hz,
torso roll/pitch rate RMS, root vertical/lateral velocity RMS, torque saturation, touchdowns per second,
and the size of the Smooth-A r_smooth penalty (0.05 action rate + 0.05 base motion) relative to the
tracking reward actually earned.

Usage: jitter.py LABEL CHECKPOINT RESOLVED_CONFIG VX OUT_JSON [SEEDS start:stop [REALISM]]
REALISM defaults to easy (no pushes in any realism).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import EvalSettings, load_env_config, load_policy, make_eval_env  # noqa: E402

SKIP = 50


def main():
    label, ckpt, rc, vx, out = sys.argv[1:6]
    a, b = (int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "5000:5010").split(":"))
    realism = sys.argv[7] if len(sys.argv) > 7 else "easy"
    torch.set_num_threads(1)
    settings = EvalSettings(realism=realism)
    cfg = load_env_config(rc)
    policy = load_policy(ckpt, cfg)
    env = make_eval_env(cfg, settings)
    cmd = np.array([float(vx), 0.0, 0.0], np.float32)
    rows = []
    for seed in range(a, b):
        obs, _ = env.reset(seed=seed, options={"desired_goal": cmd})
        A, QD, W, V, TS, C, TR = [], [], [], [], [], [], []
        term = trunc = False
        steps = 0
        while not (term or trunc) and steps < settings.max_steps:
            act = np.asarray(policy(obs), np.float32)
            obs, _, term, trunc, info = env.step(act)
            A.append(act)
            QD.append([env.data.qvel[d] for d in env.joint_dof_adrs])
            W.append(env.data.qvel[3:6].copy())
            V.append(env.data.qvel[0:3].copy())
            TS.append(float(info.get("torque_saturation_ratio", np.nan)))
            C.append(np.asarray(env._feet_contact(), float))
            TR.append(float(info.get("r_track", np.nan)))
            steps += 1
        if steps <= SKIP + 50:
            rows.append({"seed": seed, "steps": steps, "survived": False})
            continue
        A, QD, W, V = (np.array(x)[SKIP:] for x in (A, QD, W, V))
        dA = np.diff(A, axis=0)
        spec = np.abs(np.fft.rfft(QD - QD.mean(0), axis=0)) ** 2
        f = np.fft.rfftfreq(len(QD), d=env.dt)
        hf = float(spec[f > 5.0].sum() / spec[f > 0].sum())
        dom = float(f[1:][np.argmax(spec[1:].sum(1))])
        C = np.array(C)[SKIP:]
        td = float(np.sum(np.clip(np.diff(C, axis=0), 0, None)) / (len(C) * env.dt)) if np.isfinite(C).all() else None
        act_rate_ms = float(np.mean(dA ** 2))
        base_motion = float(np.mean(V[:, 2] ** 2 + np.sum(W[:, :2] ** 2, axis=1)))
        rows.append({
            "seed": seed, "steps": steps, "survived": bool(steps >= settings.max_steps and not term),
            "action_rate_rms": float(np.sqrt(act_rate_ms)),
            "joint_vel_power_above_5hz": hf, "joint_vel_dominant_hz": dom,
            "torso_roll_pitch_rate_rms": float(np.sqrt(np.mean(np.sum(W[:, :2] ** 2, axis=1)))),
            "root_vz_rms": float(np.sqrt(np.mean(V[:, 2] ** 2))),
            "root_vy_rms": float(np.sqrt(np.mean((V[:, 1] - V[:, 1].mean()) ** 2))),
            "torque_sat_frac": float(np.nanmean(TS[SKIP:])),
            "touchdowns_per_s": td,
            "smoothA_penalty_per_step": 0.05 * act_rate_ms + 0.05 * base_motion,
            "r_track_mean": float(np.nanmean(TR[SKIP:])),
        })
    env.close()
    Path(out).write_text(json.dumps({"label": label, "checkpoint": ckpt, "realism": realism, "command": cmd.tolist(),
                                     "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
