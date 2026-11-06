"""Turn a trained checkpoint into the plain numbers run_policy.py needs on the Pi. Runs on the workstation.

Usage: export_policy.py CHECKPOINT_PT RESOLVED_CONFIG OUT_NPZ [--allow-truth-inputs]
--allow-truth-inputs lets through a policy that reads true speed or budget balances. run_policy.py still refuses it on real servos, so it is for simulator tests only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from BipedRobot.evaluation_suite_v2 import load_env_config  # noqa: E402
from BipedRobot.fast.collector import env_class  # noqa: E402

JOINT_MAP = Path(__file__).resolve().parent / "joint_map.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("resolved_config")
    p.add_argument("out")
    p.add_argument("--allow-truth-inputs", action="store_true")
    args = p.parse_args(argv)

    state = torch.load(args.checkpoint, map_location="cpu")
    actor = state["actor"]
    obs_dim = int(state["obs_dim"])
    keep = actor["keep"].numpy() if "keep" in actor else np.arange(obs_dim)
    reads_truth = bool(np.isin(keep, np.r_[np.arange(3), np.arange(obs_dim - 3, obs_dim)]).any())
    if reads_truth and not args.allow_truth_inputs:
        sys.exit("This policy reads true speed or budget balances, which the robot cannot measure. Train with --actor_hide achieved,levels.")

    cfg = load_env_config(args.resolved_config)
    env = env_class(cfg)(config=cfg, render_mode=None)
    if not (env.action_filter_enabled and env.prev_action_obs_enabled):
        sys.exit("run_policy.py assumes the action filter and its state in the observation, as in the V2 configs")
    history = int(getattr(env, "history", 1))
    clock_hz = float(getattr(env, "clock_hz", 0.0))
    if obs_dim != 3 + 6 + 42 * history + (2 if clock_hz else 0) + 3:
        sys.exit(f"Unexpected observation size {obs_dim} for history {history}")
    rows = json.loads(JOINT_MAP.read_text())["rows"]
    for i, r in enumerate(rows):
        adr = env.joint_ids[i]
        name = env.model.joint(adr).name
        if r["train_index"] != i or r["train_name"] != name:
            sys.exit(f"joint_map.json row {i} says {r['train_name']}, the training model says {name}")
    s = env.obs_scale
    meta = {"checkpoint": str(args.checkpoint), "sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), "steps": state.get("steps"), "xml": env.xml_path,
            "joint_train_names": [r["train_name"] for r in rows]}
    np.savez(args.out, w0=actor["latent_pi.0.weight"].numpy(), b0=actor["latent_pi.0.bias"].numpy(), w1=actor["latent_pi.2.weight"].numpy(), b1=actor["latent_pi.2.bias"].numpy(),
             mu_w=actor["mu.weight"].numpy(), mu_b=actor["mu.bias"].numpy(), keep=keep, obs_dim=obs_dim, history=history, reads_truth=reads_truth,
             scale=np.array([s["joint_pos"], s["joint_vel"], s["base_ang_vel"], s["proj_gravity"], s["prev_action"]], np.float32),
             default=env.default_qpos, lo=env.jnt_min, hi=env.jnt_max, ctrl_min=env.ctrl_min, ctrl_max=env.ctrl_max,
             clock_hz=clock_hz, fraction=env.action_range_fraction, alpha=env.action_filter_alpha, servo_names=np.array([r["servo_name"] for r in rows]), meta=json.dumps(meta))
    env.close()
    print(f"wrote {args.out}: {len(keep)} policy inputs of {obs_dim}, reads truth inputs: {reads_truth}")


if __name__ == "__main__":
    main()
