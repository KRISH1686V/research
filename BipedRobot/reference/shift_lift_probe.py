"""Decoupled lateral-shift and slow-lift probe with optional CoM feedback aid."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml"

from BipedRobot.envs.her_loco_env import HERLocoEnv
from BipedRobot.reference.lift_diagnosis import ik, local_feet


def trial(raw, shift, feedback, lift_cm):
    config = copy.deepcopy(raw)
    config["domain_randomization"]["enabled"] = False
    env = HERLocoEnv(config=config, render_mode=None)
    env.reset(seed=7600 + int(shift * 1000) + int(feedback))
    leg, support = 0, 1
    stable = True
    max_clearance = 0.0
    for step in range(150):
        feet = local_feet(env)
        amount = shift * min(1.0, (step + 1) / 50.0)
        feet[leg][1] += amount
        feet[support][1] -= amount
        if step >= 50:
            ramp = min(1.0, (step - 49) / 100.0)
            feet[leg][2] += lift_cm * ramp
        if feedback:
            com_y = float(env.data.subtree_com[env.root_body_id, 1])
            support_y = float(env.data.xpos[env.foot_body_ids[support], 1])
            correction = np.clip(-0.4 * (com_y - support_y), -0.03, 0.03)
            feet[leg][1] += correction
            feet[support][1] += correction
        action, raw_action, _ = ik(env, feet, 0.5)
        _, _, terminated, truncated, _ = env.step(action)
        max_clearance = max(max_clearance, float(env.data.xpos[env.foot_body_ids[leg], 2]))
        if terminated or truncated:
            stable = False
            break
    env.close()
    return {"shift_cm": shift * 100, "feedback": feedback, "lift_cm": lift_cm * 100, "stable": stable, "steps": step + 1, "max_clearance_cm": max_clearance * 100}


def main():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = []
    for feedback in (False, True):
        for shift in np.arange(0.01, 0.101, 0.01):
            rows.append(trial(raw, float(shift), feedback, 0.0))
        for lift in (0.01, 0.02, 0.04):
            rows.append(trial(raw, 0.04, feedback, lift))
    out = ROOT / "BipedRobot/paper_results/shift_lift_probe.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
