"""Recompute the provisional nominal gate under error/smoothness sensitivities."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "BipedRobot/paper_results/provisional_exploit_gate.json"
CONFIGS = {
    "A": ROOT / "BipedRobot/config/pilot_gait_smooth_a.yaml",
    "A-min": ROOT / "BipedRobot/config/pilot_gait_min.yaml",
}


def main():
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    rows = []
    for label, path in CONFIGS.items():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        shaping = config["shaping"]
        for vx in (0.2, 0.4):
            analytic = gate["analytic"][label][str(vx)]
            standing = analytic["standing"]
            base_standing = standing["mean_reward"] - standing["per_term"]["r_smooth"]
            smooth = standing["per_term"]["r_smooth"]
            for error_case, achieved in (
                ("undershoot", vx - 0.08),
                ("overshoot", vx + 0.08),
            ):
                for smooth_factor in (1, 3, 5):
                    tracking = __import__("math").exp(
                        -(0.08**2) / (2 * float(config["reward"]["sigma_v"]) ** 2)
                    )
                    progress = float(shaping["progress_weight"]) * max(
                        0.0, min(1.0, achieved / vx)
                    )
                    gait = (
                        2 * (0.35 - 0.25) * float(shaping["air_time_weight"]) * 0.02
                        + 0.70 * float(shaping["single_support_weight"])
                        + 0.70 * float(shaping["swing_clearance_weight"])
                    )
                    nominal = tracking + progress + gait + smooth * smooth_factor
                    standing_total = base_standing + smooth * smooth_factor
                    rows.append(
                        {
                            "config": label,
                            "vx": vx,
                            "error_case": error_case,
                            "smooth_factor": smooth_factor,
                            "standing_total": standing_total,
                            "nominal_total": nominal,
                            "standing_to_nominal_ratio": standing_total / max(nominal, 1e-9),
                        }
                    )
    output = {
        "rows": rows,
        "adversarial_winners": gate["adversarial"],
    }
    destination = ROOT / "BipedRobot/paper_results/gate_sensitivity.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
