"""Score survive_eval.py results against the goal fixed in diagnostics/WORKING_RULES.md. The thresholds must not change once results are in.

Usage: goal_score.py BUDGETS_TEMPLATE [NO_BUDGETS_TEMPLATE]
Templates hold {s} for the seed, for example long_check/s{s}_medium.json. A seed without a file is NOT DETERMINED, and the goal cannot pass until enough seeds are in.
"""
import json
import os
import sys

import numpy as np

SEEDS = (3, 4, 5, 6, 7)
NEEDED = 3
UP, SPEED, WALK, MOVING = 90, (0.15, 0.25), 50, 50


def row(result):
    a, p = result["aggregate"], result["per_episode"]
    assert len(p) == 100, f"the goal needs 100 episodes, got {len(p)}"
    steady = [x["speed_x_disp_steady"] for x in p if x["survived"] and x["speed_x_disp_steady"] is not None]
    gaits = a["gait_classes"]
    return {"up": a["survival"], "speed": float(np.mean(steady)) if steady else None, "walk": gaits.get("alternating_walk", 0), "bound": gaits.get("running_or_bounding", 0),
            "stand": gaits.get("standing", 0), "flight": a["all_episodes"]["flight_frac"]["mean"],
            "moving": sum(x["survived"] and not x["ever_overdrawn"] and (x["speed_x_disp_steady"] or 0) >= 0.10 for x in p)}


def score(template):
    passed = have = 0
    print("seed | loose up, speed | mid up, walk, in budget and moving | result")
    for s in SEEDS:
        path = template.format(s=s)
        if not os.path.exists(path):
            print(f"{s} | NOT DETERMINED")
            continue
        have += 1
        d = json.load(open(path))["results"]
        L, M = row(d["all_loose"]), row(d["all_mid"])
        checks = {"loose up": L["up"] >= UP, "mid up": M["up"] >= UP, "loose speed": L["speed"] is not None and SPEED[0] <= L["speed"] <= SPEED[1], "mid walk": M["walk"] >= WALK, "mid moving": M["moving"] >= MOVING}
        ok = all(checks.values())
        passed += ok
        print(f"{s} | {L['up']}, {L['speed']:.3f} | {M['up']}, {M['walk']}, {M['moving']} | {'PASS' if ok else 'fail: ' + ', '.join(k for k, v in checks.items() if not v)}")
    verdict = "MET, if budgets off is clearly worse" if passed >= NEEDED else ("NOT MET" if passed + len(SEEDS) - have < NEEDED else "NOT DETERMINED")
    print(f"{passed} of {have} seeds in pass, {NEEDED} of {len(SEEDS)} needed: {verdict}")


def compare(budgets, off):
    rows = {}
    for name, template in (("budgets", budgets), ("off", off)):
        rows[name] = [row(json.load(open(template.format(s=s)))["results"]["all_loose"]) for s in SEEDS if os.path.exists(template.format(s=s))]
    print("loose budget, seeds in each arm:", {k: len(v) for k, v in rows.items()})
    for key in ("up", "speed", "walk", "bound", "stand", "flight"):
        print(key, "budgets", [round(r[key], 3) for r in rows["budgets"]], "off", [round(r[key], 3) for r in rows["off"]])


if __name__ == "__main__":
    score(sys.argv[1])
    if len(sys.argv) > 2:
        compare(sys.argv[1], sys.argv[2])
