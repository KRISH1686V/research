"""Tables from survive_eval.py results. Usage: survive_table.py summary FOLDER GLOB | scan FOLDER | arms FOLDER TEST [BUDGETS]"""
import glob
import json
import os
import sys

import numpy as np


def row(r):
    a, e, g = r["aggregate"], r["aggregate"]["all_episodes"], r["aggregate"]["gait_classes"]
    return dict(k=a["survival"], n=a["n"], secs=e["steps"]["mean"] / 50, v=e["speed_x_disp"]["mean"], walk=g.get("alternating_walk", 0), bound=g.get("running_or_bounding", 0),
                hop=g.get("bilateral_hop", 0), stand=g.get("standing", 0), irregular=g.get("irregular", 0), over=r["episodes_with_overdraft"],
                strict=sum(x.get("strict_ever_overdrawn", x["ever_overdrawn"]) for x in r["per_episode"]), rate=r["rate"], cost=r["measured_cost"],
                peak=float(np.median([x["peak_joint_speed"] for x in r["per_episode"]])), off=float(np.median([abs(x["lateral_drift_m"]) for x in r["per_episode"]])))


def summary(folder, pattern):
    for f in sorted(glob.glob(os.path.join(folder, pattern))):
        d = json.load(open(f))
        print(f"\n{os.path.basename(f)}  ({d['realism']}, pushes {d['pushes']}, {d['seconds']:.0f} s, seeds {d['seeds']})")
        print(f"{'budget':<14}{'up':>7}{'mean s':>8}{'m/s':>7}{'walk':>5}{'bound':>6}{'hop':>4}{'stand':>6}{'irreg':>6}{'off line m':>11}{'peak rad/s':>11}{'overdrawn':>10}{'strict':>7}  rate -> cost (flight, slip, torque)")
        tk = tn = 0
        for name, r in d["results"].items():
            x = row(r)
            tk, tn = tk + x["k"], tn + x["n"]
            pair = "  ".join(f"{a:.3f}->{b:.3f}" for a, b in zip(x["rate"], x["cost"])) if x["cost"] else "NOT DETERMINED"
            print(f"{name:<14}{x['k']:>4}/{x['n']:<2}{x['secs']:>8.1f}{x['v']:>7.3f}{x['walk']:>5}{x['bound']:>6}{x['hop']:>4}{x['stand']:>6}{x['irregular']:>6}{x['off']:>11.2f}{x['peak']:>11.2f}{x['over']:>10}{x['strict']:>7}  {pair}")
        print(f"{'total':<14}{tk:>4}/{tn} = {tk / tn:.0%}")


def scan(folder):
    arms = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(os.path.join(folder, "*.json"))})
    for arm in arms:
        print(f"\n{arm}: up out of 20, then speed m/s, then alternating-walk episodes; full realism with pushes, 10 s")
        for name in ("all_loose", "all_mid"):
            print(f"  {name}")
            files = sorted(glob.glob(os.path.join(folder, f"{arm}_*.json")), key=lambda f: int(os.path.basename(f).rsplit("_", 1)[1][:-5]))
            for f in files:
                x = row(json.load(open(f))["results"][name])
                print(f"    {int(os.path.basename(f).rsplit('_', 1)[1][:-5]) / 1e6:>5.2f}M  {x['k']:>3}/{x['n']}  {x['secs']:>5.1f} s  {x['v']:>6.3f}  walk {x['walk']:>2}  bound {x['bound']:>2}  stand {x['stand']:>2}")


def arms(folder, test, budgets=None):
    """One line per method, with the training seed as the unit: files <arm>_s<seed>_<test>.json. Share of episodes upright, then speed, then share walking, each per seed."""
    table = {}
    for f in glob.glob(os.path.join(folder, f"*_s[0-9]*_{test}.json")):
        name = os.path.basename(f)[:-len(f"_{test}.json")]
        if not name.rsplit("_s", 1)[1].isdigit():  # e.g. a separate _last4 set of files
            continue
        arm, seed = name.rsplit("_s", 1)
        d = json.load(open(f))["results"]
        rows = [row(r) for n, r in d.items() if budgets is None or n in budgets]
        n = sum(x["n"] for x in rows)
        table.setdefault(arm, {})[int(seed)] = (sum(x["k"] for x in rows) / n, float(np.mean([x["v"] for x in rows])), sum(x["walk"] for x in rows) / n)
    print(f"{test}, budgets {'all tested' if budgets is None else ','.join(budgets)}")
    print(f"{'method':<24}{'seeds':>6}{'up, per seed':>26}{'m/s, per seed':>26}{'walking, per seed':>26}")
    for arm in sorted(table):
        seeds = sorted(table[arm])
        cols = ["  ".join(f"{table[arm][s][i] * (100 if i != 1 else 1):.0f}" if i != 1 else f"{table[arm][s][1]:.3f}" for s in seeds) for i in range(3)]
        print(f"{arm:<24}{len(seeds):>6}{cols[0]:>26}{cols[1]:>26}{cols[2]:>26}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "arms":
        arms(sys.argv[2], sys.argv[3], sys.argv[4].split(",") if len(sys.argv) > 4 else None)
    else:
        scan(sys.argv[2])
