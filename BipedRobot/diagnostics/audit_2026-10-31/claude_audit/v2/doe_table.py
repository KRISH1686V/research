"""Stage 1 results: score per run and main effect per factor. The training seed is the unit. Score = mean survival % across V2_no_push, V2_medium and V2_train at the final checkpoint."""
import json, sys
from pathlib import Path
import numpy as np
V = Path(__file__).resolve().parent
D = json.load(open(V / "doe_design.json"))
names = [f[0] for f in D["factors"]]
labels = dict(D["factors"])
rows = []
for d in D["design"]:
    for seed in (3, 4):
        r = {"run": d["run"], "seed": seed, "on": d["on"]}
        for cond in ("V2_no_push", "V2_medium", "V2_train"):
            f = V / "doe" / f"{d['run']}_s{seed}__{cond}__final.json"
            if f.exists():
                a = json.load(open(f))["aggregate"]; e = a["all_episodes"]
                r[cond] = a["survival"] / a["n"] * 100
                r[cond + "_speed"] = e["speed_x_disp_steady"]["mean"]
                r[cond + "_walk"] = a["gait_classes"].get("alternating_walk", 0)
        st = []
        for k in ("700k", "800k", "900k"):
            f = V / "doe" / f"{d['run']}_s{seed}__V2_no_push__{k}.json"
            if f.exists():
                a = json.load(open(f))["aggregate"]; st.append(a["survival"] / a["n"] * 100)
        r["recent"] = float(np.mean(st + [r["V2_no_push"]])) if st and "V2_no_push" in r else None
        if all(c in r for c in ("V2_no_push", "V2_medium", "V2_train")):
            r["score"] = float(np.mean([r["V2_no_push"], r["V2_medium"], r["V2_train"]]))
        rows.append(r)
done = [r for r in rows if "score" in r]
print(f"{len(done)} of {len(rows)} runs tested")
print(f"{'run':6s} {'seed':4s} {'score':>5s} {'no push':>7s} {'medium':>6s} {'train':>5s} {'recent':>6s} {'speed':>6s} {'walk/30':>7s}  options on")
for r in sorted(done, key=lambda x: -x["score"]):
    print(f"{r['run']:6s} {r['seed']:<4d} {r['score']:5.1f} {r['V2_no_push']:7.0f} {r['V2_medium']:6.0f} {r['V2_train']:5.0f} {r['recent'] if r['recent'] is not None else float('nan'):6.1f} {r['V2_medium_speed'] or 0:6.3f} {r['V2_medium_walk']:7d}  {' '.join(r['on']) or 'baseline'}")
if len({r["run"] for r in done}) >= 4:
    print("\nMain effects, points of score, positive means the option helps:")
    out = []
    for n in names:
        on = [r["score"] for r in done if n in r["on"]]
        off = [r["score"] for r in done if n not in r["on"]]
        if on and off:
            sp = [r["V2_medium_speed"] or 0 for r in done if n in r["on"]], [r["V2_medium_speed"] or 0 for r in done if n not in r["on"]]
            out.append((np.mean(on) - np.mean(off), n, len(on), len(off), np.mean(sp[0]) - np.mean(sp[1])))
    for e, n, a, b, s in sorted(out, reverse=True):
        print(f"  {e:+6.1f}  {n}  {labels[n]:38s} on {a}, off {b}; speed {s:+.3f} m/s")
    if len(done) == len(rows):
        diffs = [next(r for r in done if r["run"] == d["run"] and r["seed"] == 3)["score"] - next(r for r in done if r["run"] == d["run"] and r["seed"] == 4)["score"] for d in D["design"]]
        print(f"  Noise guide: seed-to-seed standard deviation {np.std(diffs) / np.sqrt(2):.1f} points per run, standard error of an effect about {np.std(diffs) / np.sqrt(2) / np.sqrt(8):.1f}")
