"""Summarise the OP3 cost measurements (scripted_walk.py and random_costs.py outputs in this folder) into the numbers behind token_budget's rates.
Usage: python diagnostics/op3/summarize_costs.py"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ("flight", "slip", "jshake")


def walks(path: Path) -> list:
    z = json.loads(path.read_text())
    rows = []
    for r in z["results"]:
        p = r["params"]
        for x in r["runs"]:
            rows.append(dict(T=p["period_s"], cmd=p["speed"], lift=p["lift"], fell=x["fell"], vx=x["mean_vx"] or 0.0, mean=np.array(x["cost_mean"]),
                             p90=np.array(x["cost_p90_window100"])))
    return rows


def main() -> None:
    for f in sorted(HERE.glob("walk_*.json")):
        rows = walks(f)
        up = [r for r in rows if not r["fell"]]
        print(f"\n{f.name}: {len(up)} of {len(rows)} scripted walks stayed up 10 s")
        if len(up) < 10:
            up = rows
            print("  (few stayed up: statistics over all runs, up to their fall)")
        m = np.array([r["mean"] for r in up])
        w = np.array([r["p90"] for r in up])
        for k, n in enumerate(NAMES):
            print(f"  {n:7s} per-step mean over walks: min {m[:, k].min():.3g}  median {np.median(m[:, k]):.3g}  max {m[:, k].max():.3g}"
                  f"   | 90th pct of 100-step window means: max {w[:, k].max():.3g}")
        for cmd in sorted({r["cmd"] for r in up}):
            sel = np.array([r["mean"] for r in up if r["cmd"] == cmd])
            vx = np.mean([r["vx"] for r in up if r["cmd"] == cmd])
            print(f"   pattern speed {cmd:.2f} (walked {vx:.3f} m/s, n={len(sel)}): min " + ", ".join(f"{n} {sel[:, k].min():.3g}" for k, n in enumerate(NAMES))
                  + " | median " + ", ".join(f"{n} {np.median(sel[:, k]):.3g}" for k, n in enumerate(NAMES)))
    rc = HERE / "random_costs.json"
    if rc.exists():
        print("\nstanding and uniform random actions (3000 steps each):")
        for k, v in json.loads(rc.read_text())["results"].items():
            print(f"  {k:18s} " + ", ".join(f"{n} {c:.3g}" for n, c in zip(NAMES, v["cost_mean"])) + f"   falls {v['falls_per_3000_steps']}")


if __name__ == "__main__":
    main()
