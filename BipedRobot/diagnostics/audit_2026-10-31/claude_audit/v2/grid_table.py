"""UTD grid table: speed from each run's summary and progress logs, survival from evaluation_suite_v2 (easy 0.2, 30 seeds)."""
import csv, json, sys
from pathlib import Path

V = Path(__file__).resolve().parent
RUNS = V.parents[3] / "runs"
SEED = sys.argv[1] if len(sys.argv) > 1 else "4"
TARGET = 25  # worst SB3 seed at easy 0.2


def wall_at(rows, steps):
    for r in rows:
        if int(r["steps"]) >= steps:
            return float(r["wall_s"]) / 60
    return None


def cell(e, b, g):
    name = f"fast_grid2_e{e}_b{b}_g{g}_seed{SEED}"
    if (e, b, g) == (8, 256, 1):
        run = RUNS / f"fast_min_regress_20261101_seed{SEED}" / f"seed_{SEED}"
    else:
        run = RUNS / name / f"seed_{SEED}"
    if not (run / "summary.json").exists():
        return None
    s = json.load(open(run / "summary.json"))
    rows = list(csv.DictReader(open(run / "progress.csv")))
    out = {"e": e, "b": b, "g": g, "sps": s["env_steps_per_s_post_warmup"], "ups": s["updates_per_s_post_warmup"], "steps": s["steps"], "wall": s["wall_s"] / 60}
    for k, steps in (("300000", 300000), ("500000", 500000), ("final", s["steps"])):
        f = V / "grid" / f"{name}_{k}.json"
        if (e, b, g) == (8, 256, 1):
            f = V / "bench" / f"Fast-min_s{SEED}" / "easy_0.2-.json" if k == "final" else V / "grid" / f"fast_grid_e8_b256_g1_seed{SEED}_{k}.json"
        if f.exists():
            a = json.load(open(f))["aggregate"]
            out[k] = (a["survival"], a["all_episodes"]["speed_x_disp_steady"]["mean"], a["gait_classes"].get("alternating_walk", 0), wall_at(rows, steps) or s["wall_s"] / 60)
    hit = [out[k][3] for k in ("300000", "500000", "final") if k in out and out[k][0] >= TARGET]
    out["to_target"] = min(hit) if hit else None
    return out


rows = [c for c in (cell(e, b, g) for g in (1, 0.5, 0.25) for b in (256, 1024, 4096) for e in (8, 16)) if c]
print(f"seed {SEED}; target: >= {TARGET}/30 at easy 0.2. Each survival cell: survived/30, speed m/s, minutes to that checkpoint")
print(f"{'envs':>4s} {'batch':>5s} {'upd/step':>8s} {'steps/s':>7s} {'upd/s':>6s} {'steps':>8s} {'min':>5s} | {'300k':>20s} | {'500k':>20s} | {'final':>20s} | min to target")
fmt = lambda x: f"{x[0]:2d}/30 {x[1] if x[1] is not None else float('nan'):.3f} {x[3]:5.0f}m" if x else ""
for r in rows:
    tt = f"{r['to_target']:.0f}" if r["to_target"] else "not reached"
    print(f"{r['e']:4d} {r['b']:5d} {r['g']:8.2f} {r['sps']:7.0f} {r['ups']:6.0f} {r['steps']:8d} {r['wall']:5.0f} | {fmt(r.get('300000')):>20s} | {fmt(r.get('500000')):>20s} | {fmt(r.get('final')):>20s} | {tt}")
