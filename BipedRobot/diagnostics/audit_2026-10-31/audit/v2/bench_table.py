"""Print a comparison table of v2 benchmark results (all bench/<label>/*.json)."""
import glob, json, sys
from pathlib import Path
B = Path(__file__).resolve().parent / "bench"
COLS = [("speed_x_disp_steady", "v_disp", 3), ("rmse_vx_achieved_goal", "rmse*", 3), ("flight_frac", "flight", 3),
        ("double_support_frac", "DS", 3), ("lr_load_corr", "LRcorr", 2), ("swing_clearance_mm_median", "clr_mm", 1),
        ("stance_slip_mps_mean", "slip", 3), ("slip_m_per_m_travel", "slip/m", 2), ("torque_sat_frac", "tsat", 2),
        ("cost_of_transport", "CoT", 1), ("lateral_drift_m", "drift", 2)]
def f(x, d): return "  NA " if x is None else f"{x:.{d}f}"
rows = sorted(glob.glob(str(B / "*" / "*.json")))
print(f"{'policy':12s} {'cond':16s} {'surv':>9s} {'wilson95':>11s} " + " ".join(f"{c[1]:>7s}" for c in COLS) + "  gait classes")
for p in rows:
    r = json.load(open(p)); a = r["aggregate"]; s = r["settings"]
    cond = f"{s['realism']} {r['command'][0]:.1f}{' push' if s['pushes'] else ''}"
    w = a["survival_wilson95"]
    vals = " ".join(f"{f(a['all_episodes'][c[0]]['mean'], c[2]):>7s}" for c in COLS)
    print(f"{Path(p).parent.name:12s} {cond:16s} {a['survival']:>3d}/{a['n']:<3d}  [{w[0]:.2f},{w[1]:.2f}] {vals}  {a['gait_classes']}")
print("\nAll-episode means (falls included); *rmse uses the achieved-goal window, kept only for continuity with v1.")
