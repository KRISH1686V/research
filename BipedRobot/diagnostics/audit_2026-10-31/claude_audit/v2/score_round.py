"""Scores a run against the targets written in diagnostics/WORKING_RULES.md on 8 Nov (14:12, 14:14, 14:38 and the review plan), at one command. Lines with no data print ND (not determined) and do not count as passed.
Usage: score_round.py COMMAND ARM:SEED ... ; the optional env var STRESS_TAG picks the stress file's tag (default stress_all). Written 8 Nov 2026."""
import json, os, sys
from pathlib import Path
import yaml
V = Path(__file__).resolve().parent; G = V / "goal7m"; R = V.parents[3]
TAG = os.environ.get("STRESS_TAG", "stress_all")

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def score(arm, seed, cmd):
    A = f"deploylong7m{arm}"
    N = f"fast_{A}_20261102_seed{seed}"
    rc = yaml.safe_load(open(R / f"runs/{N}/seed_{seed}/resolved_config.yaml"))
    e = rc.get("resolved_env_config", rc)
    clock = 2 * float((e.get("validity") or rc.get("validity") or {}).get("clock_hz", 1.0))
    m = load(G / f"matrix/{A}_s{seed}_v{cmd}_d1.0_h0.json"); r = m["summary"] if m else None
    an = load(G / f"anomaly/{A}_s{seed}.json"); a = an["summary"] if an else None
    li = load(G / f"anomaly/lift_{A}_s{seed}.json"); l = li[0] if li else None
    st = load(G / f"export/{A}_s{seed}_{TAG}.json"); s = st["summary"] if st else None
    three = clock >= 2.9
    def g(d, k): return None if d is None else d.get(k)
    rows = []
    def add(name, val, ok): rows.append((name, val, None if val is None else bool(ok(val))))
    add("up", g(r, "survival"), lambda x: x >= 0.9)
    add("speed", g(r, "straight_m_s"), lambda x: x >= 0.085)
    sl = g(r, "step_length_m"); add("step_mm", None if sl is None else 1000 * sl, lambda x: x >= (28 if three else 40))
    add("steps_s", g(r, "real_steps_per_s"), lambda x: abs(x - clock) <= 0.15 * clock)
    add("lift_mm", g(r, "swing_lift_mm_median"), lambda x: x >= (14 if three else 20))
    add("sole_mid", g(r, "whole_sole_mid_share"), lambda x: x >= 0.8)
    add("taps", g(r, "taps_per_s_m005"), lambda x: x <= 0.5)
    add("flight", g(r, "flight_share"), lambda x: x <= 0.01)
    t = g(r, "turn_abs_rad_s"); add("turn_10s", None if t is None else 10 * t, lambda x: x <= 0.2)
    add("sideways", g(r, "sideways_abs_m"), lambda x: x <= 0.15)
    add("pitch", g(r, "pitch_mean_deg"), lambda x: abs(x) <= 3)
    for k in ("sym_stance", "sym_swing", "sym_lift", "sym_stride"):
        add(k, g(r, k), lambda x: x <= 0.10)
    add("timing_cv", g(r, "step_interval_cv"), lambda x: x <= 0.10)
    add("alternation", g(r, "alternation"), lambda x: x >= 0.95)
    add("clock_lock", g(r, "clock_lock"), lambda x: x >= 0.90)
    n_gait = len(rows)
    add("rock", g(a, "roll_ptp_deg"), lambda x: x <= 12)
    add("hip_yaw", g(a, "hip_yaw_ptp_deg"), lambda x: x <= 8)
    add("over80", g(a, "speed_over_80_share"), lambda x: x <= 0.02)
    add("foot_gap", g(a, "step_width_mm"), lambda x: 80 <= x <= 120)
    add("gap_p01", g(a, "gap_p01_mm"), lambda x: x >= 80)
    add("gap_p99", g(a, "gap_p99_mm"), lambda x: x <= 120)
    add("tilt_p99", g(a, "tilt_p99_deg"), lambda x: x <= 10)
    add("tilt_max", g(a, "tilt_max_deg"), lambda x: x <= 20)
    add("leg_short", g(l, "leg_shorten_mm"), lambda x: x >= 12)
    add("hip_share", g(l, "hip_share"), lambda x: x <= 0.5)
    add("knee_extra", g(l, "knee_extra_deg"), lambda x: x >= 15)
    n_insp = len(rows) - n_gait
    add("worst_up", g(s, "survival"), lambda x: x >= 0.9)
    add("worst_pitch", g(s, "pitch_mean_deg"), lambda x: abs(x) <= 5)
    return rows, n_gait, n_insp

if __name__ == "__main__":
    cmd = sys.argv[1]
    for spec in sys.argv[2:]:
        arm, seed = spec.split(":")
        rows, ng, ni = score(arm, int(seed), cmd)
        gait, insp, worst = rows[:ng], rows[ng:ng + ni], rows[ng + ni:]
        n = lambda part: (sum(1 for _, _, o in part if o), sum(1 for _, v, _ in part if v is None), len(part))
        ga, ins, wo = n(gait), n(insp), n(worst)
        print(f"{arm} s{seed} @{cmd}: gait {ga[0]}/{ga[2]}" + (f" (ND {ga[1]})" if ga[1] else "") + f" | inspection {ins[0]}/{ins[2]}" + (f" (ND {ins[1]})" if ins[1] else "") + f" | worst case {wo[0]}/{wo[2]}" + (f" (ND {wo[1]})" if wo[1] else ""))
        print("   misses: " + ", ".join(f"{k} {v:.3g}" for k, v, o in rows if o is False))
        print("   values: " + ", ".join(f"{k} {'ND' if v is None else format(v, '.3g')}" for k, v, o in rows if k in ("speed", "step_mm", "lift_mm", "steps_s", "rock", "foot_gap", "gap_p01", "gap_p99", "tilt_p99", "tilt_max", "leg_short", "knee_extra", "hip_share", "worst_up")))
