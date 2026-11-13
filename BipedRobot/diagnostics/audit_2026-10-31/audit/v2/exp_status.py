"""Status of every registered run: queued, training (with steps), trained, scored. Writes BipedRobot/diagnostics/EXPERIMENTS_STATUS.md and prints it.
Usage: python exp_status.py. Read only; safe to run any time. Written 11 Nov 2026. Add new arms to GROUPS."""
import csv
import glob
import os
import subprocess
from datetime import datetime

R = "/home/rpuser1/yasho/rl/BipedRobot"
V = R + "/diagnostics/audit_2026-10-31/audit/v2"
M = V + "/goal7m"
TOTAL = 7_200_000

GROUPS = [
    ("Robot policy in use", "reference", [("deploylong7mr11v", [14], "r11v s14: on the robot since 10 Nov")]),
    ("Round 16: robot conditions", "robot", [("deploylong7mr16m", [11, 12, 13, 14], "r11v + gravity-sensor offset up to 8 deg + servos 70-100%. s14 exported for the robot 11 Nov")]),
    ("Hardware policy", "robot", [("deploylong7mr18hw", [11, 12, 13], "r17wS + servo fitted to the robot + harsh conditions (for the robot video)")]),
    ("Second robot: Robotis OP3 (12 leg DOF)", "paper", [("deploylong7mop3w", [11, 12, 13, 14, 15], "budget-HER, r17wS settings, no gait reference"), ("deploylong7mop3off", [11, 12, 13, 14, 15], "budgets off")]),
    ("Round 17: fixing the tight end of the budgets", "paper", [
        ("deploylong7mr17j", [11, 12], "soft overdraft + joint shake"),
        ("deploylong7mr17w", [11, 12], "r17j + training draws budgets tighter than u 0"),
        ("deploylong7mr17n", [11, 12], "r17j + less exploration noise"),
        ("deploylong7mr17h", [11, 12], "joint shake, old hard overdraft (comparison for soft)"),
        ("deploylong7mr17w", [13, 14, 15], "THE METHOD (picked 12 Nov, 2 of 2 seeds): seeds 13-15"),
        ("deploylong7mr17wh", [11, 12], "ablation: r17w with hard overdraft (isolates u_low)"),
        ("deploylong7mr17wS", [11, 12, 13, 14, 15], "r17w + no cost on the first 5 steps (start-of-episode flight artifact)"),
        ("deploylong7mr17wNR", [11, 12], "ablation: r17w with budgets never relabeled (Saute-style)"),
        ("deploylong7mr17wSF45", [11, 12], "r17wS + flight capacity floor 0.45"),
        ("deploylong7mr17wSC", [11, 12], "r17wS + 30% single-cost tight training budgets"),
    ]),
    ("Comparisons c17 (same costs as round 17)", "paper", [
        ("deploylong7mc17plain", [11, 12, 13, 14, 15], "budgets off (plain SAC + command relabeling)"),
        ("deploylong7mc17lagmid", [11, 12, 13, 14, 15], "Lagrangian (dual ascent) at the u 0.5 limits"),
        ("deploylong7mc17pen03", [11, 12, 13, 14, 15], "fixed penalty 0.3 per unit of each u 0.5 limit"),
        ("ppolong7mc17plain", [11, 12, 13, 14, 15], "PPO, budgets off (CPU)"),
        ("deploylong7mc17ep", [11, 12, 13, 14, 15], "episode budget with r17w settings"),
        ("deploylong7mc17lagtight", [11, 12, 13, 14, 15], "Lagrangian at the u 0.25 limits"),
        ("deploylong7mc17lagmin", [11, 12, 13, 14, 15], "Lagrangian at the u 0 limits"),
        ("deploylong7mc17pen01", [11, 12, 13, 14, 15], "fixed penalty 0.1"),
        ("deploylong7mc17pen10", [11, 12, 13, 14, 15], "fixed penalty 1.0"),
    ]),
]
PLANNED = [
]


def running():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        if ("fast/train.py" in line or "fast/train_ppo.py" in line) and "--exp_name" in line:
            names.add(line.split("--exp_name")[1].split()[0])
    return names


def queued():
    q = {}
    for f in glob.glob(V + "/queue_*.txt") + glob.glob(V + "/lane_w1[67].txt"):
        for line in open(f):
            p = line.split()
            if len(p) >= 3 and not line.lstrip().startswith("#"):
                q[(p[0], int(p[2]))] = os.path.basename(f)
    return q


def steps(arm, s):
    f = f"{R}/runs/fast_{arm}_20261102_seed{s}/seed_{s}/progress.csv"
    try:
        rows = list(csv.DictReader(open(f)))
        return int(float(rows[-1]["steps"])) if rows else 0
    except (OSError, KeyError, ValueError, IndexError):
        return None


def status(arm, s, run, q):
    name = f"fast_{arm}_20261102_seed{s}"
    d = f"{R}/checkpoints/{name}/seed_{s}"
    final = os.path.exists(d + "/sb3_her_final.pt") or os.path.exists(d + "/ppo_final.zip")
    scored = len(glob.glob(f"{M}/matrix/{arm}_s{s}_v*.json")) >= 12 and os.path.exists(f"{M}/anomaly/{arm}_s{s}.json")
    if name in run:
        n = steps(arm, s)
        return f"training {n / 1e6:.1f}M of 7.2M" if n else "training"
    if final:
        return "scored" if scored else "trained, scoring"
    n = steps(arm, s)
    if n and (arm, s) in q and q[(arm, s)] == "queue_g13.txt":
        return f"gpu-13 {n / 1e6:.1f}M of 7.2M"  # progress pulled by pull_g13.sh every 10 min
    if (arm, s) in q:
        return f"queued ({q[(arm, s)]})"
    if arm.startswith("deploylong7mr17") and subprocess.run(["pgrep", "-f", "scratchpad/start_r17.sh"], capture_output=True).returncode == 0:
        return "queued (after round 16)"
    return "not started"


def main():
    run, q = running(), queued()
    lines = [f"# Experiment status, {datetime.now():%d %b %Y %H:%M}", "", "Written by `exp_status.py` (read only). Plan and rules: EXPERIMENTS.md.", ""]
    for title, kind, arms in GROUPS:
        lines += [f"## {title}", "", "| Run | What it tests | " + " | ".join(f"s{s}" for s in sorted({s for _, ss, _ in arms for s in ss})) + " |"]
        seeds = sorted({s for _, ss, _ in arms for s in ss})
        lines += ["|---|---|" + "---|" * len(seeds)]
        for arm, ss, what in arms:
            cells = [status(arm, s, run, q) if s in ss else "" for s in seeds]
            lines.append(f"| {arm.replace('deploylong7m', '')} | {what} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Planned, not queued yet", ""] + [f"- {p}" for p in PLANNED] + [""]
    text = "\n".join(lines)
    open(R + "/diagnostics/EXPERIMENTS_STATUS.md", "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
