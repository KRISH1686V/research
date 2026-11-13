"""Stage 1 screening: 12 on/off factors in 16 runs, a two-level orthogonal array. Each factor is on in exactly half of the runs and balanced against
every other factor, so main effects are estimated from all the runs, at the cost of confounding with two-factor interactions (resolved in stage 2).
Run 1 has every factor off, which is the plain V2 baseline. Writes configs under config/doe/ and the run list.
"""
import itertools, json, re
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "config" / "doe"
FACTORS = [("H", "history 5 frames"), ("R", "difficulty ramp over 40%"), ("L", "critic layer norm"), ("Q", "averaged twin Q"), ("E", "averaged policy weights"),
           ("A", "asymmetric critic"), ("M", "mirror augmentation"), ("N", "3-step returns"), ("T", "sigma_v 0.075"), ("W", "torque penalty 0.2"),
           ("G", "gait shaping off"), ("F", "smoothing: action rate 0.3, filter 0.3")]
FLAGS = {"R": ["--difficulty_ramp", "0.4"], "L": ["--layer_norm"], "Q": ["--avg_q"], "E": ["--ema_tau", "0.002"], "A": ["--asym"], "M": ["--mirror"], "N": ["--n_step", "3"]}


def array16():
    base = np.array(list(itertools.product([-1, 1], repeat=4)))
    cols = []
    for r in range(1, 5):
        for c in itertools.combinations(range(4), r):
            cols.append(base[:, list(c)].prod(axis=1))
    # Use the 4 base factors first, then the highest-order products, which alias least with pairs of the base factors.
    order = [0, 1, 2, 3, 14, 13, 12, 11, 10, 9, 8, 7]
    x = np.array(cols).T[:, order]
    x = x * x[0]  # first row all +1, then negated below so the baseline is all off
    return -x


def main():
    x = array16()
    assert (x[0] == -1).all() and (x.sum(axis=0) == 0).all(), "balanced, with the baseline first"
    assert (x.T @ x == 16 * np.eye(12)).all(), "orthogonal"
    OUT.mkdir(exist_ok=True)
    base = yaml.safe_load(open(REPO / "config/v2_shaped.yaml"))
    design, lines = [], []
    names = [f[0] for f in FACTORS]
    for i, row in enumerate(x, 1):
        on = {n for n, v in zip(names, row) if v == 1}
        name = f"doe{i:02d}"
        c = json.loads(json.dumps(base))
        if "H" in on:
            c["validity"]["history"] = 5
        if "T" in on:
            c["reward"]["sigma_v"] = 0.075
        if "W" in on:
            c["shaping"]["torque_penalty_weight"] = 0.2
        if "G" in on:
            c["shaping"]["gait_shaping"] = False
        if "F" in on:
            c["shaping"]["action_rate_weight"] = 0.3
            c["shaping"]["action_filter_alpha"] = 0.3
        path = OUT / f"{name}.yaml"
        path.write_text(f"# Stage 1 screening run {i}. On: {' '.join(sorted(on)) or 'nothing (baseline)'}. See audit/v2/doe.py.\n" + yaml.safe_dump(c, sort_keys=False))
        flags = [t for n in names if n in on for t in FLAGS.get(n, [])]
        design.append({"run": name, "on": sorted(on), "flags": flags})
        for seed in (3, 4):
            lines.append(f"run_doe_arm.sh {name} config/doe/{name}.yaml {seed} {' '.join(flags)}".strip())
    (Path(__file__).parent / "doe_design.json").write_text(json.dumps({"factors": FACTORS, "design": design}, indent=1))
    # Both seeds of a run are spread apart, so an early stop still leaves a balanced first replicate.
    lines = lines[::2] + lines[1::2]
    (Path(__file__).parent / "doe_runs.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(f"{d['run']}: {' '.join(d['on']) or 'baseline'}" for d in design))


if __name__ == "__main__":
    main()
