"""Does the slope of the value in the bucket balance equal the Lagrange multiplier? Exact answer on a small problem, before reading prices from a trained critic (E-D).

Each step the agent walks gently (reward r_low, cost 0) or fast (reward r_high, cost c). The bucket refills at rate r up to B, and an overdraft loses the step's reward.
The Lagrangian answer is known in closed form: at a limit r < c the multiplier is lambda = (r_high - r_low) / c, whatever the limit.
Value iteration on the balance gives V(b), and the claim to test is that V rises by lambda per token. V(b) is a staircase with steps every r, because tokens arrive on a schedule, so
the slope is measured across one whole action's cost: (V(b + c / 2) - V(b - c / 2)) / c. A local derivative would read the staircase's edges instead.

Usage: price_toy.py [OUT_JSON]
"""
import json
import sys

import numpy as np

R_LOW, R_HIGH, COST, GAMMA = 0.3, 1.0, 1.0, 0.99


def solve(rate, burst_steps=100, grid=4001, gamma=GAMMA):
    """Value iteration on balance b in [-B, B]; a balance below zero earns no reward. Linear interpolation between grid points."""
    cap = rate * burst_steps
    b = np.linspace(-cap, cap, grid)
    v = np.zeros(grid)
    actions = ((R_LOW, 0.0), (R_HIGH, COST))
    for _ in range(100_000):
        qs = []
        for reward, cost in actions:
            nxt = np.minimum(cap, b + rate - cost)
            qs.append(np.where(nxt >= 0.0, reward, 0.0) + gamma * np.interp(nxt, b, v))
        new = np.maximum(*qs)
        delta = float(np.abs(new - v).max())
        v = new
        if delta < 1e-9:
            break
    half = COST / 2
    slope = (np.interp(np.minimum(b + half, cap), b, v) - np.interp(np.maximum(b - half, -cap), b, v)) / (np.minimum(b + half, cap) - np.maximum(b - half, -cap))
    return b, v, slope, np.argmax(qs, axis=0)


def main():
    out = {}
    lam = (R_HIGH - R_LOW) / COST
    print(f"Lagrange multiplier in closed form: {lam:.3f} at every limit below {COST}")
    print(f"{'rate':>6}{'avg reward':>12}{'slope at 25% full':>19}{'50%':>8}{'75%':>8}{'100%':>8}{'best estimate error':>21}")
    for rate in (0.05, 0.1, 0.2, 0.4, 0.8):
        b, v, slope, _ = solve(rate)
        cap = b[-1]
        at = {f: float(slope[np.searchsorted(b, f * cap)]) for f in (0.25, 0.5, 0.75)}
        at[1.0] = float(slope[-1])
        # V at a full bucket, against the closed-form average reward of the constrained problem
        avg = (1 - GAMMA) * float(v[-1])
        out[rate] = {"lambda": lam, "slope": at, "avg_reward": avg, "closed_form_avg_reward": R_LOW + (R_HIGH - R_LOW) * rate / COST}
        err = abs(at[0.25] - lam) / lam
        print(f"{rate:>6.2f}{avg:>12.3f}{at[0.25]:>19.3f}{at[0.5]:>8.3f}{at[0.75]:>8.3f}{at[1.0]:>8.3f}{100 * err:>20.1f}%")
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
