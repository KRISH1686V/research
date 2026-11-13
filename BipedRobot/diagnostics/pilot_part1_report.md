# Pilot Part 1 report

Date: 2026-10-29

## Status

**BLOCKED; no pilot was launched.** The required walking-reference ratio gate
cannot be evaluated because the available scripted reference is not a walking
rollout: it falls before 500 steps and has negative mean shaped reward at all
three commands. A ratio to a non-positive reference is not meaningful.

The old recovery processes are stopped. Their latest preserved checkpoints are
`recovery_m0_seed0_v2/seed_0/sb3_her_model_1100000_steps.zip` and
`recovery_m0_seed1_v2/seed_1/sb3_her_model_1100000_steps.zip`.

## Offline reward probes

Each value is mean shaped reward per step from a deterministic 500-step probe
or until termination. `REF` is the existing open-loop sinusoidal reference.
`N/A` ratios are intentional: the reference reward is non-positive and the
reference did not complete the episode.

| Config | vx | Standing | REF | Hopping | Dragging | In-place | Vibration | Leaning | Sliding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.2 | 0.4166 | -0.0392 | -0.7843 | 0.3009 | -0.3341 | 0.4148 | 0.4157 | 0.4064 |
| A | 0.4 | 0.0334 | -0.5634 | -0.7408 | -0.1339 | -0.4728 | 0.0311 | 0.0402 | 0.0303 |
| A | 0.6 | 0.0070 | -0.7092 | -0.3963 | -0.4397 | -0.5017 | -0.3801 | 0.0090 | 0.0050 |
| A-min | 0.2 | 0.4164 | 0.0011 | -0.6586 | 0.2735 | -0.3208 | 0.4209 | 0.4154 | 0.4061 |
| A-min | 0.4 | 0.0337 | -0.4660 | -0.6473 | -0.1433 | -0.4351 | 0.0362 | 0.0402 | 0.0308 |
| A-min | 0.6 | 0.0093 | -0.6240 | -0.3281 | -0.4038 | -0.4607 | -0.3167 | 0.0104 | 0.0074 |

The requested exploit ratio is `exploit_reward / REF_reward`. It is **N/A for
every row**, because `REF_reward <= 0`; therefore the `< 0.85` requirement is
not passed. Positive standing, vibration, leaning, and sliding rewards at
`vx=0.2` are direct evidence that the reward still admits non-walking
behaviors.

## Config and launch checks

- Created `config/pilot_gait_min.yaml` as A-min.
- A-min removes action-rate (`0.05`), base-motion (`0.05`), and clearance
  (`0.10`) terms from A.
- A and A-min both retain action filtering, reset velocity, `n_sampled_goal=2`,
  `p_zero_cmd=0.20`, and minimum nonzero `vx=0.20`.
- The trainer now refuses pilot configs missing those required fields.
- HER consistency tests: **2 passed**, including 1,000-transition checks and
  real replay-buffer sampling with flags on and off.
- Throughput: A **32.80 steps/s**, A-min expected to be comparable to the
  measured smoke range; the validated A smoke estimate for 2M steps is
  approximately **16.95 hours**. B smoke measured **31.78 steps/s** but is
  superseded and will not be launched.

## Decision

Do not launch `pilot_gait_smooth_a_seed0_v1` or
`pilot_gait_min_seed0_v1`. A valid forward-walking reference or an evidence-
based reward calibration is required first. The raw probe data is in
`paper_results/part1_exploit_check.json`.
