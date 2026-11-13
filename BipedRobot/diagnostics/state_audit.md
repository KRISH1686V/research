# Phase 0 state audit

Audit date: 2026-10-28

## Classification

**NOT REPRODUCED — the prior Gate A1 claim is unusable.**

The prior TensorBoard claim (`eval/ep_len_02=500`,
`eval/rmse_vx=0.0874`) cannot be reproduced by a fresh standalone evaluation
with fixed seeds and a saved configuration. The prior checkpoint is not a
trustworthy M0 baseline.

## 1. Git state

- Repository: `/home/rpuser1/yasho/rl/BipedRobot`
- Branch: `stageA-fixes`
- HEAD: `5147251 stageA: add shared reward and stand pipeline`
- Parent comparison: `58e3b76 Fixed bugs and improved walking quality and sim2real fidelity`
- Working tree: heavily modified and not clean. `git status --short --branch`
  reports modifications across the existing environment, configuration,
  documentation, and submodule files, plus untracked `her/`, `tests/`,
  `trained_models/`, `checkpoints_final/`, and `config/final/`.
- No existing work was reverted during this audit.

## 2. Processes and hardware

- No `train_her_biped` or `src/mujoco/train.py` process was active at audit
  time.
- GPU: NVIDIA GeForce RTX 4080; 92 MiB / 16,376 MiB used; 0% utilization.
- CPU: 20 cores.
- Disk: 551 GB available on the root filesystem; 69% used.
- Load average: 12.31, 12.42, 12.48.
- Python used for validation: `/home/rpuser1/miniconda3/envs/rl/bin/python`.

## 3. Run and checkpoint inventory

The workspace contains the following principal runs:

| Run | Evidence |
|---|---|
| `stageA0_stand`, `stageA0_stand_safe` | standing gates |
| `stageA1_walk_safe` | earlier walking-safe baseline |
| `her_loco_1_5m` | earlier HER run |
| `sb3_sac_her_adaptive_future_k4` | adaptive HER trials |
| `her_goal_tracking_v2` | completed 1M-step tracking-only HER baseline |
| `author_walk_16worker`, `author_walk_16worker_v2` | non-HER author-style SAC runs |
| `author_walk_forward_focus` | focused non-HER forward run |
| `walkv2`, `walkv3`, `BipedV2_SAC`, `v4`, `v6` | older author project runs |

`her_goal_tracking_v2` contains `sb3_her_final.zip` and checkpoints through
1,000,000 steps. Its TensorBoard event file was last updated at the end of
the run and its checkpoint directory is approximately 77.56 MB.

## 4. TensorBoard metrics

### Current tracking-only HER baseline

Source: `runs/her_goal_tracking_v2/seed_42/SAC_1/`

| Metric | First | Last | Best |
|---|---:|---:|---:|
| `eval/ep_len_zero` | — | 500.0 at step 950,144 | 500.0 |
| `eval/ep_len_02` | — | 500.0 at step 950,144 | 500.0 |
| `eval/rmse_vx` | 0.2490 | 0.0874 at step 950,144 | 0.0874 |
| `eval/torque_sat` | 0.6304 | 0.4580 | 0.4580 |
| `curriculum/range_scale` | 1.0 | 1.749 at step 999,448 | 1.749 |
| `curriculum/success_rate` | 0.0 | 0.5667 | 0.8333 |
| `train/action_abs_gt_095` | 0.0729 | 0.0104 | 0.3229 |
| `train/tracking_reward` | 0.3527 | 0.8356 | 0.9234 |
| `rollout/ep_len_mean` | 36.25 | 455.22 | 457.26 |
| `rollout/ep_rew_mean` | 2.693 | 344.001 | 347.900 |

### Earlier A1 reference

Source: `runs/stageA1_walk_safe/seed_42/SAC_1/`

| Metric | Last |
|---|---:|
| `eval/ep_len_zero` | 500.0 at step 1,450,527 |
| `eval/ep_len_02` | 314.6 at step 1,450,527 |
| `eval/rmse_vx` | 0.0837 |
| `eval/torque_sat` | 0.0035 |
| `curriculum/range_scale` | 2.0114 |
| `curriculum/success_rate` | 0.4667 |

The HER baseline improves survival and tracking at 0.2 m/s but has much higher
torque saturation than the earlier `stageA1_walk_safe` result. That tradeoff
must be measured fairly in Phase 2.

## 5. Reports and code state

No prior `diagnostics/` report files were present in the current checkout.
The relevant implementation is:

- `envs/her_loco_env.py`: goal-conditioned velocity environment and reward.
- `train_her_biped.py`: SB3 SAC + HerReplayBuffer training entry point.
- `evaluate.py`: command-grid evaluation script.
- `config/her_loco_config.yaml`: actuator, goal, curriculum, and reward
  configuration.

The current HER environment uses a 3-element goal `[vx, vy, wz]`. The
budget-as-goal extension requested in the attached protocol has not yet been
implemented.

## 6. Tests

Focused Stage-A suite:

```text
9 passed, 1 skipped
```

Full `pytest -q` collection does not pass in this environment. Five
collection errors occurred:

- `envs/mujoco/test_mirror.py`: installed MuJoCo lacks `mujoco.viewer`.
- `envs/mujoco/test_mirror_actions.py`: stale import `envs.mujoco_env`.
- `src/mujoco/test_model.py`: installed MuJoCo lacks `mujoco.glfw`.
- `src/tools/isaaclab/test_ik.py`: Isaac Lab is not installed.
- `tests/test_env.py`: stale import `envs.mujoco_env`.

These are environment/import issues, not failures in the focused HER tests.

## 7. A1 decision

The current tracking-only HER baseline is **NOT REPRODUCED**. The fresh
10-seed standalone sweep of the final checkpoint had 3/10 survival and
RMSE 0.1363 m/s; the 800k checkpoint selected by the transparent
TensorBoard rule had 2/10 survival and RMSE 0.1488 m/s under the original
0.80-action frozen settings. The callback's own method returned low RMSE but
only 338 mean steps at 800k and 236.4 at final, using only five seeds.
No R0 configuration row achieved both the survival and RMSE gate.

The Stage-A values are now active for R2 in
`config/frozen_baseline.yaml`; the original R0 file is preserved as
`config/frozen_baseline_r0.yaml`.

## 9. Recovery status

R1 hardening is implemented in `train_her_biped.py` and
`evaluation_suite.py`. New run directories write the resolved YAML, commit,
binary-safe Git diff, imported-untracked list, command line, seed, pip freeze,
and evaluation hash before training starts. The callback and standalone
evaluator share the same episode function. The fixed-checkpoint regression
fixture is `paper_results/regression_fixed_checkpoint.json`.

R2 M0 seeds 0 and 1 are currently training under unique experiment names
`recovery_m0_seed0_v2` and `recovery_m0_seed1_v2`; seed 2 will start only
after a slot is available. R3 opt-in budget code is implemented and its
focused tests currently pass. Calibration and C1 remain prohibited until an
M0 final checkpoint passes the fresh gate.

## 7a. Required frozen-config re-evaluation

The Phase 2 safeguard created
`config/frozen_baseline.yaml` from the configuration consumed by
`train_her_biped.py`. The entry point flattens the YAML sections with
`build_env_config()`; the re-evaluation used that same construction path and
the unchanged checkpoint
`checkpoints/her_goal_tracking_v2/seed_42/sb3_her_final.zip`.

The evaluation used 10 deterministic-policy episodes per condition and
required all 500 control steps for survival:

| Realism | Command vx | Survival | Mean speed (m/s) | RMSE vx (m/s) | Torque saturation | Normalized torque RMS | Power |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 0.20 | 3/10 | 0.1090 | 0.1294 | 0.7514 | 0.8932 | 32.91 |
| easy | 0.35 | 0/10 | 0.2074 | 0.1932 | 0.7534 | 0.8980 | 41.81 |
| medium | 0.20 | 0/10 | 0.1523 | 0.1425 | 0.7750 | 0.9096 | 46.14 |
| medium | 0.35 | 0/10 | 0.2495 | 0.1878 | 0.8002 | 0.9226 | 49.78 |
| full | 0.20 | 0/10 | 0.1067 | 0.1677 | 0.7979 | 0.9135 | 46.83 |
| full | 0.35 | 0/10 | 0.3162 | 0.2010 | 0.8232 | 0.9250 | 52.21 |

Raw per-episode data is saved in
`paper_results/baseline_frozen_eval.json`. This does **not** reproduce the
earlier TensorBoard summary (`eval/ep_len_02=500` and `rmse_vx=0.0874`).
The active trainer does not save a run-specific YAML, so no independent
training configuration exists in the checkpoint directory to explain the
difference. The frozen-config A1 gate is therefore **not met** (3/10 rather
than at least 8/10 survival at 0.20 m/s), and the protocol requires stopping
before budget-goal implementation.

## 7b. Baseline setting reconciliation

The following method-independent settings were copied from the active
training YAML into `config/frozen_baseline.yaml`; current workspace values
match them. The checkpoint directory contains no run-specific configuration
override.

| Setting | Baseline training value | Current frozen value |
|---|---:|---:|
| frame skip / control rate | 4 / 50 Hz | 4 / 50 Hz |
| episode length | 500 steps | 500 steps |
| action range fraction | 0.80 | 0.80 |
| torque limit | 2.94 N m | 2.94 N m |
| delay / lag / backlash | enabled | enabled |
| torso height thresholds | 0.14 / 0.45 m | 0.14 / 0.45 m |
| upright cosine threshold | 0.58 | 0.58 |
| easy realism during training | easy | easy |
| domain randomization in easy | disabled by environment | disabled by environment |
| velocity goal ranges | vx [-0.2, 0.6], vy [-0.2, 0.2], wz [-0.5, 0.5] | same |
| curriculum initial ranges | vx [-0.1, 0.2], vy [-0.05, 0.05], wz [-0.15, 0.15] | same |
| curriculum expansion / window | 1.15 / 30 episodes | same |
| reward mode / sigmas | track / 0.15, 0.30 | same |
| fall penalty | 25.0 | 25.0 |
| HER strategy / sampled goals | future / 4 | future / 4 |
| parallel environments | 8 | 8 (trainer default for the run) |
| total training steps | 1,000,000 | 1,000,000 (checkpoint provenance) |

## 8. Next-phase plan and wall-clock estimate

After user confirmation:

1. Calibrate baseline windowed costs at the three required commands:
   approximately 30–60 minutes.
2. Extend the goal to velocity plus normalized safety budgets and add budget
   sampling/curriculum:
   approximately 2–4 hours including focused tests.
3. Implement and test budget-aware HER relabeling:
   approximately 2–4 hours.
4. Run C1 at the baseline step count using at most two concurrent runs:
   estimate from the observed HER throughput, approximately 8–24 hours.
5. Stop at Gate C if it fails; otherwise schedule the multi-method, multi-seed
   comparison:
   several days at the current single-GPU throughput.

This audit stops at the required Phase 0 gate. No budget-as-goal training was
started.
