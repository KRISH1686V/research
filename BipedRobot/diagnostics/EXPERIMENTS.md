# Experiments: plan, queue and rules

Started 11 Nov 2026. Live status of every run: `EXPERIMENTS_STATUS.md`, rewritten by
`python audit_2026-10-31/claude_audit/v2/exp_status.py` (read only, safe any time).
Predictions are written in `WORKING_RULES.md` before results; results go in `PROGRESS.md`.

## What the runs are for

| Group | Question | Runs | Seeds |
|---|---|---|---|
| Robot policy | The walk on the robot now | r11v s14 | 14 |
| Round 16 (robot) | Does training with the robot's measured gravity-sensor offset (up to 8 deg) and weak servos (70-100%) survive today's conditions better than r11v s14? | r16m | 11-14 |
| Round 17 (paper) | Can the budgets control the walk from loose to tight, with the robot staying up at the tightest setting? Four ways to fix the tight end | r17j, r17w, r17n, r17h | 11, 12 |
| Comparisons c17 (paper) | Same robot, command, costs and reward as round 17; only the cost handling differs | c17plain, c17lagmid, c17pen03, ppolong7mc17plain | 11-15 |

Round 17 variants (all measure shake on the joint angles, `jshake`):

| Run | Change | Config |
|---|---|---|
| r17j | soft overdraft: over budget scales the task reward by exp(-excess / capacity) instead of zeroing it | `config/v2_deploy_budget_alpha30_r17j_fit.yaml` |
| r17w | r17j + `token_budget.u_low: -0.3`: training also draws budgets tighter than u 0 | `..._r17w_fit.yaml` |
| r17n | r17j + `sac_target_entropy: -24`: less exploration noise | `..._r17n_fit.yaml` |
| r17h | the old hard overdraft: the comparison that says whether soft overdraft matters | `..._r17h_fit.yaml` |

Comparisons (budgets off unless named; `lagrange.normalize` pays lambda x cost / limit so joint shake counts):

| Run | Method | Config |
|---|---|---|
| c17plain | budgets off: SAC + command relabeling | `..._c17plain_fit.yaml` |
| c17lagmid | Lagrangian dual ascent at round 17's u 0.5 limits [0.0070711, 0.02, 6.1237e-5] | `..._c17lagmid_fit.yaml` |
| c17pen03 | fixed penalty, 0.3 per unit of each u 0.5 limit | `..._c17pen03_fit.yaml` |
| ppolong7mc17plain | PPO, budgets off, CPU only | `..._c17plain_fit.yaml` |

Planned, not queued (they depend on the round 17 result):
- the chosen round 17 variant, seeds 13-15
- episode budget with the chosen variant's settings, seeds 11-15
- Lagrangian at the u 0.25 and u 0.75 limits; fixed penalty 0.1 and 1.0; seeds 11-15 each

## Rules

- At most 4 training runs at once on the workstation (the owner's limit). CUDA only through the MPS server.
- gpu-13 (RTX 4090, rpuser2@192.168.29.76): in use from 12 Nov 08:53 (owner: "you can use it now"), at most 3 runs, because another user's 12.3 GB model server shares the card. Its queue is queue_g13.txt (copy on gpu-13 runs with run_queue.sh); pull_g13.sh brings finished runs back for scoring here.
- Every seed counts and is reported. No paper claim on fewer than 5 seeds.
- A round 17 variant succeeds on a seed when flight, slip and jshake each fall from u 1 to u 0 in the budget sweep with the robot up at least 90% at every setting (WORKING_RULES 11 Nov 11:00 and 11:40). The variant for the 5-seed comparisons is picked on all its seeds. If r17h does as well as r17j, soft overdraft is not claimed.
- Comparisons answer one question each and change one thing from round 17.

## How the queue works

| Step | Script (in `audit_2026-10-31/claude_audit/v2/`) |
|---|---|
| Round 16 | `lane_w16.txt` via `run_lane.sh`, started 10:15 |
| Round 17 | scratchpad `start_r17.sh`: waits for round 16, runs the GPU-only tests through MPS with nothing else training, starts `lane_w17.txt` (seed 11 of all four, then seed 12) only if they pass. Scratchpad `wait_r17all.sh` then runs the budget sweep, shake test and hwtest on each |
| Comparisons | `run_queue.sh queue_c17.txt 4`, opens when round 17's lane ends. Keeps 4 runs going in total; edit `queue_c17.txt` to add, remove or reorder; started lines move to `queue_c17.txt.started` |
| Scoring | `check_matrix2.sh`, `auto_inspect.sh`, `lift_after.sh`, `stress_after2.sh` wait for each SAC run's final checkpoint, output in `goal7m/` |

## Code options added 10-11 Nov (all off unless a config names them)

| Option | File | Test |
|---|---|---|
| cost `shake` (action second difference) | `fast/budget_env.py` | `tests/test_budget_shake.py` |
| cost `jshake` (joint-angle second difference) | `fast/budget_env.py` | `tests/test_budget_soft.py` |
| `token_budget.overdraft: soft` | `fast/budget_env.py`, `fast/budget_replay.py` | `tests/test_budget_soft.py` |
| `token_budget.u_low` | `fast/budget_env.py`, `fast/budget_replay.py`, `fast/train.py` | `tests/test_budget_soft.py` |
| `sac_target_entropy` | `fast/learner.py`, `fast/train.py` | `tests/test_budget_soft.py` |
| `lagrange.normalize` | `fast/budget_replay.py` | `tests/test_comparison_methods.py` |
