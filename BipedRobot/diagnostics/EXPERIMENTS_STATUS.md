# Experiment status, 13 Nov 2026 12:14

Written by `exp_status.py` (read only). Plan and rules: EXPERIMENTS.md.

## Robot policy in use

| Run | What it tests | s14 |
|---|---|---|
| r11v | r11v s14: on the robot since 10 Nov | scored |

## Round 16: robot conditions

| Run | What it tests | s11 | s12 | s13 | s14 |
|---|---|---|---|---|---|
| r16m | r11v + gravity-sensor offset up to 8 deg + servos 70-100%. s14 exported for the robot 11 Nov | scored | scored | scored | scored |

## Hardware policy

| Run | What it tests | s11 | s12 | s13 |
|---|---|---|---|---|
| r18hw | r17wS + servo fitted to the robot + harsh conditions (for the robot video) | queued (queue_c17.txt) | queued (queue_c17.txt) | queued (queue_c17.txt) |

## Second robot: Robotis OP3 (12 leg DOF)

| Run | What it tests | s11 | s12 | s13 | s14 | s15 |
|---|---|---|---|---|---|---|
| op3w | budget-HER, r17wS settings, no gait reference | training 6.2M of 7.2M | not started | not started | not started | not started |
| op3off | budgets off | not started | not started | not started | not started | not started |

## Round 17: fixing the tight end of the budgets

| Run | What it tests | s11 | s12 | s13 | s14 | s15 |
|---|---|---|---|---|---|---|
| r17j | soft overdraft + joint shake | scored | scored |  |  |  |
| r17w | r17j + training draws budgets tighter than u 0 | scored | scored |  |  |  |
| r17n | r17j + less exploration noise | scored | scored |  |  |  |
| r17h | joint shake, old hard overdraft (comparison for soft) | scored | scored |  |  |  |
| r17w | THE METHOD (picked 12 Nov, 2 of 2 seeds): seeds 13-15 |  |  | scored | scored | scored |
| r17wh | ablation: r17w with hard overdraft (isolates u_low) | scored | scored |  |  |  |
| r17wS | r17w + no cost on the first 5 steps (start-of-episode flight artifact) | scored | scored | scored | scored | scored |
| r17wNR | ablation: r17w with budgets never relabeled (Saute-style) | scored | scored |  |  |  |
| r17wSF45 | r17wS + flight capacity floor 0.45 | scored | scored |  |  |  |
| r17wSC | r17wS + 30% single-cost tight training budgets | training 7.1M of 7.2M | training 7.1M of 7.2M |  |  |  |

## Comparisons c17 (same costs as round 17)

| Run | What it tests | s11 | s12 | s13 | s14 | s15 |
|---|---|---|---|---|---|---|
| c17plain | budgets off (plain SAC + command relabeling) | scored | scored | scored | scored | scored |
| c17lagmid | Lagrangian (dual ascent) at the u 0.5 limits | scored | scored | scored | scored | scored |
| c17pen03 | fixed penalty 0.3 per unit of each u 0.5 limit | scored | scored | scored | scored | scored |
| ppolong7mc17plain | PPO, budgets off (CPU) | trained, scoring | trained, scoring | trained, scoring | trained, scoring | trained, scoring |
| c17ep | episode budget with r17w settings | scored | scored | scored | scored | scored |
| c17lagtight | Lagrangian at the u 0.25 limits | scored | queued (queue_g13.txt) | queued (queue_g13.txt) | queued (queue_g13.txt) | queued (queue_g13.txt) |
| c17lagmin | Lagrangian at the u 0 limits | training 6.4M of 7.2M | queued (queue_g13.txt) | queued (queue_g13.txt) | queued (queue_g13.txt) | queued (queue_g13.txt) |
| c17pen01 | fixed penalty 0.1 | scored | scored | scored | scored | scored |
| c17pen10 | fixed penalty 1.0 | queued (queue_g13.txt) | queued (queue_g13.txt) | queued (queue_g13.txt) | queued (queue_g13.txt) | queued (queue_g13.txt) |

## Planned, not queued yet

