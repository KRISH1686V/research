# Budget-as-goal HER: working specification

Written 2 November 2026 because the original `train-once-plan` doc could not be found. It describes what the code in `BipedRobot/fast/` does. It was reconstructed from the Q1 roadmap's description of the method and the standard token bucket, so if the original doc turns up, compare the two and treat any difference as an open decision.

**Source of each part:** "Roadmap" means stated in the Q1 roadmap. "Standard" means an established result. "Choice" means a design decision made while building this, which can be changed.

## Goal

One training run learns to walk under any gait limit, and the limit is chosen after training by setting a number. (Roadmap)

The goal given to the policy is:

```
[vx, vy, wz, u_flight, u_slip, u_torque]
```

Each `u` is in [0, 1], where 0 is the tightest budget and 1 the loosest. It maps on a log scale to an allowed cost per control step (Choice):

```
rate = rate_min * (rate_max / rate_min) ^ u
```

| Cost | Measured as | rate_min | rate_max |
|---|---|---|---|
| Flight | Fraction of physics substeps with no foot and no other body loading the floor (force over 0.5 N) | 0.002 | 0.2 |
| Stance slip | Contact-point sliding speed of loaded feet, in m/s | 0.02 | 0.2 |
| Torque saturation | Fraction of actuators at the torque limit | 0.05 | 0.8 |

The three costs are named in the roadmap. Their measurement matches `evaluation_suite_v2.py`. The ranges are a choice, based on costs seen in earlier runs.

`config/v2_budget_window*.yaml`, from 2 November, use `rate_min [0.005, 0.02, 0.01]` and `rate_max [0.2, 0.2, 0.2]`, from costs measured on the first survival runs: walking used flight 0.004-0.02, slip 0.02-0.06, torque 0.01-0.03. The old torque range was not binding at `u = 0.5`.

## Token bucket

Each cost has a bucket that starts full, with capacity `B = rate * burst_steps` and `burst_steps = 100` (2 s, Choice):

```
b(t+1) = min(B, b(t) + rate - c(t))
```

There is no floor, so overdraft shows as `b < 0`. The policy sees `b / B` for each bucket, clipped to [-1, 1].

**Guarantee (Standard):** while a bucket stays non-negative, total cost over any window of T steps is at most `rate * T + B`.

**Bounded memory (Choice, `token_budget.window_steps`, off by default):** on the first survival runs, tight budgets stayed overdrawn for most of the episode, because one early overspend is never forgiven. With memory W the deficit looks back at most W steps:

```
F(k) = S(k) - rate * k
d(t) = F(t) - min[max(0, t - W) <= k <= t] F(k)
```

No overdraft now means the guarantee holds for every window up to W steps, not for longer ones. Steady cost can reach `rate + B / W` without overdraft, which is `2 * rate` at `W = burst_steps = 100`. An overdraft recovers within W steps of low cost, and a small overspend recovers sooner. `tests/test_budget_window.py` checks this against brute force. `survive_eval.py` also reports the plain bucket's verdict.

## Reward (Choice, after Saute RL)

```
fall:             -fall_penalty
otherwise:        alive_bonus + base   if no bucket is overdrawn
                  alive_bonus          if any bucket is overdrawn
base = tracking + progress + smoothing, plus gait terms if gait shaping is on
```

`alive_bonus = 0.2` was added to improve survival. It may also reward standing still, so speed must always be reported next to survival.

## Hindsight relabeling

For a stored transition at step t, the command comes from the achieved velocity at a random future step t' in the same episode. (Roadmap: command and budget are relabeled jointly.) For relabeled samples, each cost's budget is chosen independently (Choice):

| Probability | New budget |
|---|---|
| 0.25 | Keep the episode's budget |
| 0.50 | The mean cost actually spent from the start of the episode up to t' |
| 0.25 | A new draw from the prior |

Bucket levels are recomputed exactly for the new budget, in closed form, using the episode's cumulative cost S (Standard form of the recursion):

```
F(k) = S(k) - rate * k
d(t) = F(t) - min[k <= t] F(k)
b(t) = B - d(t)
```

`tests/test_budget_her.py` checks relabeled levels and rewards against an independent brute-force recursion.

## Budget sampling during training (Choice)

20% of episodes are unconstrained (all `u = 1`). The rest draw each `u` from `[1 - d, 1]`, where d is the difficulty, so budgets start loose and tighten as d rises to 1 over the first 40% of training.

## Mirror relabeling (Roadmap)

The roadmap lists mirror relabeling. It is built as data augmentation: half of each batch is reflected left to right, swapping the legs and flipping `vy`, `wz` and the lateral sensor signals. Bucket levels and budgets do not change, and neither does the reward. The signs were checked against the simulator. The V2 model is about 3 mm asymmetric side to side, so the mirror is approximate.

## Prices (Roadmap, not built or tested in MuJoCo)

The roadmap's claim is that a penalty weight equals the critic's gradient with respect to the budget:

```
lambda(s, b) = dQ(s, a, b) / db
```

This specification does not contain the roadmap's soft-termination price Bellman or curvature equations, because their exact form was in another missing doc (`price-bellman-theorem`). They must be re-derived and re-checked before E-D.

A small exact check is done (`audit/v2/price_toy.py`, `price_toy.json`): two actions, value iteration on the balance, multiplier known in closed form (0.70). V rises by the multiplier per token within 5% at rates 0.05 to 0.2, 14% at 0.4, and 62% off at 0.8, where the burst outlasts the discount horizon and the limit hardly binds. Two consequences for E-D, as hypotheses for the neural case:

- Read prices where the limit binds and the bucket is not full.
- Measure across one action's cost, not by a local derivative: V(b) is a staircase with a step every `rate`, and the local slope was up to 10 times the multiplier in this example.

Also, with `gamma = 0.99` the horizon is 100 steps, the same as `burst_steps`. Spending at twice the rate takes about that long to empty the bucket, so the critic hardly feels it. A shorter burst may make budgets easier to learn. Not tested.

## What the policy can read (Choice, from 2 November)

The critic sees everything. With `--actor_hide achieved,levels` the policy does not read true speed or bucket balances, because the real robot measures neither. It reads the command, `u`, and 5 frames of joint angles, joint speeds, gyro, gravity and its previous action. `config/v2_deploy_*.yaml` also round angles to encoder steps and take joint speed from angle differences, as `hardware_tests/run_policy.py` must on the robot. Policies trained with true speed fell from 24-87 to 3-17 of 90 episodes upright when it was hidden from them.

## Comparison methods for E-B

All use the same env, reward, alive bonus, sensors and training flags (`fast/train.py`, except PPO). `tests/test_comparison_methods.py` checks them.

| Method | How | Config |
|---|---|---|
| Token-bucket budget-HER | bounded bucket memory, budgets as goals, mirror | `v2_deploy_budget`, `_gait` |
| Command HER | budgets off | `v2_deploy_cmd`, `_gait` |
| Plain SAC | budgets off, no relabeling (`n_sampled_goal: 0`) | `v2_deploy_plain` |
| Episode-budget HER | one allowance per episode, `mode: episode` | `v2_deploy_episode` |
| Dual-ascent HER-Loco | command HER with fixed limits and learned penalty weights, `lagrange` | `v2_deploy_lagrange` |
| PPO | `fast/train_ppo.py`, SB3, no HER, mirror or critic-only inputs | `v2_deploy_cmd` |

## Training setup used for survival

```bash
python fast/train.py --config config/v2_budget_survive.yaml \
  --exp_name NAME --seed S --n_envs 8 --total_steps 3000000 \
  --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints --plain_ckpt \
  --difficulty_ramp 0.4 --layer_norm --avg_q --ema_tau 0.002 --asym --mirror
```

Robot: AsterisCrack V2 at 1.66 kg (`envs/assets/robotV2/robot_v2_train.xml`). The servo, friction, backlash, delay and noise values come from the designer's published numbers, not from measurements on this robot.

## Status

| Claim | Status |
|---|---|
| Relabeling is exact | Tested against brute force |
| Window guarantee | Holds by the formula; tested on random sequences |
| Budget-HER survives on the V2 robot | Measured, no baseline yet. 3M steps, 2 seeds per arm: 87, 94, 34, 97% at medium; 54, 66, 17, 74% at full with pushes (10 s) |
| Budget-HER walks in MuJoCo | One run of four: alternating at about 0.12 m/s with gait rewards, 8/30 with pushes. Below the E-A band. Not shown without gait rewards |
| The policy follows the budget | Not shown with the plain bucket: tight budgets overdrawn in most episodes. The bounded-memory fix is untested |
| One run covers the trade-off curve (E-C) | Not tested |
| Prices match Lagrangian multipliers (E-D) | Not tested |
| Works on the real robot (E-E) | Not tested |

## Decisions to review

1. Reward when overdrawn: zero base reward, or a softer penalty.
2. Whether `alive_bonus` stays in the final method. It helps survival but is a hand-set term.
3. Whether gait shaping stays off. The paper's strongest claim needs it off.
4. Budget ranges, `burst_steps` and `window_steps`. Bounded memory is a hypothesis under test, chosen after seeing results on seeds 7000-7029, so any claim about it needs new seeds.
5. Whether the critic's budget gradient should use `u`, the rate, or the bucket level for the price calculation.
