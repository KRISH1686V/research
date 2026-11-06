# Biped Walking Project: What I Am Doing and Why

Nov 2, 2026 · @yasho

## The short version

I am teaching a small, low-cost, two-legged robot to walk, and testing a training method that could learn every walking limit in one training run. The method has not yet been shown to work on this robot.

Today, each limit on a walking robot is usually built in by hand before training. Foot slip, time with both feet off the ground, and motor strain each get a penalty weight chosen by trial and error. Changing a limit means retuning and training again.

The method I am testing makes the limit part of the goal the robot receives. If it works, one training run covers many limits, and I choose the walking style afterward by setting a number. That would save training time and remove hand tuning.

The aim is a paper in a Q1 journal, with a real robot demonstration. The work is not ready for that yet. On the correct robot model, a stronger action filter has brought alternating walking without gait rewards, using only what the robot can measure. At 7.2 million training steps, all 5 seeds pass the fixed walking goal on 100 new test seeds, and the same setup without budgets is worse on every seed. The robot still takes about 6 short steps per second, so a budget on steps is now being trained to get about 2 steps per second at the same speed. At 4.8 million steps, training without budgets almost never walked: 1 alternating walk in 500 episodes, against 226 with budgets. These results come from the simulator only. On the real robot, only an IMU tilt check has been run, and no policy has moved it.

## The problem I am solving

A walking robot has to meet several competing demands, and the usual way of handling them takes repeated hand tuning and retraining. That is the cost I am trying to remove.

### Why limits matter

Staying upright is not enough. A robot can stay up and still walk badly, in ways that would hurt real hardware or stop it working. I track three habits:

| Habit | What it means | Why it matters on a real robot |
| --- | --- | --- |
| Flight | Both feet off the ground | Each landing is a hard impact on the servos and printed parts |
| Stance slip | A foot slides while carrying weight | The robot loses its footing and cannot push forward reliably |
| Torque saturation | Motors held at their strength limit | The servo has nothing left to recover with |

These matter more on my robot because it uses hobby servos. They give at most 2.94 Nm at 12 V, so there is little strength to spare.

### Why the usual approach is slow

The usual approach adds a penalty for each bad habit to the training reward. Each penalty has a weight, and people find good weights by trial and error. One training run gives one walking style for one set of weights.

So a new requirement means new weights and a new run. Showing a trade-off, such as speed against foot slip, takes several separate runs. On this machine, one 3-million-step run takes about 4 to 5 hours when four run together.

### Why the common shortcut has a weakness

A popular training shortcut is hindsight relabeling. If the robot was told to walk at 0.2 m/s but actually moved at 0.1 m/s, training also treats that attempt as a success at 0.1 m/s. This gets far more learning from each attempt.

The weakness is that it only looks at speed. In principle, a hop that happens to reach the right speed also counts as a success. The method is meant to close that gap by making the limits part of what gets relabeled.

On the old simulator and robot, hopping was rare. On the correct robot with longer training, relabeling speed alone gave running or bounding in 338 of 500 episodes and an alternating walk in 1. The risks section covers what that means for the paper.

### Why simulation alone is not enough

I train in simulation because a real robot cannot fall thousands of times. But a policy can do well in a simulator and fail on real hardware when the simulator is wrong. I have already seen this kind of failure inside simulation itself, described in the robot and simulator section.

## The idea: make the limit part of the goal

The method is called budget-as-goal hindsight relabeling. The robot is told how fast to walk and how much of each bad habit it may spend. Training then learns all those combinations from one stream of experience.

&#91;embedded content: training loop · 6 steps, then choosing a walking style\]

Ordinary hindsight training relabels only the speed. Step 4 relabels the budgets as well, which is the new part.

### 1. The goal includes budgets

A normal goal says "walk forward at 0.2 m/s". Here the goal also carries three budget numbers, one each for flight, slip and torque. Each number runs from 0 (tightest) to 1 (loosest).

```latex
g = [v_x,\; v_y,\; \omega_z,\; u_{\text{flight}},\; u_{\text{slip}},\; u_{\text{torque}}]
```

The policy reads this goal at every step. After training, I choose a walking style by changing those three numbers, with no retraining.

### 2. Each budget works like a token bucket

Think of a prepaid account that refills at a steady rate and has a maximum balance. At each control step, 50 times a second, the robot pays for the cost it actually caused. If the balance goes below zero, it stops earning its walking reward until the balance recovers.

```latex
b_{t+1} = \min(B,\; b_t + r - c_t)
```

Here b is the balance, r the refill rate, c the cost this step and B the maximum. During training, the critic sees the balances. The policy does not, because the real robot has no way to measure them. It only receives the budget it was given.

I use a bucket instead of a single total per episode because a total allows bad bursts. A robot could spend its whole allowance on three seconds of hopping and still pass. The bucket limits spending over every time window, not just the whole episode:

```latex
\sum_{t \in \text{any window of } T \text{ steps}} c_t \;\le\; r\,T + B
```

This guarantee is a standard token-bucket result. It holds as long as the balance stays at or above zero.

The first full test showed a flaw in this design. Because the balance never forgets, one early overspend can leave a tight budget in debt for the rest of the attempt. The robot then has no walking reward left to earn and no reason to follow the budget. The version training now only looks back 2 seconds, so the guarantee covers every window up to 2 seconds and the balance can recover. I also reset the limit ranges to match the costs measured in that test. On their own, these changes did not bring walking, and tight limits were still often missed.

### 3. Hindsight relabels the command and the budgets together

After an attempt, training revisits each moment and asks what goal it would have been good for. The command is replaced by a speed the robot really reached later in that attempt. Each budget is then kept, replaced by what the robot really spent, or redrawn at random.

So a single attempt teaches about many speeds and many limits. The balances for a new budget can be recomputed exactly with a short formula, without replaying the attempt. My tests check that formula against a slow step-by-step calculation.

### 4. Mirroring gives free experience

A left-right flip of a walking motion is also a valid walking motion. So half of each training batch is flipped, swapping the legs and reversing sideways signals. The model is about 3 mm off perfect symmetry, so the flip is a close approximation, not exact.

### Current settings

These are my working choices, not final or proven values. They can change after testing.

| Setting | Value | Why |
| --- | --- | --- |
| Flight refill rate | 0.002 to 0.2 per step (fraction of time in flight) | Covers almost none up to a lot |
| Slip refill rate | 0.02 to 0.2 m/s per step | Based on costs seen in earlier runs |
| Torque refill rate | 0.05 to 0.8 per step (fraction of motors at limit) | Based on costs seen in earlier runs |
| Maximum balance | 100 steps' worth of refill (2 seconds) | Allows short bursts, such as a recovery step |
| Budget tightening | Loose at the start, full range by 40% of training | So the robot learns to walk before it learns limits |
| Unconstrained episodes | 20% of episodes | Keeps plain walking in the training data |
| Survival bonus | 0.2 per step while upright | Added to raise survival; speed must be reported beside it because it could reward standing still |

## Why this could be new and useful

The new part is treating rolling gait budgets as hindsight goals, relabeled together with the speed command, for legged walking. The project roadmap records about 30 literature searches that found no earlier version. That check has not been repeated, so the paper should say "to our knowledge" and claim "first" only for this part.

The working title is "Budget-as-Goal Hindsight Relabeling: Training a Low-Cost Biped Once for Every Gait Constraint". It has three planned contributions.

| Contribution | What it adds | Status |
| --- | --- | --- |
| The method | Token-bucket gait budgets, command relabeling and mirroring together, so one run learns every limit | Running in MuJoCo. At 4.8 million steps, 2 of 5 seeds pass the fixed goal, and 3 are needed |
| Prices | Reads what each unit of budget is worth directly from the trained critic, instead of training penalty weights | Checked numerically on toy problems, per the roadmap. In MuJoCo, critic prices rise as limits tighten on 2 seeds, but they have not been compared with penalty multipliers |
| Hardware demo | The method on a 12-joint robot built from hobby servos | Not started |

### Why it would be useful

- **Less training.** One run could replace a series of runs with different penalty weights.
- **Less hand tuning.** The strongest result would be walking with no hand-tuned gait rewards at all.
- **Choice after training.** The same trained policy could walk carefully on a slippery floor or more freely elsewhere, by changing a number.
- **Low-cost hardware.** Hobby servos make limits matter more, and a working result would be reachable for small labs.

### The price idea, in plain words

A penalty weight says how much a unit of cost hurts. If the critic's estimated value depends on the budget, then its slope with respect to the budget says how much one extra unit of budget is worth:

```latex
\lambda(s, b) = \frac{\partial Q(s, a, b)}{\partial b}
```

If this matches the penalty weights found by a standard constrained method, I can read those weights off one trained network. The theory also links to bid prices and Hotelling's rule from economics. The equations in the original theory note must be re-derived, because that note could not be found.

### What must be true before any of this can be claimed

1. Budget-HER walks in MuJoCo on the correct robot model.
2. It does better than, or ties with, command HER with and without hand-tuned gait rewards, over 5 seeds.
3. The policy actually follows the budget it is given.
4. One run matches the trade-off curve of several separately trained policies.
5. Critic prices come within about 20% of standard multipliers.
6. The robot walks on real hardware, and changing the budget visibly changes its walking.

None of these has been shown yet. So the Q1 potential is real but conditional, and the work is not Q1-ready today.

The roadmap lists related work to cite and compare against: Saute RL, CCPO, SeqRoute, BCRL, Hindsight Preference Replay, Generalized Hindsight, QCP, CHRL, CaT, Kim et al., FastTD3 and FastSAC, along with the networking token bucket, bid prices and Hotelling's rule. Budgeted RL (Carrara et al., 2019) should be added, because it also puts the budget in the state. SeqRoute already relabels budgets in hindsight, but offline, as a total per session, for routing chat requests between language models. A fuller check on 6 November found no paper with the whole claim. Relabeling total budgets, the token-bucket recursion, budget inputs on a legged robot and command relabeling on legs are each already published. What survives, to our knowledge, is rolling budgets relabeled in hindsight with exact recomputation, together with the command, during online training, for several gait limits in one legged policy. Twelve checks remain before submission.

## The robot and the simulator

The robot is printed and ready, but every result so far comes from simulation. The simulator's servo, friction and noise values are the designer's published numbers, not measurements of my robot.

### The robot

| Part | Detail | Status |
| --- | --- | --- |
| Design | AsterisCrack BipedRobot V2, an open design | Confirmed as my robot |
| Joints | 12: hip yaw, hip roll, hip pitch, knee, ankle pitch and ankle roll on each leg | Model matches my robot files |
| Servos | Feetech ST3215 hobby servos, 12 V, 2.94 Nm at most | Real response not measured |
| Weight | 1.66 kg in the design files | Real weight not measured |
| Computer | Raspberry Pi 5 | Control loop timing not measured |
| Sensors | BNO085 motion sensor, plus each servo's position sensor | Real noise not measured |
| Control rate | 50 commands a second, over two servo buses | Real loop not tested with a policy |

The designer's own notes say the robot still walks badly and that its backlash was never measured. So I should not assume the published numbers are right for my build.

### Why I use MuJoCo, and why realism matters

MuJoCo is a physics simulator that lets the robot fall thousands of times safely. I use it on the CPU with the real robot model. I decided against faster GPU physics engines and made training faster by rebuilding the learning code instead.

The simulator only helps if it is close enough to reality, and I found out that mine was not. I took 5 policies that looked good and retested them with more realistic physics, out of 150 test episodes:

| Test condition | Episodes survived, of 150 |
| --- | --- |
| Original test | 140 |
| Motors 20% weaker | 127 |
| Realistic servo speed limits | 124 |
| Pushes lasting 25 ms | 76 |
| Control delay of 40 ms | 14 |
| Pushes lasting 100 ms | 0 |
| All corrections together, with pushes | 0 |

That result changed the project. Survival in the original simulator did not predict survival under realistic conditions.

### Mistakes I found and corrected

1. **Friction had no effect.** Low floor friction was ignored, because MuJoCo uses the higher of the foot's and the floor's values.
2. **Pushes were too short.** They lasted 25 ms, not the intended 0.1 s.
3. **Motor strength was wrong.** It scaled the joint target, not the torque.
4. **The robot model was wrong.** All training before the V2 model used a different robot, with different feet (75 by 60 mm against 90 by 38 mm).

The original code stays unchanged so old results can be reproduced. The fixes live in new files. All policy results from before the V2 model are not results for my robot.

### What the simulator now assumes

These come from the designer's published figures and need checking on my robot:

- Control delay of 0 to 50 ms, and 1.5 degrees of backlash.
- Foot friction from 0.25 to 0.8, and body mass within 20% of nominal.
- Pushes of 3 N lasting 0.1 s, plus sensor noise and sensor offsets.
- A servo model that loses strength as speed rises.

Run on the same sine-wave test as the real robot, the simulated servo lags 127 to 177 ms at 1 Hz. It moves only about 73% as far as commanded at that speed. Real servos may be faster, and a robot test will show that. If they are, the simulator must change.

## What I have built so far, and why

The tools for a fair test are in place: a trainer about 20 times faster, a corrected simulator, the budget method itself, and a safe way to measure the real robot. Each piece answers a specific problem.

| Piece | What it does | Why I needed it | How I checked it |
| --- | --- | --- | --- |
| Fast trainer | Runs learning and replay on the GPU | The comparison needs dozens of runs, and 1 million steps took about 20 hours. It now takes about 57 minutes alone | Same update math as the standard library, within 0.00000011 after 60 updates |
| Corrected simulator | Fixes friction, pushes and motor strength; adds servo limits and sensor noise | Survival in the old simulator did not hold up under realistic physics | Separate tests for each correction |
| Correct robot model | Builds a training model of the V2 robot, 1.661 kg | Earlier training used a different robot | Stands 500 of 500 steps with no policy, on 5 of 5 seeds, on easy and full realism |
| Budget method | Token buckets in the simulator, plus exact relabeling during training | This is the method under test | Relabeled balances match a slow step-by-step calculation |
| Mirror | Flips half of each training batch left to right | Free extra experience | Signs checked against the simulator |
| Survival upgrades | Memory of the last 5 readings, gradually harder training, a critic that sees the true state, an averaged final policy, and a survival bonus | The first corrected policies had no memory, faced full difficulty from the start, and learned unstably | Tested on 2 November: results are under Where things stand |
| Independent evaluator | Scores policies separately from the training code, using contact forces and fresh random seeds | Numbers logged during training mislead. A fallen robot can even look fast | It has already caught false results |
| Robot test script | Measures real delay, servo response and sensor noise, and checks joint directions | Replaces the designer's numbers with my robot's numbers | 27 safety and arithmetic tests pass on pretend servos. Not yet run on a real servo |
| Joint check | Compares the training model with my robot's files, joint by joint | One wrong joint would make the robot fall at once | No mismatches in axes, chain, offsets or left-right signs. Real directions still need the robot test |

The joint check also showed that hip pitch and hip roll are listed in a different order in training and on the robot. The code that runs a policy on the Pi must swap them. That code is not written yet.

I also tried many training variations, but I stopped treating them as a contest. The goal is longer survival for budget-HER, not picking winners, so I combined what was expected to help and test that.

## The experiments

Five experiments test the claims, one after another, and none has passed yet. Each exists because a careful reviewer would ask its question.

&#91;embedded content: experiment order · 5 experiments, 2 fallbacks\]

Dashed paths show what changes if E-A misses its date or E-B shows no benefit. The table gives the details of each experiment.

| ID | Question | Why it is needed | How it is run | Passes if | Status |
| --- | --- | --- | --- | --- | --- |
| E-A | Can the robot walk at all? | Comparisons mean nothing if nothing walks | At least 100 test episodes in medium realism, 3 training seeds, shaped and unshaped rewards reported separately | At least 90% survival, and a real speed of 0.15 to 0.25 m/s when told 0.2 | Not met |
| E-B | Is budget-HER better? | This is the paper's main claim | 5 seeds each: PPO, plain SAC, command HER with and without gait rewards, episode-budget HER, token-bucket budget-HER with mirror, and dual-ascent HER-Loco | Best or tied on survival, clearly less flight, no budget window broken. Time to walk is reported. The key test is walking without hand-tuned gait rewards | Not started |
| E-C | Does one run really replace many? | This is the practical benefit | One budget-HER policy swept across budgets, against separately trained penalty policies | One run matches the trade-off curve of several runs | Not started |
| E-D | Can prices be read from the critic? | This is the second contribution | Critic slope against SAC-Lagrangian multipliers at 3 fixed limits | Within about 20% | Not started |
| E-E | Does it work on the real robot? | A simulation result alone would not show that the method works on hardware | Best policy on the robot, with sensor logs, video, and two different budgets | It walks, and changing the budget visibly changes the walking | Not started |

Three points about how I run these:

- **Speed means real speed.** E-A counts the speed the robot achieved, not the speed it was told. A policy that survives by barely moving does not pass.
- **"Minimal" has a strict meaning.** I only call a reward minimal when gait shaping is off, so shaped and unshaped results stay separate.
- **Each seed counts as one result.** Test episodes from one seed are not independent proof, since one seed can get lucky.

Testing so far has also shown that a policy can change sharply between nearby training checkpoints, such as 24 of 30 at 800,000 steps and 0 of 30 at 900,000 steps. So I fix in advance which checkpoint counts, and use the final one.

## Where things stand

At 7.2 million training steps, with a stronger action filter and no gait rewards, budget-HER meets the fixed walking goal in the simulator on all 5 seeds, on 100 new test seeds, using only what the robot can measure. Without budgets, the same training stayed up in 81.4 of 100 episodes against 98.6, took alternating steps in 28 of 500 against 334, and spent 13.8% of the time off the ground against 4.4%. At 4.8 million steps, 2 of 5 seeds passed. At the mid-level budget, the 5 seeds stayed up in 94 to 100 episodes, took alternating steps in 63 to 83, and moved at 0.145 to 0.203 m/s. All three failures are at the loose budget: seed 3 stayed up in 68 and seed 4 in 87 (goal 90), and seed 7 moved at 0.142 m/s (goal 0.15). At the loose setting, training without budgets stayed up about as often (89.8 against 90.4 on average), but took alternating steps in 1 of 500 episodes against 226, spent 11.9% of the time off the ground against 6.1%, and moved at 0.120 against 0.193 m/s. Under pushes the survival gain is small: 20.0 against 18.4 of 30 over 10 s, and 13.0 against 12.4 over 30 s. Before the filter: Across 5 seeds with gait rewards, budgets cut time off the ground and bring far more alternating walking. The survival gain is small, and one seed mostly stands still. Without gait rewards, budgets nearly double survival under pushes and cut time off the ground to about a third, at about the same average speed when there are no pushes. With pushes, budget runs are slower, so part of the survival gain may simply be lower speed. The gait is still mostly bounding, not a clean walk. On the real robot, only an IMU tilt check has been run, with torque off, and no policy has moved it. A scan of saved checkpoints showed walking was still improving at 4.8 million steps, which led to the 7.2 million step runs. The lab floor is not treated as a constraint, because it can be changed on the real robot.

### Proven, by measurement in simulation

None of these shows that the real robot can walk.

| Finding | Evidence | Limit |
| --- | --- | --- |
| At 7.2 million steps the fixed walking goal is met, and budgets clearly beat no budgets | Seeds 3 to 7, 100 new test seeds, medium realism, command 0.2 m/s: all 5 pass. Loose budget, with and without budgets: upright 98.6 and 81.4 of 100 on average; alternating walk 334 and 28 of 500; running or bounding 36 and 366; both feet off the ground 4.4% and 13.8%; speed 0.211 and 0.188 m/s. Budgets do better on survival, walking and air time on every seed | Simulator only. About 6 short steps per second, so the gait is not yet what the robot needs. Pushes and tight limits remain weak |
| Neither the smoothness loss nor the stronger filter is adopted | Seeds 3, 4 and 7 at 7.2 million steps. Smoothness loss: output change about 0.44 per step against 0.63, but seed 3 stayed up in only 53 of 100 at the mid budget. Filter 0.2: alternating walk at mid 67, 59, 73 against 86, 100, 95. Neither changed the step rate | One batch each, with untuned weights |
| At 7.2 million steps, all 4 finished seeds pass the goal's walking measures | Seeds 3, 4, 5 and 7, 100 new test seeds, medium realism, command 0.2 m/s, with only training length changed from 4.8 million. Loose budget: upright 94, 100, 100, 100; speed 0.217, 0.172, 0.218, 0.200 m/s. Mid budget: upright 100, 99, 98, 100; alternating walk 86, 100, 95, 95; within budget while moving at 0.10 m/s or more 79, 95, 94, 92. Tight budget: speed 0.166, 0.199, 0.176, 0.178 m/s; within budget while moving 46, 68, 74, 79. The same 4 seeds also pass on the test seeds used at 4.8 million. At the mid budget the robot takes about 6 real steps per second, each about 3.1 to 3.8 cm long | A result so far: seed 6 and the comparison without budgets are still training, so the goal is not complete. At the tight budget, the strict check passes only 15, 14, 0 and 8 episodes, because slip runs above its rate. Pushes remain weak: over 30 s, upright 18, 23, 21 and 7 of 30, and seed 7 fell more than at 4.8 million. Still about 7 foot landings per second and about 10 mm of lift. Simulator only |
| At the lab floor's friction, 1 of 5 seeds passes the same measures | 4.8 million steps, friction 0.3 to 0.4, 100 new test seeds, medium realism, command 0.2 m/s, seeds 3 to 7. Loose budget: upright 82, 87, 100, 98, 99; speed 0.191, 0.208, 0.210, 0.175, 0.126 m/s; alternating walk 0, 13, 33, 24, 1. Mid budget: upright 99, 100, 99, 97, 100; alternating walk 50, 70, 70, 43, 61; within budget while moving at 0.10 m/s or more 61, 67, 96, 33, 43. Tight budget, within budget while moving: 2, 1, 18, 0, 2. Without budgets at loose: upright 87.4 against 93.2 on average; alternating walk 0 against 71 of 500; speed 0.104 against 0.182 m/s; both feet off the ground 13.9% against 8.5% | Not part of the goal, and not a constraint, because the floor and feet can be changed on the real robot. Simulator only. The friction number is as reported, and how it was measured is not recorded. Seed 3 without budgets mostly stands, which flatters its survival |
| Falls and limit breaks are more common on slippery floors | All 5 seeds at 4.8 million steps, 100 new episodes per seed. Loose budget, upright by friction quartile from lowest to highest: 88, 90, 97, 97%. Seed 3: 29 of 50 on the slippery half, 44 of 50 on the other. Mid budget: 93 of 500 episodes went over budget, always on the flight limit first, with mean friction 0.38 against 0.56 for the rest | Diagnostic seeds, not a goal check. This shows where the failures happen, not why. The lab floor is reported to be about 0.3 to 0.4 with PLA feet, but it can be changed on the real robot, so it is not a constraint. The fixed test range stays as it is |
| Walking was still improving at 4.8 million steps | All 5 seeds combined, 200 new episodes per budget, at 2.4, 3.2, 4.0 and 4.8 million steps. Mid budget: alternating walk 2, 8, 114, 159; within budget while moving at 0.10 m/s or more 0, 2, 107, 164. Loose budget: alternating walk 31, 30, 41, 80; speed 0.015, 0.112, 0.172, 0.189 m/s; upright 175, 186, 172, 187 | Survival at the loose budget shows no clear trend. Single seeds swing between checkpoints. Trend check only, not a goal check |
| A code review and the tests found no bug in the budget code | Buckets and relabeling match brute force exactly, including after the replay memory has been overwritten several times. All tests pass. Every smoothing weight and the torque penalty are 0 in these configs, so the reward is tracking, progress, staying alive and falling only, and "no gait rewards" holds | This shows the code does what it is meant to. It does not show that the simulator matches the robot |
| Nothing in training discourages jittery movement | At the mid budget, 4.0 to 4.8 million steps, the policy's output changes by about 0.6 per step (RMS, on a -1 to 1 scale), with about 7 foot landings per second and about 8 mm of lift. The action filter is the only smoother | Not a bug. A smoothness penalty would count as shaping, so it would have to be reported as a separate arm |
| At 4.8 million steps, 2 of 5 seeds pass the fixed walking goal; 3 are needed | Seeds 3 to 7, 100 new test seeds, medium realism, command 0.2 m/s. Loose budget: upright 68, 87, 99, 98, 100; speed 0.197, 0.208, 0.228, 0.190, 0.142 m/s. Mid budget: upright 97, 99, 100, 94, 100; alternating walk 74, 83, 63, 76, 83; within budget while moving at 0.10 m/s or more 79, 76, 96, 64, 78. Seeds 5 and 6 pass | The goal is not met. Seeds 3 and 4 fail on falls at the loose budget, seed 7 on speed. Longer training also stretched the difficulty ramp, so the two effects are not separated. Limits are still sometimes broken: the strict check at mid gives 73, 78, 92, 61, 71 episodes clean |
| At 4.8 million steps, training without budgets bounds rather than walks | Same setup, 5 seeds, loose budget, 100 new test seeds. With and without budgets: alternating walk 226 and 1 of 500; running or bounding 120 and 338; both feet off the ground 6.1% and 11.9%; speed 0.193 and 0.120 m/s; upright 90.4 and 89.8 per 100 | No survival gain at loose. Under pushes, upright of 30 is 20.0 and 18.4 over 10 s, and 13.0 and 12.4 over 30 s, on reused test seeds. Seed 3 without budgets stands still in 85 of 100. Seed 7 is faster without budgets, 0.169 against 0.142 m/s |
| Tight budgets are still not met while moving | 4.8 million steps, 100 new test seeds, all budgets at the tight setting. Upright 100, 97, 100, 97, 100; speed 0.071, 0.080, 0.085, 0.047, 0.121 m/s; within budget while moving at 0.10 m/s or more 10, 17, 16, 2, 44 | Not part of the fixed goal, but it limits the claim that one policy covers every limit |
| A stronger action filter brought walking and limit control without gait rewards | Seeds 3 and 4, 100 new test seeds, medium realism, loose, mid and tight budgets. Upright: 95, 96, 98 and 94, 100, 98. Alternating walk: 30, 74, 77 and 49, 83, 65 (before the filter, under 1%). Speed: 0.23, 0.19, 0.14 and 0.15, 0.12, 0.09 m/s. Time with both feet off the ground: 6.8, 2.6, 1.2% and 4.8, 1.2, 0.5%. Stayed within budget while moving at 0.10 m/s or more: 92, 65, 50 and 81, 75, 34 | At 3 million steps. These are the 2 best of 5 seeds. Seeds 5, 6 and 7 walked in only 14 to 19 of 270 episodes and mostly stood at mid and tight budgets. The filter was chosen using earlier test seeds, then checked on these new ones. At full realism with pushes: upright 62 to 83, alternating walk 15 to 49, and limits mostly broken when tight |
| With the filter, budgets are needed for alternating walking | Paired seeds 3 to 6, loose budget, with and without budgets. Alternating walk: 35 and 0 of 120 episodes. Both feet off the ground: 5.3% and 18.8%. Upright: 27 and 24 of 30; with pushes, 17 and 12; 30 s with pushes, 10 and 4. Steady speed: 0.15 and 0.16 m/s | 4 seeds per method, 30 episodes per seed. Most of the walking comes from 2 seeds |
| Holding heading with the IMU mostly stops the turning, with no retraining | All 5 seeds at 4.8 million steps, 100 new episodes per seed, loose and mid budgets. Median heading change over 10 s fell from 0.25 to 0.52 rad to 0.03 to 0.10 rad. Upright changed by at most 4 of 100, and speed by -0.008 to +0.023 m/s. Using only the gyro, with its noise and bias, turning fell to 0.11 to 0.23 rad. Upright averaged 91.2 against 92.8 at loose and 96.8 against 98.4 at mid, with seed 3 losing the most: 73 to 67 and 98 to 91 | Simulator only. The real IMU has only passed a tilt check, and has not been used to steer. Not tested with pushes. Sideways drift of 0.19 to 0.73 m remains, and this does not fix the seeds that fail the goal |
| A tighter speed reward made things worse | Upright across 9 budgets: 68 and 89%, against 81 and 97% before. Almost no alternating walking on one seed, none on the other | 2 seeds, 30 episodes per budget |
| Long motor delay hurts the filter policies most, not pushes | Mid budget, upright of 30, seeds 3 and 4. Baseline: 26, 30. Pushes alone: 24, 29. Delay 30 ms: 25, 30. Delay 50 ms: 12, 25 | 2 seeds, 30 episodes each. The robot's real delay has not been measured |
| The bridge's servo limits would make these policies fall | Upright of 30, seeds 3 and 4. Target speed capped at 4.6 rad/s: 25, 30. At 2.3 rad/s: 7, 17. At 1.2 rad/s: 0, 0. Torque at 70%: 23, 29. At 50%: 12, 22 | Simulated limits, not hardware measurements. Servo acceleration is not modelled |
| Speed does not follow the command well | Commands of 0.10, 0.15, 0.20, 0.25 and 0.30 m/s gave 0.005, 0.06, 0.19, 0.18 and 0.14 m/s on seed 3, and 0.009, 0.09, 0.13, 0.12 and 0.11 m/s on seed 4 | Mid budget, at 3 million steps. A longer memory of 10 frames walked less on 2 seeds, so it did not help |
| Prices read from the critic rise as limits tighten | Flight price at mid, tight and tightest settings: 6.8, 19.4 and 42.2 on seed 3; 3.8, 14.6 and 43.1 on seed 4 | Not yet compared with penalty multipliers, so this does not prove the price claim |
| The fast trainer learns about as well as the standard one | Survival of 57 against 53 of 60 at 1 million steps, easy realism, 0.2 m/s; 46 against 47 at 900,000 | 2 seeds per side, so not proof of equal learning |
| Survival in the old simulator did not hold up | 140 of 150 fell to 0 of 150 with realistic physics | Old robot model |
| Policies on the old setup could survive but walked too slowly | Medium realism, 100 test episodes per seed, with gait rewards: 99, 88 and 99 survived, at 0.109, 0.087 and 0.127 m/s. Goal is 0.15 to 0.25 m/s | Wrong robot, uncorrected simulator |
| A tighter speed reward helped one seed only | 100 of 100 at 0.174 m/s on one seed; 98 at 0.060 and 63 at 0.090 on the others | Wrong robot, uncorrected simulator |
| Gait rewards were not essential on the old setup | Without them, seed 4 survived 29 of 30 at 0.155 m/s; with them, 28 of 30 at 0.170 m/s | One seed, easy realism, wrong robot |
| Hopping was rare | 0 hops in 2,790 episodes | Old simulator and robot; may differ on V2 |
| Corrected physics made survival far harder | Old robot baseline, seed 3: 7 of 30 at medium. With no delay, 29 of 30 at only 0.033 m/s | One seed, wrong robot |
| An early budget-HER did not walk | At 900,000 steps: at most 2 of 30 alternating walks at any budget. One seed mostly stood, two moved and fell | Early version, wrong robot, shorter training. Not a fair test of the method |
| Budget-HER stays up much longer on the correct robot | Share of episodes still upright across 9 budgets, four runs: 87, 94, 34 and 97% at medium realism over 10 s. At full realism with pushes: 54, 66, 17 and 74% over 10 s, then 49, 49, 19 and 50% over 30 s at 3 budgets | 2 seeds each, with and without gait rewards. No baseline yet, so this does not show that budgets caused it |
| Only one of those four runs really walks | With gait rewards, seed 4: alternating walk in 24 of 30 episodes at 0.12 m/s with a loose budget, and 8 of 30 with pushes. Without gait rewards: quick bounding, with no alternating walk at loose budgets | Below the 0.15 to 0.25 m/s goal. Steps are short and quick: about 7 foot landings per second and about 10 mm of lift |
| The policy does not keep to tight budgets | Asked for a flight rate of 0.002 per step, one run used 0.11. Without gait rewards, the tightest budgets were overdrawn 59 to 100% of the time. With all budgets tight, the robot moved only 0.02 to 0.06 m/s | Measured on the first bucket design, which never forgets an overspend |
| Results swing between saved checkpoints of a run | One run stayed up in 18 of 20 episodes at 2.25 million steps and 0 of 20 at 2.75 million. Its final checkpoint stayed up in 17% of episodes | 20 episodes per checkpoint and per budget. I do not choose checkpoints from test results |
| The earlier policies depended on a speed reading the robot does not have | Upright in 24 to 87 of 90 episodes with true speed. With speed reading as the command: 18 to 61. With speed reading 0 and balances reading full: 3 to 17 | A speed estimate on the robot was not tested |
| A policy can stay up and move using only what the robot can measure | Budgets with gait rewards, 100 new test seeds, loose budget: 94 upright at medium realism at 0.14 m/s, and 60 upright at full realism with pushes | Best of 3 training seeds; another mostly stood still. About 11 foot landings per second and 5 mm of lift, so closer to a shuffle than a walk. About a third of upright episodes reach the 0.15 to 0.25 m/s goal |
| With gait rewards, budgets cut time off the ground and bring more walking | Mean of 5 seeds, loose budget, medium realism, with and without budgets. Both feet off the ground: 4.9% and 14.4%. Upright: 27.8 and 26.0 of 30. Steady speed: 0.12 and 0.14 m/s. Alternating walking: 37 and 3 of 150 episodes | 30 episodes per seed. One budget seed mostly stood still. The survival gap shrank going from 3 to 5 seeds, and is small at 30 s with pushes (13 and 12 of 30). Tight limits while moving are still unsolved. |
| One policy now takes alternating steps in over half its episodes, using only what the robot can measure | Budgets with gait rewards, seed 6, 100 new test seeds: alternating walk in 57, upright in 95, at 0.14 m/s. At full realism with pushes: 28 and 76, at 0.09 m/s | One seed; seed 7 reaches 28 of 100. About 9 foot landings per second and 9 mm of lift. The path curves about 0.9 m sideways in 10 s. Below the speed goal |
| Without gait rewards, budgets clearly help survival and air time | Mean of 3 seeds, loose budget, medium realism, with and without budgets. Both feet off the ground: 9.3% and 29.7%. Upright: 25 and 17 of 30; with pushes, 15 and 7 of 30. Steady speed: 0.13 m/s for both, or 0.16 and 0.24 m/s with pushes. On 100 new test seeds, the two better budget seeds stayed up 95 and 86 times at about 0.16 m/s | Mostly bounding or irregular steps, with almost no alternating walking. The third budget seed stayed up only 18 of 30 times. Speeds vary a lot between seeds. Tight limits while moving are still unsolved |
| The 2-second bucket alone did not bring walking | With true speed as an input: upright 73 and 88% against 87 and 94% before, and almost no alternating walking. One seed still ignored tight budgets, and the other met them by barely moving | 2 seeds. The limit ranges changed at the same time |
| Averaging the last 4 saved checkpoints trades speed for survival | On new seeds, upright rose from 56 to 83% of 1,080 episodes. On the best later run, speed fell from 0.14 to 0.11 m/s. Without gait rewards, the robot mostly stood still | So I report final and averaged policies separately |
| GPU sharing made training about 2.8 times faster | 3 million steps in 1.7 hours against 4.7 hours, with 4 runs at once. A short test run gave identical numbers | A GPU fault can stop all 4 runs, and training cannot resume |
| Relabeling is exact | Matches a step-by-step calculation in tests | Shows the arithmetic is right, not that the method works |
| The training model matches my robot's files | No mismatches across 12 joints | File check only |

### Hypotheses, not yet tested

- One budget-HER run can cover all the limits.
- Budget-HER can walk well with no hand-tuned gait rewards.
- The critic's slope matches standard penalty multipliers within about 20%.
- The survival gain came from memory, gradual difficulty and steadier learning. These went in together, so their effects have not been separated.
- Some seeds settle on standing still under limits and never find walking. More reward for moving, or longer training, may fix this. Both are being tested.
- The simulated servo may be slower than the real one, because delay, lag and backlash may partly double-count.

### Unknown: not determined

| Question | How it gets answered |
| --- | --- |
| Does budget-HER do better than the same setup without budgets? | So far budgets show less time off the ground on every seed tested, and more walking with gait rewards. The survival gain is small with gait rewards, and no-gait seeds 6 and 7 are still to come. More seeds give results around 17:30 on 3 November |
| Does walking at tight budgets repeat on more seeds, and hold up with pushes? | Same runs: requested cost against measured cost |
| How does the robot really behave: weight, delay, servo response, sensor noise? | Robot test script, about 15 minutes on the robot |
| How much backlash does the real robot have? Foot friction on the lab floor is reported as about 0.3 to 0.4, with PLA feet, but how it was measured is not recorded. | Not covered by the current script; needs a further test |
| Do real joints move the way the model expects? | The directions step of the robot test |
| Does any setup reach 0.15 to 0.25 m/s with at least 90% survival, on 3 seeds? | E-A on the V2 model |
| Can any trained policy walk on the real robot? | E-E |

An earlier foot-lift probe was invalid, so I do not use it as evidence.

## Risks and fallback plans

The biggest risk is that walking on the correct robot is not reached by Nov 7, 2026, the date set for the walking gate. If it is missed, I stop tuning and change the publication plan, as described below.

| Risk | Why it matters | What I will do |
| --- | --- | --- |
| The walking gate is missed by 7 November | Without a walking baseline, the comparison means little | Stop tuning, and report what the Q1 journal paper still needs |
| Budget-HER is not better than command HER | The main claim fails | The paper becomes an analysis of when and why hindsight fails for gaits, with the theory, still aimed at a Q1 journal |
| The hopping problem barely appears | "Less flight" had almost nothing to beat on the old setup, with 0 hops in 2,790 episodes. On the correct robot, runs without gait rewards spend 12 to 19% of the time with both feet off the ground at a loose budget, so this risk looks smaller. At 4.8 million steps, runs without budgets were running or bounding in 338 of 500 episodes | Compare against baselines with and without gait rewards. The main test is walking without hand tuning, and train-once value |
| The simulator does not match the robot | Policies may fail on hardware | Measure the robot, then update the simulator before final runs |
| The servo bridge limits are stricter than training | The bridge allows 1.2 rad/s and half torque. In the simulator, these limits make current policies fall | Measure both settings. The Pi code does not go through the bridge and uses the servo's fastest setting. Raise the torque limit for ground tests |
| Unsafe joint limits in `servos.yaml` | They allow the knee to bend the wrong way, up to about +120 degrees, where the designer's limit is about +5 | Tighten them before any walking test. The test script already uses the tighter limits |
| Two supporting notes are missing | The method plan and the price theory note could not be found | I wrote a working method spec. The price equations must be re-derived before E-D |
| Shared computer | Another job uses about 11 of the 20 CPU cores and must not be disturbed | At most 4 runs at once, which slows each run |

### Parked for now

These are deliberately not being worked on, so that time goes to the experiments above: a servo heat model, a separate price network, curriculum-based relabeling as a contribution, and GPU physics engines.

## What happens next

The next decision comes from a fourth budget, on steps, now training on 2 seeds, with all 5 by 7 November. It aims for about 2 steps per second at the same speed, with a higher foot lift, and adds no gait reward. A policy smoothness loss and a stronger action filter were tried first, and neither is adopted. Work on the real robot starts on Monday 9 November.

1. **Make walking reliable: at least 3 of 5 seeds passing the fixed measures.** Report survival, speed, whether the robot really walks, and whether its costs follow the budget, against the same setup without budgets.
2. **Measure the real robot.** Starts on Monday 9 November. Calibrate, hang it, run the test script for about 15 minutes, and weigh it.
3. **Update the simulator with the real numbers.** Retrain if they differ enough to matter.
4. **Train the other comparison methods.** PPO, plain SAC, episode budgets and learned penalties. The code exists, but only short test runs have been done.
5. **Run E-A, then E-B, E-C and E-D.** Use seeds fixed in advance, final checkpoints, and fresh test seeds.
6. **Try the Pi code.** It runs a policy 50 times a second, maps the joint order, and includes safety cut-offs. It has passed tests in the simulator and on pretend servos, but has never run on real servos.
7. **Run E-E on the robot.** Start supported, then move to short walks with sensor logs and video, at two different budgets.

The roadmap allows about a month for this. That timing is a target and is not certain, since it depends on steps 1 and 2.

## Glossary

These are the terms used above, in plain language.

| Term | Meaning |
| --- | --- |
| Backlash | Small free play in a servo's gears before it starts pushing |
| Budget | How much of a bad habit the robot may spend |
| Budget-HER | The method under test: hindsight relabeling of both command and budgets |
| Checkpoint | A saved copy of the policy at a point during training |
| Command HER | Hindsight relabeling of the speed command only |
| Critic | The part of training that estimates how good a situation and action are |
| Dual ascent / SAC-Lagrangian | Standard methods that adjust penalty weights automatically during training |
| Episode | One attempt, up to 500 steps (10 seconds) in my tests |
| Flight | Time when nothing is touching the ground |
| Gait | The pattern of a walk, such as step timing and foot lift |
| Gait shaping | Extra hand-tuned rewards that push the robot toward a chosen walking pattern |
| Hindsight relabeling (HER) | Treating what the robot really did as if it had been the goal, so that each attempt teaches more |
| Medium / full realism | Simulator difficulty. Easy has no delay, lag, backlash or randomness. Medium adds a 10 ms delay, lag and random physics. Full adds delay up to 50 ms, backlash and pushes |
| Mirror | Using a left-right flip of an experience as extra training data |
| MuJoCo | The physics simulator I use |
| Policy | The trained network that turns sensor readings and the goal into joint targets |
| PPO, SAC | Two widely used reinforcement learning algorithms. I train with SAC |
| Price | What one more unit of budget is worth to the robot |
| Q1 | The top quarter of journals in a field, by citation ranking |
| Reinforcement learning | Learning by trial and error from rewards |
| Seed | The starting random number of a run. Each training seed counts as one independent result |
| Stance slip | Sliding of a foot that is carrying weight |
| Survival | Staying upright to the end of a test episode |
| Token bucket | An allowance that refills steadily up to a maximum, and limits spending over any time window |
| Torque saturation | Motors held at their strength limit |
