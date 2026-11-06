# Biped Project: Done and Left

Nov 4, 2026 · @yasho

## Where it stands

The simulator walking goal is met: at 7.2 million training steps all 5 seeds pass on 100 new test seeds, and the same setup without budgets is worse on every seed. The robot still takes about 6 short steps per second, so a fourth budget on steps is now training to get about 2 steps per second at the same speed. The method comparisons, the real robot and the writing are still to come.

The target is a Q1 journal paper with the real robot. The aim is to finish all testing by 1 December, with about 15 December as the backup, and to submit in mid to late December. These dates are estimates. Acceptance cannot be promised, and no policy has moved the real robot yet.

## Done

The method is built and tested, and walking in the simulator now passes on every seed finished so far. All results below come from the simulator unless marked as the real robot.

| What | Result | How strong |
| --- | --- | --- |
| Walking at 7.2 million steps | Seeds 3, 4, 5 and 7 all pass on 100 new test seeds. Loose budget: upright 94, 100, 100, 100 of 100; speed 0.217, 0.172, 0.218, 0.200 m/s. Mid budget: alternating walk 86, 100, 95, 95 of 100; within budget and moving 79, 95, 94, 92 of 100 | 4 seeds, so a result so far. The same seeds also pass on a second set of test seeds |
| Budgets on and off at 4.8 million steps | Loose budget, all 5 seeds: alternating walk 226 and 1 of 500; both feet off the ground 6.1% and 11.9%; speed 0.193 and 0.120 m/s; upright 90.4 and 89.8 of 100 | 5 seeds. Clear on gait, no survival gain. Must be repeated at 7.2 million steps |
| Training setup | Correct robot model, realistic physics, stronger action filter (0.3), left-right mirroring, no gait rewards | Each step measured against the last |
| Only measurable inputs | The policy reads no true speed and no budget balances | Checked in the saved policies |
| Budget code | Rolling budgets and relabeling match brute-force checks exactly, including after the replay memory is overwritten | 124 training tests and 27 robot-tool tests pass; code review found no bug |
| Fast trainer | About 500 steps per second per run, with 4 runs at once | Measured. Learning matched the standard trainer on 2 seeds per side |
| Heading hold from the gyro | Turning over 10 s falls from 0.25-0.52 to 0.11-0.23 rad; upright falls from 92.8 to 91.2 of 100 at loose | 5 seeds. An option, not part of the goal |
| Early price check | Flight prices from the critic rise as limits tighten: 6.8, 19.4, 42.2 on seed 3 and 3.8, 14.6, 43.1 on seed 4 | 2 seeds at 3 million steps. Not yet compared with multipliers |
| Robot package | Test scripts and the 7.2 million step policies for seeds 4 and 5, ready to copy to the Pi | Run on pretend servos only |
| IMU check, real robot | Passed: gravity and gyro agree, mount `x,y,z`, torque off | Checks direction, not accuracy |

## Running now

The workstation is running these on its own. Progress is as of 06:40 on 6 November, and the finish times are estimates.

| What | Status | Expected |
| --- | --- | --- |
| Step budget, seeds 3 and 4 | Started 06:36 | About 11:15 today, with the gait check after |
| Foot-lift budget, seeds 3 and 4 | Started 07:04 | About 11:45 today |
| All three style budgets together (steps, foot lift, flat foot), seeds 3, 4 and 6 | Starts when the step runs end | About 16:00 today |
| The same, seeds 7 and 5 | Starts when the lift runs end | About 16:30 today |
| Plain SAC baselines | Paused, so all 4 GPU slots go to the walking style | After the style is decided |

## Left

The simulator work can finish by about 8 November if nothing fails, and the real robot starts on Monday 9 November. All dates are estimates.

**Simulator (runs on the workstation)**

- [x] Goal verdict: met in the simulator on 5 of 5 seeds, with budgets off worse on every seed
- [x] Jitter tests: neither the smoothness loss nor the stronger filter is adopted. Both kept about 6 steps per second and walked less at mid budget
- [ ] Step budget for about 2 steps per second at the same speed and a higher foot lift: seeds 3 and 4 today, all 5 by Saturday 7 November
- [ ] Freeze the procedure, once the step budget is decided, by about 8 November
- [ ] Other methods, 25 runs: PPO, plain SAC, command relabeling with gait rewards, episode budgets and dual ascent, 5 seeds each. Plain SAC has started
- [ ] Train once against several runs, 20 runs: fixed penalties at 4 settings, 5 seeds each
- [ ] Prices against multipliers, about 10 more runs: re-derive the price equations first, because the original note is missing, then run dual ascent at 3 limits

**Checks**

- [ ] Full literature check (the truth check): first pass done. No paper breaks the full claim, but only the rolling-budget combination is new. 12 checks remain before submission
- [ ] Final claims on test seeds never used before
- [ ] Confidence intervals across seeds, for the paper

**Real robot (done by hand, from Monday 9 November)**

- [ ] Sort out the false 150 °C servo reading, which stops the scripts
- [ ] Run the 15-minute robot test and weigh the robot
- [ ] Update the simulator with the real numbers, and rerun the comparisons if they change enough (about 3 days), by about 16 November
- [ ] Run the policy in stages: hanging, then supported, then free walking, between 12 and 30 November
- [ ] Final trials at loose and mid budgets: about 10 tries per setting, with logs and slow-motion video, reporting every failure too. Aim for 1 December, with about 15 December as the backup

**Paper**

- [ ] Write the paper and make the figures, alongside testing
- [ ] Submit, in mid to late December

## Risks and weak spots

The real robot is the biggest risk, and finishing all testing by 1 December is roughly an even chance by my judgement.

| Risk or weak spot | Why it matters | Now |
| --- | --- | --- |
| Real robot transfer | The policy takes about 6 very short steps per second, about 3.5 cm long, with about 10 mm of foot lift. The simulator does not model servo acceleration, so real servos may not keep up | Not tested. Each fix cycle takes about a day |
| Comparisons | Budgets have not yet faced the other methods. If they do not win, the paper becomes an analysis of when and why hindsight fails, still aimed at a Q1 journal | Runs start once the procedure is frozen |
| Novelty | Budget relabeling alone is not new: SeqRoute does it, offline, for routing chat requests. The claim rests on the specific combination for legged walking | Full check running |
| Reruns after robot measurements | If the real numbers change the simulator enough, the comparisons must run again | About 3 days |
| New robot design | Changing weight or geometry means updating the model and retraining | Keep the design fixed unless the new one is final |
| Pushes | Loose budget over 30 s with pushes: upright 18, 23, 21, 7 of 30. Seed 7 got worse than at 4.8 million steps | Open |
| Tight limits | Speed is close to target, but the strict check passes only 15, 14, 0, 8 of 100, because slip runs above its rate | Open |
| Jitter | The policy's output still changes by about 0.6 per step on a -1 to 1 scale | Two fixes queued |
| Seed 3 at loose budget | Only 30 of 100 alternating walks, though it passes the goal | Open |
| Shared machine | Another job uses about 11 of the 20 CPU cores, so run times vary | Dates may move |
