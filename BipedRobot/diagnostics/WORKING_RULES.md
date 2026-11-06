# Working rules

Set on 3 November 2026 at the owner's request, to keep the work focused. These sit on top of the owner's own rules, which still apply: protected files, no commits, PID 1604315 untouched, at most 4 runs at once, unsupported metrics are NOT DETERMINED, and no claims about real-world walking or paper readiness.

## The one goal

A policy that uses only what the robot can measure and no gait rewards, and passes this on at least 3 of 5 seeds fixed in advance:

| Measure, at command 0.2 m/s, medium realism, 100 new test seeds | Pass |
|---|---|
| Stays up | At least 90% at loose and mid budgets |
| Steady speed at a loose budget | 0.15 to 0.25 m/s |
| Alternating walk at a mid budget | At least 50% of episodes |
| Stays up, never over budget, and moves at 0.10 m/s or more, at a mid budget | At least 50% of episodes |

The same setup with budgets off must then be clearly worse. These measures don't change once results are in.

## Rules

1. **Work on one blocker at a time.** Every run must answer a question about the current blocker. If it doesn't, it waits.
2. **Every seed counts.** Averages and headlines use all seeds trained. A good seed may be used for a demo or a diagnosis, but never as the result.
3. **Say how strong the evidence is.** One or two seeds is "an early sign". Three is "a result so far". Only five supports a conclusion. No "big result" before then.
4. **Test on new seeds.** A claim needs test seeds that played no part in choosing the change, and the report says which.
5. **One change at a time, against a paired baseline.** Combine changes only after each has worked alone.
6. **Each idea gets two batches.** If an idea doesn't move the blocker's measure on 3 seeds by then, drop it and record why.
7. **Run the cheap test first.** Use existing checkpoints before spending a training batch.
8. **Check the code before believing a surprise.** When results jump either way, rerun the regression check.
9. **No side work.** No new features, tools or experiments unless they serve the current blocker. Keep the parked list parked.
10. **Simulator only for now.** On 4 November the owner chose to perfect the procedure first and start the physical robot on Monday 9 November. No robot steps, fixes or reminders until they raise it. Don't tune the simulator to guesses about the robot.
11. **Keep the GPU useful.** Leave a queue running, ordered by the current blocker.
12. **7 November is real.** If the goal isn't met by then, stop tuning and report what the journal paper still needs. The owner's only target is a Q1 journal, so don't propose a conference.
13. **Correct myself first.** Each report starts with any earlier claim that turned out wrong.
14. **Check before changing the queue.** Confirm the wait list isn't empty before starting a new queue, and that the run names are free.

## Current blocker

Human-like walking, at the owner's request (6 November): about 2 steps per second at the same speed, the swing foot lifted clear of the floor instead of rubbing it, and a full foot on the ground while the other is in the air. The fixed goal is met in the simulator at 7.2M on all 5 seeds, with budgets off clearly worse; the robot still takes about 6 short, low steps per second.

Targets, set before any results, at the style setting the arm trains for (on 100 new test seeds for survival and speed, and the gait report for the rest):
- 1.5 to 2.5 real steps per second
- steady speed 0.15 to 0.25 m/s at command 0.2
- stays up at least 90%
- median swing lift at least 20 mm
- both feet off the ground at most 1% of the time (walking has no flight phase)
- contact chatter at most 0.5 per second
- full-foot stance: NOT DETERMINED until it is measured

Arms, one change each against plain 7.2M: a step budget (seeds 3, 4 first) and a low-step budget (lift under 20 mm, seeds 3, 4 first). Combine them only after each works alone (rule 5). No gait rewards.

## Log of ideas

| Idea | Result | Status |
|---|---|---|
| 2-second bucket | No walking gain on its own | Kept as the budget design |
| Tighter speed reward (sigma_v 0.075) | Worse on 2 seeds, with alpha 0.6 | Dropped |
| Action filter alpha 0.3 | Walks on 2 of 5 seeds; budgets off never walks on 4 seeds | Kept, not reliable yet |
| 10 frames of memory | Walks less on 2 seeds | First batch, not promising |
| Progress weight 1.0 | Seeds 5, 6: alternating walk up from 19, 14 to 94, 76 of 270, but in big circles (1.3 to 1.6 rad in 10 s). Seeds 3, 4: 117, 93 of 270, no better than the filter alone | Dropped after two batches |
| Slow yaw-rate input with progress 1.0 | Stopped at 0.7M, no result. It was aimed at the circling from progress 1.0 | Dropped with progress 1.0 |
| Tighter yaw reward (sigma_w 0.075) with progress 1.0 | Never run, same reason | Dropped with progress 1.0 |
| Longer training, 4.8M steps | All 5 seeds walk at mid; 2 of 5 pass the goal on 100 new seeds. Also stretches the difficulty ramp, so the two are not separated | Kept |
| Averaging the last 4 checkpoints | 30 episodes at loose: seed 3 still 25 of 30, no clear gain | Not a fix for loose |
| Longer training, 7.2M steps, ramp still ending at 1.92M | Scan of 2.4M to 4.8M: mid walking 2 to 159 of 200 and still rising. At 7.2M, 4 of 4 seeds pass the walking measures, on seeds 10000-10099. Only steps changed. Pushes still weak, and seed 7 fell more at full realism | Kept |
| Half of training episodes also at the lab floor's friction, 0.3 to 0.4 (7.2M) | Never run. The owner can change the floor or feet on the real robot, so the lab floor is not a constraint | Dropped 4 Nov at the owner's request |
| Stronger action filter, alpha 0.2, 7.2M | Jitter is unpenalised and alpha 0.6 to 0.3 brought walking. Seeds 3, 4, 7: mid walking 67, 59, 73 against 86, 100, 95; step rate unchanged | First batch, not adopted |
| Heading hold from the gyro alone | All 5 seeds, seeds 10200-10299: turning 0.25-0.52 down to 0.11-0.23 rad. Upright 92.8 to 91.2 at loose, 98.4 to 96.8 at mid, mostly seed 3. No training | Kept as an option, reported beside the goal, not part of it |
| Policy smoothness loss (CAPS), 7.2M, weights 0.03 and 0.14 | Measured: output moves about 2 per step and 0.44-0.51 from input noise of 0.01, while the critic barely cares (dQ/da 0.4-0.6). A training loss, not a reward. Seeds 3, 4, 7: output change 0.42-0.46 against 0.60-0.64, but seed 3 mid up 53 and mid walking fell; step rate unchanged | First batch, not adopted |
| Fourth budget on steps, 7.2M | Owner wants about 2 steps per second. Built and tested; running on seeds 3, 4, then 7, 5, 6 | Testing |
| Low-step budget, lift under 20 mm, 7.2M | Owner says the feet nearly rub the floor; lift is about 8-12 mm now. Running on seeds 3, 4 | Testing |
