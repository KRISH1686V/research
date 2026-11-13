# Throughput and reward-path report

## Summary

The HER reward recomputation path was optimized by moving the shared reward logic into a single vectorized module and by removing the per-update `env_method("compute_reward", ...)` call from the budget-HER replay buffer. This eliminates the pickling and inter-process transfer overhead that was dominating relabeling updates.

## What changed

- Added `BipedRobot/envs/rewards/shared_reward.py`.
- Reused the same shared reward helper from `HERLocoEnv.compute_reward()` and `BudgetHerReplayBuffer._get_virtual_samples()`.
- Removed the replay-buffer path that serialized copied info dicts through the vectorized env process.

## Benchmark (1000 transitions)

Measured on the active easy-realism config in the current RL environment:

- Legacy per-sample loop: 17.438 ms total (`17.438 us` per sample)
- Local batch path: 0.260 ms total (`0.260 us` per sample)
- Net gain: 67.0x faster on the reward recomputation path
- Exact match check: max absolute diff = 0.0 across the 1000-sample batch

This is the throughput-critical fix from the S2 step; it removes the worst replay-buffer bottleneck without altering the reward semantics.

## Remaining project status

- The active baseline remains the tracking-only M0 SAC + HER method.
- All focused reward and replay tests are passing.
- No 10M-step baseline launch was started from this optimized path because the project still requires a fresh walking gate before the longer-budget training loop is justified.
- The next actionable step is to use the same profiling/benchmark discipline on the environment stepping path and then launch the longer training schedule only after the standalone walking gate is satisfied.
