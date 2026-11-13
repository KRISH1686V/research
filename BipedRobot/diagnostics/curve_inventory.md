# SAC/HER curve inventory and recovery diagnosis

Date: 2026-10-28

This inventory uses saved TensorBoard scalars and the fresh standalone R0
evaluations. TensorBoard values are training diagnostics only; gates use fresh
standalone evaluation under the saved run configuration.

## Comparable SB3 HER curves

| Run | Steps | Latest rollout length | Latest `eval/ep_len_zero` | Latest `eval/ep_len_02` | Latest RMSE vx | Latest torque saturation | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `her_goal_tracking_v2` seed 42 | 1,000,000 | 455.2 | 500 | 500 | 0.0874 | 0.458 | Best historical TensorBoard curve, not reproducible standalone |
| `recovery_m0_seed0_v2` seed 0 | ~140,800 | 404.2 | 500 | 153 | 0.2856 | 0.698 | Upright behavior improving, forward command not learned yet |
| `recovery_m0_seed1_v2` seed 1 | ~140,700 | 339.5 | 500 | 414.6 | 0.2567 | 0.413 | Better command tracking than seed 0, still below gate |
| `stageA1_walk_safe` seed 42 | 1,500,000 | 355.1 | 500 | 314.6 | 0.0837 | 0.0035 | Stronger actuator regularization, inconsistent walking survival |
| `her_loco_1_5m` seed 42 | 1,500,000 | see saved event file | see saved event file | see saved event file | see saved event file | see saved event file | Earlier HER experiment; not used as current baseline |

The current M0 runs have not reached the mandated 300k decision point. They
remain active. No conclusion about their final gate should be made from these
early points.

## R0 standalone evidence

The prior `her_goal_tracking_v2` final checkpoint failed the fresh frozen
evaluation:

- Original frozen settings (`range_fraction=0.80`, thresholds `0.14/0.58`):
  3/10 survival at 0.20 m/s, RMSE 0.1363.
- The 800k checkpoint selected from the TensorBoard curve:
  2/10 survival, RMSE 0.1488.
- The best tested configuration row reached 8/10 survival but RMSE 0.1637,
  so it still failed the required RMSE threshold.
- The callback itself reported low RMSE, but only over five seeds and with mean
  episode lengths 338.0 (800k) and 236.4 (final). Its low RMSE was therefore
  not a valid walking gate.

Raw evidence:

- `paper_results/r0_all_checkpoints.json`
- `paper_results/r0_config_matrix.json`
- `paper_results/r0_callback_eval.json`
- `paper_results/baseline_frozen_eval.json`

## Why training takes so long

The active trainer uses:

- MuJoCo physics with actuator delay, lag, backlash, and per-step metrics.
- Eight subprocess environments.
- SB3 SAC with `gradient_steps=-1`, which performs approximately one gradient
  update per environment transition batch.
- Repeated 500-step callback evaluations.

The observed throughput is approximately 38 environment steps/s for the M0
runs. At that rate, 1.5M steps is approximately 10.9 hours per seed. Two runs
are concurrent, so the three-seed baseline is approximately 20–22 hours plus
evaluation.

The older author SAC runs are not a valid speed comparison: they use the
separate custom `src/mujoco` trainer and different observation/reward/model
path. Mixing their curves with SB3 HER would invalidate the paper comparison.

## Root-cause diagnosis

1. **The historical gate was an evaluation/provenance failure.** The
   TensorBoard callback used five seeds and a mutable evaluation environment;
   the standalone evaluator uses fixed seeds and the saved configuration.
2. **The policy is not yet command-robust.** Rollout episode length can improve
   while `eval/ep_len_02` remains low. Standing/survival and forward tracking
   are separate capabilities in this environment.
3. **The active recovery config is stricter than the irreproducible run.**
   Recovery M0 uses the Stage-A specification: action fraction 0.50 and
   termination thresholds 0.15/0.65. The prior HER run used 0.80 and 0.14/0.58
   in the workspace at the time it trained. This difference is intentional and
   recorded, not silently hidden.
4. **Longer training is not automatically the fix.** The old run's final
   checkpoint was worse than some intermediate checkpoints in the R0 sweep.
   The hardened pipeline will report final checkpoints and learning curves,
   rather than selecting a flattering checkpoint after the fact.

## Paper-oriented path

The project is currently in baseline recovery, not paper-result generation.
The correct order is:

1. Finish the three M0 seeds with provenance.
2. Gate their final checkpoints using fresh standalone evaluation.
3. If at least two pass, calibrate costs from a passing final checkpoint and
   request authorization before C1.
4. If they fail, stop spending GPU time on budget training and move to a
   Phase-1 diagnosis/ablation: command curriculum, action fraction, reward
   scale, and SAC update ratio, each with a new run and fixed evaluation.

No budget training should be used to conceal a failing tracking baseline.
