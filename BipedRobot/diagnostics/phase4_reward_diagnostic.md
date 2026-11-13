# Phase 4 reward diagnostic

This diagnostic is read-only with respect to existing checkpoints. No training
run was launched.

## New decomposition

For the pilot configuration, the per-step reward is:

```text
tracking(achieved_goal, desired_goal)
+ 0.30 * clip(achieved_vx / desired_vx, 0, 1)       [desired_vx > 0.1]
+ r_smooth
+ r_flight
+ r_gait_move                                      [||desired_xy|| > 0.1]
+ r_gait_still                                     [otherwise]
```

The gait and smoothness components are computed during the environment step,
stored in `info`, and reconstructed by `compute_reward()` for HER relabeling.
Existing checkpoints were evaluated without modifying their files.

## Phase 0 exact values

1. Tracking reward:

```text
r_track =
  exp(-(||v_xy - v_xy*||^2) / (2 * sigma_v^2))
  * exp(-(wz - wz*)^2 / (2 * sigma_w^2))
```

The kernels use the `2 * sigma^2` Gaussian denominator and are multiplied.
The frozen values are `sigma_v = 0.15` and `sigma_w = 0.30`.

2. The maximum tracking reward is `1.0`, at exact velocity match.

3. Command ranges are `vx = [-0.2, 0.6]`, `vy = [-0.2, 0.2]`,
   `wz = [-0.5, 0.5]`.

4. Curriculum starts at `curriculum_scale = 1.0`. It advances when the
   callback has 30 completed nonzero-command episodes and at least 70% have
   mean tracking reward >= 0.6 without a fall. Each advance multiplies the
   scale by `1.15`. There is no fixed time-based progression speed: advancement
   is episode-driven and checked every environment callback step.

5. The achieved-goal window is 25 control steps, or 0.5 seconds at
   `control_dt = 0.02`.

6. HER uses `n_sampled_goal = 4` in the frozen baseline,
   `goal_selection_strategy = FUTURE`, and `copy_info_dict = True`.
   The proposed pilot configs use `n_sampled_goal = 2`.

7. Control timing is `dt = 0.02 s`, `frame_skip = 4`, maximum episode length
   `500` control steps, and `gamma = 0.99`.

8. Installed SB3 is `2.9.0`. HER calls
   `self.env.env_method("compute_reward", ...)`, targeting `indices=[0]`,
   rather than calling the environment method directly. It passes
   `next_obs["achieved_goal"]`, the relabeled desired goal, and copied info
   dictionaries. Fall status is read from `info["fell"]`.

## Phase 4 table

Means are over 10 fixed seeds (`1000-1009`) at each command. The reference is
an open-loop sinusoidal joint-action rollout with reset-velocity assistance; it
did not sustain forward walking, so its ratio is a diagnostic comparison, not
a walking claim.

| Command | Policy/reference | Tracking | Progress | Smooth | Flight | Gait | Total | Mean vx | Survival | Standing total / reference total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | stageA0_stand 200k | 0.2453 | 0.0445 | -0.0988 | -0.0235 | 0.1034 | 0.0230 | -0.0219 | 0.0 | -0.070 |
| 0.2 | stageA0_stand_safe 200k | 0.3407 | 0.0818 | -0.1123 | -0.0195 | 0.0655 | 0.1265 | 0.0287 | 0.0 | -0.385 |
| 0.2 | recovery seed0 1100k | 0.3845 | 0.0274 | -0.0345 | -0.0098 | -0.0150 | 0.3527 | 0.0046 | 1.0 | -1.075 |
| 0.2 | recovery seed1 1100k | 0.4119 | 0.0292 | -0.0275 | -0.0056 | -0.0010 | 0.3791 | 0.0182 | 0.9 | -1.155 |
| 0.2 | scripted reference | 0.0880 | 0.0524 | -0.1035 | -0.0517 | 0.0504 | -0.3281 | -0.0582 | 0.0 | — |
| 0.4 | stageA0_stand 200k | 0.0300 | 0.0234 | -0.1030 | -0.0238 | 0.0980 | -0.3229 | -0.0637 | 0.0 | 0.715 |
| 0.4 | stageA0_stand_safe 200k | 0.0614 | 0.0744 | -0.2089 | -0.0212 | 0.0798 | -0.3984 | 0.0575 | 0.0 | 0.882 |
| 0.4 | recovery seed0 1100k | 0.0604 | 0.0872 | -0.1527 | -0.0312 | 0.0389 | -0.2909 | 0.1047 | 0.0 | 0.644 |
| 0.4 | recovery seed1 1100k | 0.0350 | 0.0516 | -0.1761 | -0.0319 | 0.0931 | -0.4290 | 0.0299 | 0.0 | 0.950 |
| 0.4 | scripted reference | 0.0307 | 0.0629 | -0.1196 | -0.0423 | 0.0364 | -0.4515 | 0.0197 | 0.0 | — |

Two ratios exceed the amended `0.85` stop threshold at `0.4 m/s`
(`stageA0_stand_safe = 0.882`, `recovery seed1 = 0.950`). Because the
scripted reference itself failed to walk and had negative total reward, these
ratios are not evidence that standing is equivalent to walking. They do,
however, trigger the requested stop condition and show that the diagnostic
reference needs to be replaced by a genuinely forward-moving reference before
pilot weight calibration is considered final.

## Proposed pilot weights

These are starting values, not final calibrated values:

- progress: `0.30`
- air-time touchdown: `3.0 * clip(t_air - 0.25, -0.1, 0.15)`
- single support: `0.05` per step
- swing clearance: `0.10 * min(peak / 0.05, 1)`
- flight: `-0.50` per step
- zero-command touchdown: `-0.30`
- action rate: `0.05` after ramping from zero
- action second difference: `0.002`
- base vertical velocity and roll/pitch rate: `0.05`
- action filter alpha: `0.60`
- reset velocity probability: `0.50`

The pilot curriculum proposes `p_zero_cmd = 0.20` and initial `vx` in
`[0.15, 0.35]`; thus 80% of commands are nonzero and all nonzero initial
commands are at least `0.15 m/s`.

## Pilot plan, not launched

- Run A: `pilot_gait_smooth_a_seed0_v1`, config
  `config/pilot_gait_smooth_a.yaml`, 1-2M steps, 8 environments.
- Run B: `pilot_gait_smooth_b_seed1_v1`, config
  `config/pilot_gait_smooth_b.yaml`, 1-2M steps, 8 environments.

Run B adds second-difference/posture shaping and requests SAC gSDE with
`sde_sample_freq = 4`. It must first pass a short compatibility smoke test
with HER before any pilot launch. Neither pilot is approved or started by this
task.
