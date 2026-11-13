# Stage 0 and controller report

Date: 2026-10-29

## Status

The offline reference gate is **blocked**. No SAC pilot, training run, or
existing checkpoint was modified.

The Stage 0 measurements used the A-derived configuration with domain
randomization disabled, pushes disabled, actuator delay/lag/backlash retained,
and action range fraction 0.50. The measured single-process environment
throughput was 1,530.5 environment steps/s.

## Stage 0 model facts

| Quantity | Value |
|---|---:|
| Total model mass | 1.4680 kg |
| CoM height after reset | 0.1643 m |
| Leg length | 0.2386 m |
| Maximum geometric step length, 2 x leg length | 0.4772 m |
| Foot size, right | 0.0395 x 0.0604 x 0.0755 m |
| Foot size, left | 0.0391 x 0.0606 x 0.0755 m |
| Torque limit | 2.94 N m per actuator |
| Action range fraction | 0.50 |
| Hip-pitch target swing range | 1.1781 rad |

Joint limits in actuator order are: r_hip_z [-1.5708, 0.6981], r_hip_x
[-2.0071, 2.0071], r_hip_y [-1.5708, 0.7854], r_knee [0, 2.3562],
r_ankle_y [-1.9199, 1.9199], r_ankle_x [-1.5708, 0.4363], l_hip_z
[-0.6981, 1.5708], l_hip_x [-2.0071, 2.0071], l_hip_y [-0.7854, 1.5708],
l_knee [0, 2.3562], l_ankle_y [-1.9199, 1.9199], and l_ankle_x
[-0.4363, 1.5708] radians.

Action mapping is asymmetric: for positive action a, target =
default_qpos + a * 0.50 * (joint_max - default_qpos); for negative action,
target = default_qpos + a * 0.50 * (default_qpos - joint_min). Action values
are clipped to [-1, 1].

The geometric step-length calculation is an upper bound, not a dynamic
guarantee. At 1.5, 2.0, and 2.5 steps/s, required step lengths are:

| Command | 1.5 steps/s | 2.0 steps/s | 2.5 steps/s | Froude |
|---:|---:|---:|---:|---:|
| 0.2 m/s | 0.1333 m | 0.1000 m | 0.0800 m | 0.0171 |
| 0.4 m/s | 0.2667 m | 0.2000 m | 0.1600 m | 0.0684 |
| 0.6 m/s | 0.4000 m | 0.3000 m | 0.2400 m | 0.1538 |

All are geometrically below 0.4772 m. The current controller, not geometry,
is the limiting factor.

## Actuator frequency response

With DR off and delay/lag/backlash retained, hip-pitch sinusoidal action
response was:

| Frequency | Action 0.5 amplitude | Action 1.0 amplitude |
|---:|---:|---:|
| 0.5 Hz | 0.2363 rad, -103.4 ms | 0.3509 rad, -103.5 ms |
| 1.0 Hz | 0.2296 rad, -73.3 ms | 0.3287 rad, -97.4 ms |
| 1.5 Hz | 0.2134 rad, -83.1 ms | 0.2449 rad, -125.3 ms |
| 2.0 Hz | 0.1737 rad, -98.8 ms | 0.1822 rad, -125.9 ms |
| 3.0 Hz | 0.1093 rad, -100.2 ms | 0.1117 rad, -107.7 ms |

The pipeline materially caps swing amplitude at 1.5-3 Hz, especially at action
amplitude 1.0.

## Standing recovery and evaluation settings

The zero-action standing probe survived 500 steps for 0, 0.1, 0.5, and 1.0 N
one-step torso pushes; the observed maximum tilt was 0.0236 rad. These are
small impulse probes, not a complete basin-of-attraction measurement.

Evaluation config uses domain randomization enabled with friction [0.6, 1.4],
mass [0.90, 1.10], motor strength [0.85, 1.15], easy realism with pushes
disabled. Actuator delay is sampled [0.010, 0.030] s around nominal 0.020 s,
servo lag tau is 0.020 s, and backlash deadband is 0.005 rad.

## Controller iteration results

The closed-loop phase/state-machine controller is in
`BipedRobot/reference/walking_controller.py`. It uses simulator contacts and
torso velocity, swing-foot targets, damped least-squares IK on scratch
MjData, asymmetric action inversion, and saturation logging.

Iteration 1, DR off, seeds 1000-1002:

| Command | Survival | Mean vx | Steady RMSE | Double support | Peak swing height |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 3/3 | ~0.000 m/s | ~0.202 m/s | 0.994-1.000 | ~0.024 m |
| 0.4 | 3/3 | ~0.002 m/s | ~0.401 m/s | 0.990-1.000 | ~0.024 m |
| 0.6 | 3/3 | ~0.012 m/s | ~0.593 m/s | 0.990-1.000 | ~0.025 m |

Iteration 2 changed frequency to 2.0 Hz, duty factor to 0.50, swing height
to 0.05 m, step-length gain to 1.5, velocity gain to 0.5, IK damping to
0.01, and IK iterations to 20. It produced the same standing behavior,
indicating the target-to-action path needs deeper IK/model-frame debugging.

The time-box condition is met: after two design iterations the controller
cannot hold 0.15 m/s at the 0.2 command with DR off. The work stops here as
required. The saved raw Stage 0 output is
`BipedRobot/paper_results/reference_stage0.json`; the first controller
evaluation is `BipedRobot/paper_results/closed_loop_reference_dr_off.json`.

Recommendation: do not launch the pilots. Fix the IK frame/Jacobian/action
mapping path and rerun the DR-off controller gate first.
