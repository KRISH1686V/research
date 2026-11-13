# Robotis OP3: the second robot

The project's budget-as-goal method (SAC + hindsight relabeling of the command and of token-bucket gait budgets, round r17wS) on a second
12-DOF servo biped, so the paper can show that the method transfers. Simulation only.

## What it is

- **Robot.** Robotis OP3, a 3.15 kg (model) servo-driven humanoid with 6 joints per leg: hip yaw, hip roll, hip pitch, knee, ankle pitch, ankle roll,
  the same leg layout as our 1.66 kg STS3215 biped. The 12 leg joints are the action space. Arms and head are **welded** at a neutral pose (arms down
  about 19 deg out from vertical, elbows straight, head level): their 8 joints and actuators are removed and each body is fixed where its joint would hold
  it, keeping its mass and inertia. Welded rather than held by stiff servos because (1) the real OP3 holds them with position servos, and the arm
  deflection under walking loads at kp 21.1 is a few mrad; (2) the policy observes only the legs, so free arm joints would be state it cannot see; (3)
  the model then has exactly 12 actuators like our biped, so HERLocoEnv's actuator handling applies unchanged (with the arm actuators kept, mj_resetData
  would drive them to the T-pose unless the env re-commanded them every reset).
- **Model.** `envs/assets/robotis_op3/` is MuJoCo Menagerie's `robotis_op3`, unmodified (commit in `MENAGERIE_COMMIT.txt`, Apache-2.0
  `LICENSE` kept). `make_train_xml.py` writes `op3_train.xml` and `scene_train.xml` beside it: arms and head welded, leg actuators renamed after their
  joints, sole boxes named. Gains and limits are Menagerie's: position servos kp 21.1 N m/rad, force range +-5 N m, joint damping 1.084, armature
  0.045, frictionloss 0.03, 2 ms timestep, contact solref 4 ms. Each foot is two overlapping boxes; every sole box counts as that foot.
- **Env.** `op3_env.py`: `Op3BudgetEnv(Op3Mixin, BudgetLocoEnv)`, the biped's own classes (HERLocoEnv -> ValidLocoEnv -> GaitCostEnv -> BudgetLocoEnv) with an
  OP3 description: joint, foot and torso names (swapped into her_loco_env's module constants only while HERLocoEnv.__init__ runs; her_loco_env.py is
  not changed), multi-box feet, the standing pose, and an action map. 50 Hz control, 10 physics substeps of 2 ms.
- **Actions (policy coordinates).** In the OP3's joint conventions a left-right symmetric pose has opposite angles on the two legs for all 12 leg joints,
  so the right leg's actions are negated: target = standing pose + side * a * scale, with scale (rad, per leg) hip yaw 0.3, hip pitch 0.8, hip roll 0.3,
  knee 0.8, ankle pitch 0.6, ankle roll 0.3 about the standing pose (hips 0.4, knees 0.8, ankles 0.4 rad; the knee reaches straight and 1.6 rad, never
  backwards). One action value means the same motion on either leg, so the mirror of an action is a pure swap of the legs, as the replay's mirror
  augmentation does.
- **Mirror maps.** `mirror.py`, same layout and signatures as `fast/mirror.py`, which it reuses: legs swapped and every joint angle and joint speed
  negated; command (vx, -vy, -wz), gyro (-wx, wy, -wz), gravity (gx, -gy, gz); budgets and levels unchanged. Checked against the simulator (largest
  observation error 0.0009 over 0.1 s, tests/test_op3_env.py).
- **Costs.** flight, slip, jshake as for the biped (fast/gait_costs.py, fast/budget_env.py): flight = fraction of physics substeps with no foot over
  0.5 N and no other floor load (in tenths of a step here, quarters on the biped); slip = mean over loaded foot-substeps of the force-weighted horizontal
  contact-point speed (m/s); jshake = mean over the 12 joints of the squared second difference of the true joint angles (rad^2). **One difference, on
  purpose:** the biped's code takes the foot's velocity at its centre of mass (mj_objectVelocity with mjOBJ_BODY reports it there) and adds the lever arm
  from the body's frame origin, so its "slip" is the speed of a point offset from the contact by the foot's centre-of-mass offset (26 mm on V2), not the
  contact point's (on V2 under random actions the mean contact speed reads 1.35x the true one; on the OP3 2.9x). The OP3 uses the contact point's speed
  (equal to mj_jac at the contact to machine precision); `env.op3.slip_reference: biped_code` reproduces the biped's arithmetic exactly.
- **Reward.** No gait reference, no gait clock, no gait shaping: tracking with r17wS's command-relative kernel (reward.track_rel; with the Gaussian
  sigma 0.15, standing still at 0.1 m/s would keep 0.80 of tracking), progress 0.3, base motion 0.05 (vz^2 + wx^2 + wy^2), torso tilt cost past 8 deg,
  alive bonus 0.2, fall -25. Action filter alpha 0.3 (5 Nov benchmark).
- **Realism.** As the biped's stack where it applies: friction 0.4-1.0, torso mass +-15%, motor strength +-15% (kp and torque limit), delay 0-20 ms,
  6 N pushes for 0.1 s (0.19 m/s velocity change; the biped: 3 N on 1.66 kg, 0.18 m/s), torque-speed line from the XM430-W350 datasheet at 12 V (4.1 N m stall, 4.82 rad/s no load;
  not identified on an OP3), sensor noise and bias as the biped's, 4096-step encoder, 5 frames of history. Not ported (refused at construction): the
  biped's gait reference, clock and foot-mesh terms.
- **Command.** vx 0.10-0.25 m/s, vy = wz = 0, 15% standing (r17wS's layout). See "Command range" below.

## Command range

vx 0.10-0.25 m/s forward (vy = wz = 0, 15% standing, `her.relabel_dims [vx]`, as r17wS). Chosen by dynamic similarity with our biped and checked in this env:

- Leg length (hip pitch axis to ankle) is 0.220 m on the OP3 and 0.144 m on V2 (0.072 + 0.072 m), so equal Froude number v^2 / (g L) means
  v_OP3 = 1.24 v_V2. The OP3 range 0.10-0.25 m/s is Fr 0.005-0.029: the biped's r17wS command 0.08-0.12 m/s (Fr 0.005-0.010) at the low end, its 5 Nov
  benchmark's 0.2 m/s (Fr 0.028) at the top.
- Open-loop scripted walks in this env (diagnostics/op3/scripted_walk.py, 1.7-3.3 steps/s, 5-7.5 cm steps) walk 0.10-0.19 m/s steadily; patterns asking
  0.25 m/s reach only 0.21 m/s (easy and full realism). So 0.25 m/s is near what this servo model (4.82 rad/s no load) gives a simple gait,
  and the range stays where a walk without flight exists.

## Budget rates

`token_budget` in op3_budget_r17wS.yaml, per control step:

| cost | rate_min (u = 0) | rate_max (u = 1) | u = -0.3 (u_low) |
|---|---|---|---|
| flight (share of 2 ms substeps) | 0.001 | 0.1 | 0.00025 |
| slip (m/s, contact point) | 0.002 | 0.05 | 0.00076 |
| jshake (rad^2) | 7e-6 | 1.5e-4 | 2.8e-6 |

Measured before any training on the OP3 env itself (outputs in diagnostics/op3, summary by `summarize_costs.py`): 24 open-loop walking patterns
(period 0.6-1.0 s, 0.1-0.25 m/s, lift 25-35 mm, sway 25 mm) x 3 seeds x 10 s at easy and at full realism, with action noise of std 0 / 0.1 / 0.2 (as
SAC's sampled actions); standing still; uniform random actions (what training starts with). With window = burst = 100 a bucket is overdrawn only when
some T <= 100 steps cost more than r (T + 100), so a steady cost binds above 2 r:

- u = 0 binds: 2 r_min is below the slip of every measured walk (lowest 0.0046 m/s) and below its jshake (lowest 1.68e-5), and below the median flight
  of the walks at 0.15 m/s and faster (0.010-0.10); only the cleanest walks at 0.10-0.15 m/s spend less (0-0.002). Flight's 0.001 is the lowest rate whose bucket (0.1) still holds one 2 ms flight substep, so
  u = 0 does not punish a single event; u < 0 does. No capacity floor (that variant is still being tested on the biped).
- u = 1 is free: 2 r_max is above every measured 100-step window (flight 0.175, slip 0.050, jshake 5.9e-5 at their worst) and above random exploration
  (0.063, 0.020, 8.7e-5 at full realism).
- Ratios r_max / r_min 100, 25, 21 (the biped's r17wS: 50, 25, 6).
- Standing still costs flight 0, slip 1e-4, jshake 4e-8, inside every budget: tight budgets are always satisfiable by standing, which costs tracking reward.

Measured with the open-loop pattern, not a trained policy: a trained policy's costs (logged as cost_* in progress.csv) should be checked against these
ranges early in the first GPU run.

## How to train

Configs: `config/op3_budget_r17wS.yaml` (budgets on, r17wS settings) and `config/op3_budget_off.yaml` (budgets off, the c17plain comparison; only
`token_budget` differs). `env.robot_type: op3` makes `fast/collector.env_class` build the OP3 classes and `fast/train.py` use the OP3 mirror maps and hash
the OP3 files; biped configs do not have the key, so their path is unchanged. Same launch as the biped (run_budget_deploy.sh flags):

    cd /home/rpuser1/yasho/rl/BipedRobot
    python fast/train.py --config config/op3_budget_r17wS.yaml --exp_name fast_deploylong7mop3r17wS_20261102_seed11 --seed 11 --n_envs 8 \
      --total_steps 7200000 --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints --plain_ckpt --difficulty_ramp 0.26666666666666666 \
      --layer_norm --avg_q --ema_tau 0.002 --asym --mirror --save_every 400000 --actor_hide achieved,levels --save_critic

Observation 222 numbers (3 achieved + 6 goal + 5 x 42 frames + 3 levels; the biped's 225 add its gait clock and slow-yaw input), critic-only state 46. The biped's evaluation
tools (survive_eval.py and others) assume the biped's model and have not been ported.

## Files

- `envs/op3/op3_env.py`, `envs/op3/mirror.py`, `envs/op3/make_train_xml.py`, this README
- `envs/assets/robotis_op3/` (Menagerie, unmodified) plus generated `op3_train.xml`, `scene_train.xml`
- `config/op3_budget_r17wS.yaml`, `config/op3_budget_off.yaml`
- `tests/test_op3_env.py`
- `diagnostics/op3/scripted_walk.py`, `diagnostics/op3/random_costs.py` and their outputs (cost measurements behind the rates)
