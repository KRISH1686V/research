# Biped robot test, Tuesday 10 Nov

Nothing here has run on the real robot yet. Everything said about these policies comes from the simulator.

Do one step at a time, and go on only when the last one looked right. **Film every step from the side and the front, and send me the results folder after every run, including runs that stop early.** The logged walks are what let me measure your robot's real vibration and fix what fails.

## What's in this folder

| File | What it is | sha256 starts with |
|---|---|---|
| `policy_r11ve_s12.npz` | **Use this first, at `--speed 0.1`.** Trained with the designer's measured vibration and an exactly left-right symmetric policy. 28 of 29 targets; stayed up 100% at that vibration in the simulator | `79def090ec8a` |
| `policy_r11ve_s12_replay.npz` | Its recorded joint targets at 0.1, for Step 3 | `23b1793a37ba` |
| `policy_r11v_s14.npz` | **Backup, at `--speed 0.1`.** Trained with the vibration only. Faster (0.116 m/s), 27 of 29; also 100% at that vibration | `97c562a22917` |
| `policy_r11v_s14_replay.npz` | Its recorded joint targets at 0.1 | `8cd1e716883e` |
| `policy_r9rf_s12.npz` | **Third, at `--speed 0.05`.** All 29 targets in the simulator, but its feet chatter at the designer's vibration (73% stayed up) | `dd7b5b48031b` |
| `${P}_replay.npz` | Its recorded joint targets at 0.05 | `68d68f31b391` |
| `policy_r9rf_s11.npz` and its replay | Spare, at `--speed 0.05` | `c1f42d10bb49`, `9eb990a0d663` |
| `policy_r16m_s14.npz` | **Use this first, from 11 Nov afternoon, at `--speed 0.1`.** Trained for what your robot showed on 11 Nov: the gravity sensor reading up to 8 deg off, and servos at 70-100%. All 31 simulator targets, 0.103 m/s. With your robot's offset (6 deg forward, 5 deg right) it stays up 10 of 10 and drifts 3-4 cm in 10 s; r11v s14 stays up 8 of 10 and drifts 37 cm | `c1e94dd62fd9` |
| `policy_r16m_s14_replay.npz` | Its recorded joint targets at 0.1 | `eb32ca5c659d` |
| `policy_r16m_s11.npz` | Backup, same training, at `--speed 0.1`. All 31 targets, slower (0.088 m/s) | `10c91a8dea14` |
| `policy_r16m_s11_replay.npz` | Its recorded joint targets at 0.1 | `d2d2485759e3` |
| `policy_r14p_s12.npz` | **New 11 Nov, at `--speed 0.1`.** Trained with gear play inside the servo. In the simulator's robot-like test its joints shake about a third less than r11v s14 (0.71 against 1.09 deg), but it walks slower (0.06-0.08 m/s) | `eb63f22d02a3` |
| `policy_r14p_s12_replay.npz` | Its recorded joint targets at 0.1 | `94adcf3582cb` |
| `policy_r15s_s12.npz` | **New 11 Nov, at `--speed 0.1 --budget 1,1,1`.** Its third budget number means shake, not torque. Never give it 0 as the third number: in the simulator it falls | `8444794165f5` |
| `policy_r15s_s12_replay.npz` | Its recorded joint targets at 0.1 | `1318a65a19b4` |
| `run_policy.py` | Runs a policy at 50 Hz with safety stops. It needs `robot_test.py` beside it | |
| `robot_test.py` | Servo and IMU code | |
| `temp_watch.py` | Watches servo temperature | |

**Pick the policy once, then every command below uses it.** Type this in the Pi terminal before Step 1, and again in any new terminal:

```bash
P=policy_r11ve_s12; V=0.1
```

To switch to the backup: `P=policy_r11v_s14; V=0.1`. For the third: `P=policy_r9rf_s12; V=0.05`. Each replay file was recorded at its policy's own speed, so keep P and V together.

`run_policy.py` prints the policy's sha256 when it starts. Check it matches the table.

If a better policy comes out of tonight's training, I'll give you the new file name and sha. The steps stay the same.

## 12 Nov evening: softer servos (experiment, for the shivering)

`run_policy.py` and `robot_test.py` were updated on 12 Nov (sha256 starts `1f25c290761d` and `02021236fc50`). New options `--servo-p` and `--servo-d` set the servos' position gains **for this power-up only**: torque goes off first, a servo whose permanent-memory lock is open stops the run before anything is written, every value is read back, and switching the servos off and on brings back the stored gains (P 32, D 32). The policies were trained against simulated servos, so this is an experiment, not a sure fix.

Try it hanging first, feet off the ground, and compare the shake with the same run without the option:

```bash
cd ~/robot_test_10nov
python3 run_policy.py policy_r16m_s14.npz --stage hang --action-scale 0.5 --seconds 15 --speed 0.1 --budget 0.5,0.5,0.5 --servo-p 24 --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws
```

If it is calmer, try `--servo-p 20`, then the supported walk with the same option. After you switch the robot off and on, run `--stage readback` once to confirm the stored gains still read P 32, D 32.

## 11 Nov afternoon: the new policy

Same supported walk as this morning, with the new file. Harness on, feet on the floor, type GO:

```bash
cd ~/robot_test_10nov
python3 run_policy.py policy_r16m_s14.npz --stage supported --torque-limit 700 --action-scale 0.5 --seconds 15 --speed 0.1 --budget 0.5,0.5,0.5 --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws
```

If it walks the full 15 s, try `--action-scale 0.7`, then `--torque-limit 1000`. It has never been on the robot, so a 20 s hang (`--stage hang`, feet off the ground) first is the safe start.

## 11 Nov: shake comparison

Shake is what made it fall on 10 Nov, so first compare shake on the same charged battery (12.6 V), robot hanging, full motion, 15 s each:

```bash
for P in policy_r11v_s14 policy_r14p_s12; do python3 run_policy.py $P.npz --stage hang --torque-limit 1000 --action-scale 1 --seconds 15 --speed 0.1 --budget 0.5,0.5,0.5 --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws; done
```

Send both results folders. Then Step 4 (supported) with the one that shook less. For `policy_r15s_s12`, use `--budget 1,1,1`.

## Know this before you start

- **Stopping drops the legs.** Ctrl-C or any fault turns torque off at once, so keep a hand ready to catch the robot and a hand near the power switch.
- **It won't stop walking if you tell it 0.** It keeps walking at command 0, so give it clear floor ahead.
- **A sideways knock can tip it.** In the simulator about 3 N for 0.2 s knocks it over about 4 times in 10, and a steady 100 g sideways pull from a cable or harness tips it every time. Keep the harness slack and route cables from above.
- **Keep the fan off** for the first runs.
- **Use a fully charged battery.** The run stops below 11.1 V.

## Before you start

**1. Weigh the whole robot** with the battery on, and send me the number. The simulator's robot is 1,661 g. If yours is under about 1,500 g, the feet may chatter, so tell me before walking. If you're adding the 200 g, fit it first, centred and low, and leave it on for every step below.

**2. Copy this folder to the Pi** from the workstation, once it can see the Pi. Back up the old copy first:

```bash
ssh rpuser2@biped-pi.local "cp -r ~/biped_robot_test ~/biped_robot_test_backup_$(date +%Y%m%d)"
scp -r /home/rpuser1/biped_budget_her/robot_test_10nov rpuser2@biped-pi.local:~/
```

**3. The torque cap in `hardware.yaml` is now 1000** (it was 500; set on 10 Nov, the old file is kept as `hardware.yaml.bak_20261110`). The policies trained with full torque.

**4. Before every run:** stop `servo_bridge`, because only one program can use the servos. On the Pi, go to the folder:

```bash
cd ~/robot_test_10nov
```

Every command below uses `--clear-accel-cap --servo-accel 0`. The servo's acceleration cap comes back each time the power is switched off and on, so it's cleared at the start of every run. Leave out `--hold-heading`.

## Quick checks you said are done

If you did not actually run these, or if anything has changed since, run them now. They take a few minutes. If you ran the first and third, send me the output: the overload numbers and the delay are the two I've never seen.

```bash
# No torque: prints each servo's settings, including overload protection
python3 run_policy.py $P.npz --stage readback --clear-accel-cap --servo-accel 0 --biped-ws ~/biped_ws/src/biped_ws

# Robot hanging: checks joint directions in 3 poses. If a pose is wrong, answer n and stop
python3 run_policy.py $P.npz --stage poses --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws

# Robot hanging: measures the real loop delay. Tell me if it is over 40 ms before walking
python3 run_policy.py $P.npz --stage delay --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws
```

## The steps

**Step 1: hanging, 30% actions, 20 s.** Feet off the ground. The legs should step in a steady left-right rhythm, about 2 steps a second, both legs alike.

```bash
python3 run_policy.py $P.npz --stage hang --speed $V --budget 0.5,0.5,0.5 --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws
```

**Step 2: hanging again, full actions, 15 s.** Same setup, full motion. This shows the swing you'll see on the ground, with no risk of a fall.

```bash
python3 run_policy.py $P.npz --stage hang --torque-limit 1000 --action-scale 1 --seconds 15 --speed $V --budget 0.5,0.5,0.5 --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws
```

**Step 3: loaded replay, in the harness.** Feet on the floor, the harness holding the robot. It plays the recorded joint targets with no feedback from the policy. It prints how closely each joint followed. This is the test of whether the real knees can make this motion, so send me these numbers before going on.

```bash
python3 run_policy.py $P.npz --stage replay --replay-file ${P}_replay.npz --torque-limit 1000 --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws
```

**Step 4: supported walking, 15 s.** Feet on the floor, held by a harness or a person. It will walk forward at about 0.09-0.12 m/s, 1.4-1.8 m in 15 s, so give it room. It asks you to type GO.

```bash
python3 run_policy.py $P.npz --stage supported --torque-limit 1000 --speed $V --budget 0.5,0.5,0.5 --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws
```

**Step 5: on the ground, 15 s.** Harness slack and ready to catch. It asks you to type GO.

```bash
python3 run_policy.py $P.npz --stage ground --speed $V --budget 0.5,0.5,0.5 --clear-accel-cap --servo-accel 0 --imu-mount x,y,z --biped-ws ~/biped_ws/src/biped_ws
```

## If something goes wrong

| What you see | What it may be | What to do |
|---|---|---|
| Feet chatter or tap, the robot buzzes | Vibration, or the robot is light | Stop. Send the log, the video and the robot's weight |
| Steps cut short, "joint far from its target" | Knees too weak at this speed | Stop. Send the replay numbers from Step 3 |
| Torque cuts out mid-step | Servo overload protection | Stop. Run the readback and send the output |
| Fast, rushed steps | Loop delay over 40 ms | Run the delay stage and send the result |
| Feet drag or slide | Lift smaller than the simulator's | Send a side video |
| A foot pivots, the robot twists | Grip too low | Change the floor or the sole, and tell me what you changed |
| Stops at 11.1 V | Battery sag | Charge it. If it happens on a full battery, tell me |
| Stops "too hot" | Servos at 65 C | Let them cool. Tell me how long the run lasted. Don't raise the limit |
| Curves off to one side | Normal so far: 5-14 cm per 10 s in the simulator | Give it room and send the video |
| Anything else | | Send the files and a phone video, and stop there |

## What stops a run

Torque goes off if a servo or the IMU stops answering, a servo reaches 65 C, the supply drops below 11.1 V, the robot tilts more than 35 degrees, a joint stays far from its target for 0.5 s, control loops run late, or you press Ctrl-C or the SSH link drops.

Before torque comes on, it also refuses to start if any servo's stored gains are not P 32, D 32, I 0, or if the acceleration cap is still on after clearing. Neither script writes to the servos' permanent memory.

Don't use `sudo` for now. It can change where the script finds your calibration and Python packages.

## Trying the scripts without a robot

Add `--mock --yes` to `run_policy.py`. It runs on pretend servos, and the numbers mean nothing about the real robot.
