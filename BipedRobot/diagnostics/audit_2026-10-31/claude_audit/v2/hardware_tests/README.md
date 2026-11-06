# Testing on the real robot

None of this has run on real servos yet. Do the steps in order, and send the results of each one before moving on.

**Every time:** stop `servo_bridge` first, because only one program can use the servos. Keep a hand on the power switch. Ctrl-C or any problem turns torque off and the legs drop.

## 1. Measure the robot (about 15 minutes)

Hang the robot so both feet are off the ground, and have a kitchen scale ready. Copy `robot_test.py` to the Pi, then run:

```bash
python3 ~/robot_test.py all --biped-ws ~/biped_ws
```

If it says there is no calibration, run `biped_calibrate` first. It asks you to confirm each joint's direction by eye. Send the `.tar.gz` it makes.

These numbers replace guesses in the simulator: timing, sensor noise, servo response, joint directions and weight.

## 2. Find how the IMU is mounted (2 minutes, no torque)

Wait until step 1 has passed and a trained policy exists. Make the policy file on the workstation:

```bash
python export_policy.py CHECKPOINT.pt resolved_config.yaml policy.npz
```

It refuses a policy that reads true speed or budget balances, because the robot cannot measure them. Copy `run_policy.py`, `robot_test.py` and `policy.npz` to the same folder on the Pi. Then hold the robot in your hands and run:

```bash
python3 run_policy.py policy.npz --stage imu --biped-ws ~/biped_ws
```

It asks you to tilt the robot and then prints the `--imu-mount` to use.

## 3. Run the policy, one stage at a time

Stay at a stage until it looks right. Send the `.tar.gz` after each run.

| Stage | Robot | Torque limit | Actions | Time |
|---|---|---|---|---|
| `hang` | Both feet off the ground | 500 | 30% | 20 s |
| `supported` | Feet touching, held by a harness or a person | 700 | Full | 15 s |
| `ground` | Harness slack, ready to catch | 1000 | Full | 15 s |

```bash
python3 run_policy.py policy.npz --stage hang --speed 0.2 --budget 1,1,1 --imu-mount x,y,z --biped-ws ~/biped_ws
```

- `hardware.yaml` can lower the torque limit. The script says so if it does. The policy was trained with the full limit, so tell me if it does.
- `--budget` sets flight, slip and torque, from 0 (tightest) to 1 (loosest).
- `--hold-heading` steers back to the starting direction using the IMU. It assumes the IMU's z axis is close to vertical.
- Don't use `sudo` for now. It can change where the script finds your calibration and Python packages. The run works without it, and `summary.json` shows how many loops ran late.

## What stops a run

A servo or the IMU not answering, heat, low voltage, tilt over 45 degrees, a joint far from its target for 0.5 s, late control loops, Ctrl-C, or the SSH link dropping. Torque goes off, so the robot must always be hanging or in a harness.

## Testing without a robot

Add `--mock --yes` to run on pretend servos. The numbers mean nothing about the real robot. All tests: `BIPED_WS=/path/to/biped_ws tests/run_all_tests.sh`.
