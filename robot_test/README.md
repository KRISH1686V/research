# Biped robot test files

These files measure the real robot, then run a trained policy on it.

**So far only the IMU check has run on the real robot, and it passed with mount `x,y,z`.** No policy has moved the robot. Only do Step 1 for now. Send me the results, and wait until I've checked them before Step 3.

## What's in this folder

| File | What it does |
|---|---|
| `robot_test.py` | Measures the robot: timing, sensor noise, servo response, joint directions and weight |
| `run_policy.py` | Runs a trained policy at 50 Hz, with safety cut-offs. It needs `robot_test.py` beside it |
| `policy_7m_s4.npz` | **Use this one first.** Trained for 7.2M steps: budgets, no gait rewards, stronger action filter, seed 4 |
| `policy_7m_s5.npz` | The same, seed 5. Slightly faster, and stays up slightly less with pushes |
| `policy_filter_s3.npz`, `policy_filter_s4.npz` | Older 3M policies. Keep them for comparison only |
| `README.md` | This guide |

**About these policies:** they're the best so far. In the simulator, at mid-level limits, seed 4 stayed up in 99 of 100 test episodes at 0.21 m/s and took alternating steps in 100. Seed 5 stayed up in 98 at 0.22 m/s, with alternating steps in 95. With pushes over 10 s at a loose budget, they stayed up in 27 and 25 of 30. The steps are still short and quick, about 6 per second and 3.5 cm long, and real servos may not manage them. They have not been shown to walk on the real robot.

## Before you start

**1. Copy this folder to the Pi.** Run this on your PC, with your own Pi username and address:

```powershell
scp -r C:\Users\krish\Downloads\biped_robot_test PI_USER@PI_ADDRESS:~/
```

**2. Check the Pi has what it needs:**
- Your code in `~/biped_ws`.
- A calibration file. If a script says there isn't one, run `biped_calibrate`.
- The Python packages. If a script says one is missing, install it:

```bash
pip install numpy pyserial pyyaml adafruit-circuitpython-bno08x
```

**3. Do this before every run:**
- Stop `servo_bridge`, because only one program can use the servos.
- Keep your hand on the power switch.
- Keep your fingers clear of the joints.

Ctrl-C or any problem turns torque off and **the legs drop**. That's why the robot must always hang or be in a harness.

## Step 1: Measure the robot (about 15 minutes)

Hang the robot so both feet are off the ground, and have a kitchen scale ready. First run the quick check, which uses no torque:

```bash
cd ~/biped_robot_test
python3 robot_test.py check --biped-ws ~/biped_ws
```

If it stops with "too hot", copy the whole message and send it to me. Don't raise the temperature limit on your own. If a servo that isn't hot reads 150 C, the test can't continue until we've dealt with that servo.

If the check is clean, run the full test:

```bash
python3 robot_test.py all --biped-ws ~/biped_ws
```

It asks you questions as it goes, including whether each joint moved the right way. If a joint moves the wrong way, answer no and the test stops.

**Send me the `robot_test_<date>.tar.gz` file it creates in your home folder. Stop here until I've checked it.**

## Step 2: Find how the IMU is mounted (done)

This passed on 4 November with `x,y,z`. Only repeat it if the IMU has been moved:

```bash
python3 run_policy.py policy_7m_s4.npz --stage imu --biped-ws ~/biped_ws
```

## Step 3: Run the policy, one stage at a time

Move to the next stage only when the last one looks right. Send me the `.tar.gz` file after each run.

| Stage | Robot | Torque limit | Actions | Time |
|---|---|---|---|---|
| `hang` | Both feet off the ground | 500 | 30% | 20 s |
| `supported` | Feet touching, held by a harness or a person | 700 | Full | 15 s |
| `ground` | Harness slack, ready to catch | 1000 | Full | 15 s |

```bash
python3 run_policy.py policy_7m_s4.npz --stage hang --speed 0.2 --budget 0.5,0.5,0.5 --hold-heading --imu-mount x,y,z --biped-ws ~/biped_ws
```

- `--stage` is `hang`, then `supported`, then `ground`.
- `--budget` sets flight, slip and torque, from 0 (tightest) to 1 (loosest). 0.5 is where these policies walk best in the simulator.
- `--hold-heading` steers back to the starting direction using the IMU. It assumes the IMU's z axis is close to vertical.
- If `hardware.yaml` lowers the torque limit, the script tells you. Let me know if it does, because the policy was trained with the full limit.
- Don't use `sudo` for now. It can change where the script finds your calibration and Python packages. The run works without it, and `summary.json` shows how many loops ran late.

## What stops a run

Torque goes off if any of these happen:
- A servo or the IMU stops answering.
- A servo gets too hot, or the voltage drops too low.
- The robot tilts more than 45 degrees.
- A joint stays far from its target for 0.5 s.
- Control loops run late.
- You press Ctrl-C, or the SSH link drops.

Neither script writes to the servos' permanent memory.

## Trying the scripts without a robot

Add `--mock --yes` to `run_policy.py`, or `--mock --yes --quick` to `robot_test.py`. They run on pretend servos, and the numbers mean nothing about the real robot.
