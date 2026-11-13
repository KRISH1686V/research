# Budget-as-goal HER for a 12-DOF biped

A walking policy (SAC + HER, MuJoCo) whose goal includes three budgets, for flight, foot slip and joint shake, so the gait trade-off can be changed while it walks. Work in progress.

## What is where

| Path | What it does |
|---|---|
| `BipedRobot/fast/train.py` | Trains the policy (SAC + HER) |
| `BipedRobot/fast/train_ppo.py` | PPO baseline |
| `BipedRobot/fast/budget_env.py` | Budgets: token bucket, costs, reward gate |
| `BipedRobot/fast/budget_replay.py` | Relabels budgets in the replay buffer |
| `BipedRobot/fast/valid_env.py` | Servo and realism model |
| `BipedRobot/envs/` | Robot environment and MuJoCo models (do not edit `her_loco_env.py`) |
| `BipedRobot/config/` | One YAML per run type; the method is `v2_deploy_budget_alpha30_r17w_fit.yaml` |
| `BipedRobot/tests/` | Unit tests |
| `BipedRobot/diagnostics/` | Working rules, progress log, evaluation and queue scripts |
| `models/<arm>_s<seed>/` | Final checkpoint, config and training curve of each finished run |
| `robot/robot_test_10nov/` | Exported policies and the Raspberry Pi runner |

## How to run

Install `BipedRobot/requirements.txt`. Run from the folder that contains `BipedRobot/`.

Train:

```bash
python BipedRobot/fast/train.py --config BipedRobot/config/v2_deploy_budget_alpha30_r17w_fit.yaml --exp_name deploylong7mr17w --seed 11
```

Test:

```bash
python -m pytest BipedRobot/tests
```

Evaluate a model at tight, middle and loose budgets:

```bash
cd BipedRobot/diagnostics/audit_2026-10-31/audit/v2
python compare_eval.py deploylong7mr17w 11 --u 0,0,0 0.5,0.5,0.5 1,1,1
```

Run on the robot: see `robot/robot_test_10nov/README.md`.

Scripts contain absolute paths from the training workstation; change them on another machine.
