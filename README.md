# Budget-as-Goal Hindsight Relabeling for a Low-Cost Biped

One training run that learns a walking policy for many gait limits at once. Limits on flight (both feet off the ground), stance slip and motor torque, and optionally step rate, foot lift and flat-foot stance, are rolling token-bucket budgets. The policy reads them as part of its goal. During training, past steps are relabeled in hindsight with new speed targets and new budgets, and every bucket level is recomputed exactly, so the same data teaches every limit.

**Status (6 November 2026).** All results are from the MuJoCo simulator. No trained policy has moved the real robot yet.

- At 7.2 million training steps, all 5 training seeds pass the project's fixed walking goal on 100 new test episodes each.
- The same setup without budgets is worse on every seed. Averaged over 5 seeds, it stays up in 81.4 of 100 episodes against 98.6, walks with alternating steps in 28 of 500 against 334, and has both feet in the air 13.8% of the time against 4.4%.
- The robot still takes about 6 short steps per second. Step-rate, foot-lift and flat-foot budgets are in testing.
- The comparisons with other methods have not run yet. See `docs/done_and_left.md`.

## Start here

| Read | For |
| --- | --- |
| `docs/project_overview.md` | What the project is, why, and every result so far |
| `docs/learning_the_codebase.md` | A guided tour of the code, with a worked example of the relabeling formula |
| `docs/done_and_left.md` | What is finished and what remains |
| `BipedRobot/diagnostics/reports/Budget relabeling novelty check.md` | Literature check: what is new and what is not |
| `BipedRobot/diagnostics/WORKING_RULES.md` | The rules used to judge results |
| `BipedRobot/diagnostics/PROGRESS.md` | The full experiment log |

## The relabeling formula

For a stored episode with cumulative cost `S(k)` before step `k`, a relabeled rate `r`, window `W` and bucket size `B = r × W`:

```
F(k) = S(k) − r·k
d(t) = F(t) − min over max(0, t−W) ≤ k ≤ t of F(k)
over budget when d(t) > B
```

`BipedRobot/fast/budget_replay.py` (`window_levels`, `sample`) computes this for a whole batch on the GPU. `BipedRobot/tests/test_budget_window.py` checks it against a step-by-step bucket.

## Layout

| Folder | Contents |
| --- | --- |
| `BipedRobot/fast/` | The method: budget environment, GPU replay with budget relabeling, SAC learner, training loop |
| `BipedRobot/envs/` | Walking environment and the robot's MuJoCo model (`assets/robotV2/`) |
| `BipedRobot/config/` | Training configs. The main one is `v2_deploy_budget_alpha30.yaml` |
| `BipedRobot/tests/` | Tests, including exact checks of the budgets and relabeling |
| `BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2/` | Scoring and analysis scripts, launch scripts, robot scripts (`hardware_tests/`), and result files (`results/`) |
| `models/` | Final 7.2 million step policies for seeds 3 to 7, with budgets (`deploylong7m`) and without (`deploycmdlong7m`), each with its `resolved_config.yaml` |
| `robot_test/` | The package for the Raspberry Pi: robot measurement, policy runner, exported policies |
| `docs/` | The project documents above |

## Running it

Python 3.11, with the packages in `BipedRobot/requirements.txt`. The results used MuJoCo 3.2.7, PyTorch 2.7.1 (CUDA 11.8), Stable-Baselines3 2.9.0 and Gymnasium 1.0.0. Run commands from the folder that contains `BipedRobot/`.

```bash
python -m pytest BipedRobot/tests/test_budget_window.py -q                          # the formula against a step-by-step count
python BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2/watch_bucket.py      # watch buckets drain and the reward gate act
```

`BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2/run_budget_deploy.sh` shows the full training command and the tests that follow each run.

## Before running on another machine

- Many configs and shell scripts use absolute paths from the original workstation (`/home/rpuser1/yasho/rl`). Change them to your paths.
- Scoring scripts read results from paths such as `goal7m/`. In this copy, those result files are under `.../v2/results/`.
- Training runs, intermediate checkpoints and large benchmark folders are not included.
- The robot's own drivers (`biped_ws`) are a separate repository and are not included.

## Credits and license

The base environment and training code (`BipedRobot/envs/`, `train_her_biped.py`, `evaluation_suite*.py` and others) come from a project released under the MIT License by Pablo Gomez Martinez. See `BipedRobot/LICENSE.txt`, which must stay with the code. The budget method, the `fast/` package and the analysis tools were built on top of it.

The work uses [MuJoCo](https://mujoco.org), [PyTorch](https://pytorch.org) and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
