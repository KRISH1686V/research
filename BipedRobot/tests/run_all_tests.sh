#!/bin/bash
# Every test of the fast trainer and the robot tools, then the default-path digest, which must not change. Usage: BIPED_WS=/path/to/biped_ws tests/run_all_tests.sh
ROOT=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-/home/rpuser1/miniconda3/envs/rl/bin/python}
cd "$ROOT"
FAILED=0
nice -n 5 $PY -m pytest BipedRobot/tests/test_fast_learner.py BipedRobot/tests/test_valid_env.py BipedRobot/tests/test_gait_costs.py BipedRobot/tests/test_gait_costs_speed.py BipedRobot/tests/test_doe_features.py BipedRobot/tests/test_budget_her.py \
  BipedRobot/tests/test_budget_window.py BipedRobot/tests/test_budget_wrap.py BipedRobot/tests/test_friction_focus.py BipedRobot/tests/test_smoothness_loss.py BipedRobot/tests/test_step_budget.py BipedRobot/tests/test_lift_budget.py BipedRobot/tests/test_author_style.py BipedRobot/tests/test_gait_ref.py BipedRobot/tests/test_servo_planner.py BipedRobot/tests/test_accel_cap.py BipedRobot/tests/test_run_policy_ref.py BipedRobot/tests/test_run_policy_slowyaw.py BipedRobot/tests/test_run_policy_safety.py BipedRobot/tests/test_clean_gait.py BipedRobot/tests/test_actor_hide.py BipedRobot/tests/test_sensed_joints.py BipedRobot/tests/test_comparison_methods.py BipedRobot/tests/test_gait_clock.py BipedRobot/tests/test_run_policy.py -q || FAILED=1
[ -n "${BIPED_WS:-}" ] && { nice -n 5 $PY tests/test_robot_test.py || FAILED=1; }
TMP=$(mktemp -d)
cd BipedRobot
D=$(timeout 300 nice -n 5 $PY fast/train.py --config config/pilot_gait_min_v2.yaml --exp_name digest --seed 77 --total_steps 6000 --mode graph --log_dir $TMP/runs --checkpoint_dir $TMP/ckpt --log_every 100000 2>&1 | grep parameter_digest | cut -d'"' -f4)
rm -rf $TMP
if [ "$D" = "1a9d668edb3245f01151ac630098d5433101426776e5b0b8287c0e4844214658" ]; then echo "default-path digest unchanged"; else echo "DIGEST CHANGED: $D"; FAILED=1; fi
[ $FAILED = 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit $FAILED
