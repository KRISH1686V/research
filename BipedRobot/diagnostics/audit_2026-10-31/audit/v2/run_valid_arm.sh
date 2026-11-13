#!/bin/bash
# One run on the corrected simulation: fast learner, 1M steps, then tests on the old and the corrected simulation.
# The gate reads the final checkpoint only. Evaluations continue in the background so the next run can start.
# Usage: run_valid_arm.sh ARM CONFIG SEED [extra fast/train.py flags]
set -u
A=$1; C=$2; S=$3; shift 3
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
N=fast_${A}_20261102_seed$S; L=Fast-${A}_s$S
cd $R/BipedRobot
nice -n 5 $PY fast/train.py --config $C --exp_name $N --seed $S --n_envs 8 --total_steps 1000000 --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints "$@" > runs/$N.log 2>&1 || { echo "FAIL train $N"; exit 1; }
cd $R
D=BipedRobot/checkpoints/$N/seed_$S; CK=$D/sb3_her_final.zip; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
ev() { nice -n 5 $PY BipedRobot/evaluation_suite_v2.py --checkpoint $1 --resolved-config $RC --realism medium --command 0.2 0 0 --seeds $2 --label $L --out $3 > /dev/null 2>&1 || echo "FAIL eval $3"; }
(
  mkdir -p $V/validity/retrain
  for K in V_medium V_no_push V_push25 V_push100 V_delay_0 V_delay_60 V_servo_curve V_friction_0.4 V_torque_x0.8; do echo $K; done | \
    xargs -P 3 -I{} sh -c "nice -n 5 $PY $V/validity_audit.py $L $CK $RC {} $V/validity/retrain/${L}__{}.json > /dev/null 2>&1 || echo FAIL audit $L {}"
  ev $CK 5100:5200 $V/ea/${L}_final_medium_0.2_seeds5100-5199.json
  for K in 800000 900000; do ev $D/sb3_her_model_${K}_steps.zip 5000:5030 $V/ea/${L}_${K}_medium_0.2.json; done
  nice -n 5 $PY $V/jitter.py $L $CK $RC 0.2 $V/jitter/${L}_medium_0.2.json 5000:5010 medium > /dev/null 2>&1 || echo "FAIL jitter $L"
  nice -n 5 $PY $V/paths.py $L $CK $RC medium 0.2 $V/paths/${L}_medium_0.2.json > /dev/null 2>&1 || echo "FAIL paths $L"
  echo "evaluated $L $(date +%H:%M)"
) >> $R/$V/valid_evals.log 2>&1 &
echo "trained $N $(date +%H:%M)"
