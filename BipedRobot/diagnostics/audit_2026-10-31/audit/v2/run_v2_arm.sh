#!/bin/bash
# One non-budget run on the V2 robot: fast learner, 1M steps, then tests. The gate reads the final checkpoint only.
# Usage: run_v2_arm.sh ARM CONFIG SEED
set -u
A=$1; C=$2; S=$3; shift 3
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
N=fast_${A}_20261102_seed$S; L=Fast-${A}_s$S
cd $R/BipedRobot
nice -n 5 $PY fast/train.py --config $C --exp_name $N --seed $S --n_envs 8 --total_steps 1000000 --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints "$@" > runs/$N.log 2>&1 || { echo "FAIL train $N"; exit 1; }
cd $R
D=BipedRobot/checkpoints/$N/seed_$S; CK=$D/sb3_her_final.zip; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
(
  mkdir -p $V/v2
  for K in V2_train V2_no_push V2_medium V2_push_6N V2_delay_60 V2_friction_0.2 V2_torque_x0.5; do echo $K; done | \
    xargs -P 3 -I{} sh -c "nice -n 5 $PY $V/validity_audit.py $L $CK $RC {} $V/v2/${L}__{}.json > /dev/null 2>&1 || echo FAIL audit $L {}"
  nice -n 5 $PY $V/jitter.py $L $CK $RC 0.2 $V/jitter/${L}_medium_0.2.json 5000:5010 medium > /dev/null 2>&1 || echo "FAIL jitter $L"
  nice -n 5 $PY $V/paths.py $L $CK $RC medium 0.2 $V/paths/${L}_medium_0.2.json > /dev/null 2>&1 || echo "FAIL paths $L"
  echo "evaluated $L $(date +%H:%M)"
) >> $R/$V/v2_evals.log 2>&1 &
echo "trained $N $(date +%H:%M)"
