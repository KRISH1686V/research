#!/bin/bash
# One budget-HER run on the corrected simulation, then budget-obedience tests in medium and full realism (no pushes).
# Usage: run_budget_arm.sh ARM CONFIG SEED [extra fast/train.py flags]
set -u
A=$1; C=$2; S=$3; shift 3
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
N=fast_${A}_20261102_seed$S; L=Fast-${A}_s$S
cd $R/BipedRobot
nice -n 5 $PY fast/train.py --config $C --exp_name $N --seed $S --n_envs 8 --total_steps 1000000 --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints "$@" > runs/$N.log 2>&1 || { echo "FAIL train $N"; exit 1; }
cd $R
D=BipedRobot/checkpoints/$N/seed_$S; CK=$D/sb3_her_final.pt; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
mkdir -p $V/budget
(
  for RL in medium full; do nice -n 5 $PY $V/budget_eval.py $L $CK $RC $RL $V/budget/${L}_$RL.json > /dev/null 2>&1 || echo "FAIL budget eval $L $RL" & done
  wait
  echo "evaluated $L $(date +%H:%M)"
) >> $R/$V/budget_evals.log 2>&1 &
echo "trained $N $(date +%H:%M)"
