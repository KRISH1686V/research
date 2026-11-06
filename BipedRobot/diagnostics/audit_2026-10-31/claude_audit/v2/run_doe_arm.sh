#!/bin/bash
# One screening run: fast learner on the V2 robot, 1M steps, then tests. Final checkpoint for the score, 700k-900k for stability.
# Usage: run_doe_arm.sh RUN CONFIG SEED [fast/train.py flags]
set -u
A=$1; C=$2; S=$3; shift 3
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
N=fast_${A}_20261102_seed$S; L=${A}_s$S
cd $R/BipedRobot
nice -n 5 $PY fast/train.py --config $C --exp_name $N --seed $S --n_envs 8 --total_steps 1000000 --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints --plain_ckpt "$@" > runs/$N.log 2>&1 || { echo "FAIL train $N"; exit 1; }
cd $R
D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
(
  mkdir -p $V/doe
  for J in "sb3_her_final V2_no_push final" "sb3_her_final V2_medium final" "sb3_her_final V2_train final" "sb3_her_model_700000_steps V2_no_push 700k" "sb3_her_model_800000_steps V2_no_push 800k" "sb3_her_model_900000_steps V2_no_push 900k"; do set -- $J; echo "$D/$1.pt $2 $V/doe/${L}__$2__$3.json"; done | \
    xargs -P 2 -L 1 sh -c "nice -n 5 $PY $V/validity_audit.py $L \$0 $RC \$1 \$2 > /dev/null 2>&1 || echo FAIL audit $L \$1 \$2"
  echo "evaluated $L $(date +%H:%M)"
) >> $R/$V/doe_evals.log 2>&1 &
echo "trained $N $(date +%H:%M)"
