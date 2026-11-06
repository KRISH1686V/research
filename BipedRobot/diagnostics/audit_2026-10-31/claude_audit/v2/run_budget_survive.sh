#!/bin/bash
# Budget-HER for survival on the V2 robot, then budget tests in medium and full realism.
# Usage: run_budget_survive.sh ARM CONFIG SEED
set -u
A=$1; C=$2; S=$3
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
N=fast_${A}_20261102_seed$S; L=${A}_s$S
cd $R/BipedRobot
nice -n 5 $PY fast/train.py --config $C --exp_name $N --seed $S --n_envs 8 --total_steps 3000000 --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints --plain_ckpt \
  --difficulty_ramp 0.4 --layer_norm --avg_q --ema_tau 0.002 --asym --mirror --save_every 250000 > runs/$N.log 2>&1 || { echo "FAIL train $N"; exit 1; }
cd $R
D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
mkdir -p $V/budget
for RL in medium full; do nice -n 5 $PY $V/budget_eval.py $L $D/sb3_her_final.pt $RC $RL $V/budget/${L}_$RL.json > /dev/null 2>&1 || echo "FAIL budget eval $L $RL" & done
wait
echo "evaluated $N $(date +%H:%M)"
