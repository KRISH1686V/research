#!/bin/bash
# Budget-HER with bounded bucket memory and its command-HER baselines, 3M steps, then tests: medium and full without pushes (seeds 7000-7029, as the first
# survival runs), full with pushes at 10 s (9200-9229) and 30 s (9230-9259). Baselines have no budgets, so they are only tested at u = 1.
# Usage: run_budget_next.sh ARM CONFIG SEED
set -u
A=$1; C=$2; S=$3
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
N=fast_${A}_20261102_seed$S; L=${A}_s$S
# GPU sharing without time-slicing, when the private MPS daemon is up. Same results: the default-path digest matches under it.
M=/home/rpuser1/.biped_mps
[ -S $M/pipe/control ] && export CUDA_MPS_PIPE_DIRECTORY=$M/pipe CUDA_MPS_LOG_DIRECTORY=$M/log
cd $R/BipedRobot
nice -n 5 $PY fast/train.py --config $C --exp_name $N --seed $S --n_envs 8 --total_steps 3000000 --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints --plain_ckpt \
  --difficulty_ramp 0.4 --layer_norm --avg_q --ema_tau 0.002 --asym --mirror --save_every 250000 > runs/$N.log 2>&1 || { echo "FAIL train $N"; exit 1; }
cd $R
D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
if [[ $A == cmd* ]]; then B=all_loose; B30=all_loose; else B=all; B30=all_loose,all_mid,all_tight; fi
mkdir -p $V/next
for T in "medium 0 10 7000:7030 $B medium" "full 0 10 7000:7030 $B full" "full 1 10 9200:9230 $B full_push_10s" "full 1 30 9230:9260 $B30 full_push_30s"; do
  set -- $T
  nice -n 5 $PY $V/survive_eval.py $L $D/sb3_her_final.pt $RC $V/next/${L}_$6.json $1 $2 $3 $4 $5 > /dev/null 2>&1 || echo "FAIL eval $L $6" &
done
wait
echo "evaluated $N $(date +%H:%M)"
