#!/bin/bash
# As run_budget_next.sh, for policies that read only what the real robot can measure: the critic still sees true speed and balances, the policy does not.
# Usage: run_budget_deploy.sh ARM CONFIG SEED
set -u
A=$1; C=$2; S=$3
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
N=fast_${A}_20261102_seed$S; L=${A}_s$S
TOTAL=3000000; [[ $A == *long5m* ]] && TOTAL=4800000
K=$((TOTAL / 12)); RAMP=0.4
# long7m: more steps only. The ramp still ends at 1.92M as in long5m, and checkpoints land on the same steps, so the two can be compared directly.
[[ $A == *long7m* ]] && { TOTAL=7200000; K=400000; RAMP=0.26666666666666666; }
# caps: policy smoothness as a training loss. Each weight pulls about a quarter as hard as the critic's action gradient, measured on the 4.8M policies (|dQ/da| 0.4-0.6, step 1.9-2.2, noisy gap 0.44-0.51).
EXTRA=""; [[ $A == *caps* ]] && EXTRA="--caps_t 0.03 --caps_s 0.14 --caps_sigma 0.01"
M=/home/rpuser1/.biped_mps
[ -S $M/pipe/control ] && export CUDA_MPS_PIPE_DIRECTORY=$M/pipe CUDA_MPS_LOG_DIRECTORY=$M/log
cd $R/BipedRobot
nice -n 5 $PY fast/train.py --config $C --exp_name $N --seed $S --n_envs 8 --total_steps $TOTAL --realism full --mode graph --log_dir runs --checkpoint_dir checkpoints --plain_ckpt \
  --difficulty_ramp $RAMP --layer_norm --avg_q --ema_tau 0.002 --asym --mirror --save_every $K --actor_hide achieved,levels --save_critic $EXTRA > runs/$N.log 2>&1 || { echo "FAIL train $N"; exit 1; }
cd $R
D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
if [[ $A == *cmd* ]]; then B=all_loose; B30=all_loose; else B=all; B30=all_loose,all_mid,all_tight; fi
mkdir -p $V/deploy
$PY $V/avg_ckpts.py $V/deploy/${L}_last4.pt $D/sb3_her_model_$((TOTAL - 3 * K))_steps.pt $D/sb3_her_model_$((TOTAL - 2 * K))_steps.pt $D/sb3_her_model_$((TOTAL - K))_steps.pt $D/sb3_her_final.pt
for T in "medium 0 10 7000:7030 $B medium" "full 0 10 7000:7030 $B full" "full 1 10 9200:9230 $B full_push_10s" "full 1 30 9230:9260 $B30 full_push_30s"; do
  set -- $T
  nice -n 5 $PY $V/survive_eval.py $L $D/sb3_her_final.pt $RC $V/deploy/${L}_$6.json $1 $2 $3 $4 $5 > /dev/null 2>&1 || echo "FAIL eval $L $6" &
  nice -n 5 $PY $V/survive_eval.py $L $V/deploy/${L}_last4.pt $RC $V/deploy/${L}_last4_$6.json $1 $2 $3 $4 $5 > /dev/null 2>&1 || echo "FAIL eval $L last4 $6" &
done
wait
echo "evaluated $N $(date +%H:%M)"
