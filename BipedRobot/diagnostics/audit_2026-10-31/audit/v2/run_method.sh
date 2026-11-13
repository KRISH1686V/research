#!/bin/bash
# One line of the queue. PPO has its own trainer and uses no GPU; every other method goes through run_budget_deploy.sh.
# Usage: run_method.sh ARM CONFIG SEED
set -u
A=$1; C=$2; S=$3
[[ $A != ppo* ]] && exec ./run_budget_deploy.sh "$A" "$C" "$S"
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
N=fast_${A}_20261102_seed$S; L=${A}_s$S
TOTAL=3000000; [[ $A == *long5m* ]] && TOTAL=4800000; [[ $A == *long7m* ]] && TOTAL=7200000
# c17 comparisons (11 Nov 2026): the same randomization and smoothness schedule as SAC (difficulty ramp ends at 1.92M, as run_budget_deploy.sh long7m)
EXTRA=""; [[ $A == *c17* ]] && EXTRA="--difficulty_ramp 0.26666666666666666 --match_sac_schedule"
cd $R/BipedRobot
nice -n 5 $PY fast/train_ppo.py --config $C --exp_name $N --seed $S --n_envs 8 --total_steps $TOTAL --log_dir runs --checkpoint_dir checkpoints $EXTRA > runs/$N.log 2>&1 || { echo "FAIL train $N"; exit 1; }
cd $R
D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
mkdir -p $V/deploy
for T in "medium 0 10 7000:7030 medium" "full 0 10 7000:7030 full" "full 1 10 9200:9230 full_push_10s" "full 1 30 9230:9260 full_push_30s"; do
  set -- $T
  nice -n 5 $PY $V/survive_eval.py $L $D/ppo_final.zip $RC $V/deploy/${L}_$5.json $1 $2 $3 $4 all_loose > /dev/null 2>&1 || echo "FAIL eval $L $5" &
done
wait
echo "evaluated $N $(date +%H:%M)"
