#!/bin/bash
# How much do the survival policies lean on inputs the real robot cannot measure? No retraining. Same realism and seeds as budget/*_full.json (full, no pushes, 7000-7029).
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
cd $R
L=$V/hidden_inputs/jobs.txt; : > $L
for A in budgetsurvive budgetsurvivegait; do for S in 3 4; do
  N=fast_${A}_20261102_seed$S; D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  for H in none vel_zero vel_cmd levels_full vel_zero,levels_full; do
    T=$H; [ "$H" = none ] && T=""
    echo "${A}_s$S $D/sb3_her_final.pt $RC $V/hidden_inputs/${A}_s${S}_${H/,/+}.json full 0 10 7000:7030 all_loose,all_mid,flight_mid $T" >> $L
  done
done; done
cat $L | xargs -P 2 -L 1 nice -n 10 $PY $V/survive_eval.py
echo "hidden inputs finished $(date +%H:%M)" >> $V/hidden_inputs/done.log
