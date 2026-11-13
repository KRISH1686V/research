#!/bin/bash
# Harder survival tests of the 2026-11-02 survival runs: full realism with pushes, new seeds, 10 s and 30 s, then every saved checkpoint. At most 6 at once.
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
cd $R
L=$V/survive2/jobs.txt; : > $L
for A in budgetsurvive budgetsurvivegait; do for S in 3 4; do
  N=fast_${A}_20261102_seed$S; D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  echo "${A}_s$S $D/sb3_her_final.pt $RC $V/survive2/${A}_s${S}_full_push_10s.json full 1 10 9200:9230 all" >> $L
  echo "${A}_s$S $D/sb3_her_final.pt $RC $V/survive2/${A}_s${S}_full_push_30s.json full 1 30 9230:9260 all_loose,all_mid,all_tight" >> $L
  for K in 250000 500000 750000 1000000 1250000 1500000 1750000 2000000 2250000 2500000 2750000; do
    echo "${A}_s$S $D/sb3_her_model_${K}_steps.pt $RC $V/survive2/scan/${A}_s${S}_$K.json full 1 10 9300:9320 all_loose,all_mid" >> $L
  done
  echo "${A}_s$S $D/sb3_her_final.pt $RC $V/survive2/scan/${A}_s${S}_3000000.json full 1 10 9300:9320 all_loose,all_mid" >> $L
done; done
cat $L | xargs -P 6 -L 1 nice -n 5 $PY $V/survive_eval.py
echo "survive2 finished $(date +%H:%M)" >> $V/survive2/done.log
