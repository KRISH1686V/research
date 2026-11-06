#!/bin/bash
# Do averaged weights swing less than single checkpoints? Same test as survive2/scan (full, pushes, 10 s, seeds 9300-9319, all_loose and all_mid).
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
cd $R
L=$V/avg/jobs.txt; : > $L
for A in budgetsurvive budgetsurvivegait; do for S in 3 4; do
  N=fast_${A}_20261102_seed$S; D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  C="$D/sb3_her_model_2250000_steps.pt $D/sb3_her_model_2500000_steps.pt $D/sb3_her_model_2750000_steps.pt $D/sb3_her_final.pt"
  $PY $V/avg_ckpts.py $V/avg/${A}_s${S}_last4.pt $C
  $PY $V/avg_ckpts.py $V/avg/${A}_s${S}_last8.pt $D/sb3_her_model_1250000_steps.pt $D/sb3_her_model_1500000_steps.pt $D/sb3_her_model_1750000_steps.pt $D/sb3_her_model_2000000_steps.pt $C
  for K in last4 last8; do echo "${A}_s$S $V/avg/${A}_s${S}_$K.pt $RC $V/avg/${A}_s${S}_$K.json full 1 10 9300:9320 all_loose,all_mid" >> $L; done
done; done
cat $L | xargs -P 1 -L 1 nice -n 10 $PY $V/survive_eval.py
echo "avg finished $(date +%H:%M)" >> $V/avg/done.log
