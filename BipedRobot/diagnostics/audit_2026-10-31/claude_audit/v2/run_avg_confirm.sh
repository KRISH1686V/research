#!/bin/bash
# Fresh-seed check of the rule "average the last 4 saved checkpoints" against the final checkpoint. Full, pushes, 10 s, new seeds 9400-9429, all 9 budgets.
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
cd $R
L=$V/avg/confirm_jobs.txt; : > $L
for A in budgetsurvive budgetsurvivegait; do for S in 3 4; do
  N=fast_${A}_20261102_seed$S; D=BipedRobot/checkpoints/$N/seed_$S; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  echo "${A}_s$S $V/avg/${A}_s${S}_last4.pt $RC $V/avg/confirm_${A}_s${S}_last4.json full 1 10 9400:9430 all" >> $L
  echo "${A}_s$S $D/sb3_her_final.pt $RC $V/avg/confirm_${A}_s${S}_final.json full 1 10 9400:9430 all" >> $L
done; done
cat $L | xargs -P 2 -L 1 nice -n 10 $PY $V/survive_eval.py
echo "avg confirm finished $(date +%H:%M)" >> $V/avg/done.log
