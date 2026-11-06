#!/bin/bash
# Usage: check_steps7m.sh ARM SEED... As each 7.2M step-budget run ends: the goal budgets plus the 2 steps per second settings on seeds 10000-10099,
# then real steps, step length and foot lift from gait_report.py on seeds 9700-9719. Detached.
A=$1; shift
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
G=$V/goal7m; mkdir -p $G/gait
check() {
  local S=$1 N=fast_${A}_20261102_seed$1
  local F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  until [ -f $F ]; do sleep 60; done
  sleep 10
  nice -n 10 $PY $V/survive_eval.py ${A}_s$S $F $RC $G/${A}_s$S.json medium 0 10 10000:10100 all_loose,all_mid,all_tight,loose_steps2,mid_steps2 > $G/${A}_s$S.log 2>&1 || echo "FAIL ${A}_s$S" >> $G/fail.log
  for B in mid_steps2 loose_steps2 all_mid; do
    nice -n 10 $PY $V/gait_report.py gait_${A}_s$S $F $RC $G/gait/${A}_s${S}_$B.json medium 10 9700:9720 $B > $G/gait/${A}_s${S}_$B.log 2>&1 || echo "FAIL gait ${A}_s$S $B" >> $G/fail.log
  done
}
for S in "$@"; do check $S & done
wait
echo finished > $G/done_$A.log
