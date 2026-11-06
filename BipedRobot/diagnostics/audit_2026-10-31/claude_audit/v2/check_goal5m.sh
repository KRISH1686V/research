#!/bin/bash
# Usage: check_goal5m.sh ARM SEED... As each 4.8M run ends, test it on the goal seeds used for the plain 4.8M runs, 9600-9699, so the two are paired. Reused seeds, so a comparison and not a claim. Detached.
A=$1; shift
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
G=$V/paired5m; mkdir -p $G
check() {
  local S=$1 N=fast_${A}_20261102_seed$1
  local F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt
  until [ -f $F ]; do sleep 60; done
  sleep 10
  nice -n 10 $PY $V/survive_eval.py ${A}_s$S $F BipedRobot/runs/$N/seed_$S/resolved_config.yaml $G/${A}_s$S.json medium 0 10 9600:9700 all_loose,all_mid,all_tight > $G/${A}_s$S.log 2>&1 || echo "FAIL ${A}_s$S" >> $G/fail.log
}
for S in "$@"; do check $S & done
wait
echo finished > $G/done_$A.log
