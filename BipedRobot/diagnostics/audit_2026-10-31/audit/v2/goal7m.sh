#!/bin/bash
# As each 7.2M run finishes, test its final checkpoint on seeds no check has used (10000-10099). Detached, so the results are ready even if no session is open.
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
G=$V/goal7m
mkdir -p $G
check() {
  local A=$1 S=$2 B=$3 N=fast_$1_20261102_seed$2
  local F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt
  until [ -f $F ]; do sleep 60; done
  sleep 10
  nice -n 10 $PY $V/survive_eval.py ${A}_s$S $F BipedRobot/runs/$N/seed_$S/resolved_config.yaml $G/${A}_s$S.json medium 0 10 10000:10100 $B > $G/${A}_s$S.log 2>&1 || echo "FAIL ${A}_s$S" >> $G/fail.log
}
for S in 3 4 5 6 7; do check deploylong7m $S all_loose,all_mid,all_tight & done
for S in 3 4 5 6 7; do check deploycmdlong7m $S all_loose & done
wait
echo finished > $G/done.log
