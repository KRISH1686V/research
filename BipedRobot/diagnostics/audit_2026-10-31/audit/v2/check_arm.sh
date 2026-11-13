#!/bin/bash
# Usage: check_arm.sh ARM. As each 7.2M run of ARM finishes, test its final checkpoint against the fixed goal (seeds 10000-10099) and on the lab floor (seeds 10400-10499). Detached.
A=$1
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
G=$V/goal7m; L=$V/lab_floor/long7m
B=all_loose,all_mid,all_tight
check() {
  local S=$1 N=fast_${A}_20261102_seed$1
  local F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  until [ -f $F ]; do sleep 60; done
  sleep 10
  nice -n 10 $PY $V/survive_eval.py ${A}_s$S $F $RC $G/${A}_s$S.json medium 0 10 10000:10100 $B > $G/${A}_s$S.log 2>&1 || echo "FAIL ${A}_s$S" >> $G/fail.log
  $PY $V/lab_rc.py $RC $L/rc_${A}_s$S.yaml
  nice -n 10 $PY $V/survive_eval.py lab_${A}_s$S $F $L/rc_${A}_s$S.yaml $L/${A}_s$S.json medium 0 10 10400:10500 $B > $L/${A}_s$S.log 2>&1 || echo "FAIL ${A}_s$S" >> $L/fail.log
}
for S in 3 4 5 6 7; do check $S & done
wait
echo finished > $G/done_$A.log
