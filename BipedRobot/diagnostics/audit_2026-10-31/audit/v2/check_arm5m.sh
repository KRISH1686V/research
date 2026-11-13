#!/bin/bash
# Usage: check_arm5m.sh ARM SEED... As each 4.8M run ends, test it on the same seeds as the plain 4.8M runs, so the two are paired: goal seeds 9600-9699, lab floor seeds 10300-10399. Reused seeds, so a comparison and not a claim. Detached.
A=$1; shift
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
G=$V/paired5m; mkdir -p $G
B=all_loose,all_mid,all_tight
check() {
  local S=$1 N=fast_${A}_20261102_seed$1
  local F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  until [ -f $F ]; do sleep 60; done
  sleep 10
  nice -n 10 $PY $V/survive_eval.py ${A}_s$S $F $RC $G/${A}_s$S.json medium 0 10 9600:9700 $B > $G/${A}_s$S.log 2>&1 || echo "FAIL ${A}_s$S" >> $G/fail.log
  $PY $V/lab_rc.py $RC $G/rc_${A}_s$S.yaml
  nice -n 10 $PY $V/survive_eval.py lab_${A}_s$S $F $G/rc_${A}_s$S.yaml $G/lab_${A}_s$S.json medium 0 10 10300:10400 $B > $G/lab_${A}_s$S.log 2>&1 || echo "FAIL lab_${A}_s$S" >> $G/fail.log
}
for S in "$@"; do check $S & done
wait
echo finished > $G/done_$A.log
