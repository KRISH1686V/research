#!/bin/bash
# Usage: check_straight.sh ARM SPEED SEED... As each 7.2M run of ARM ends: survive_eval at command SPEED on seeds 10000-10099 at all_loose and all_mid,
# without and with the gyro heading hold, then gait_report at SPEED on seeds 9700-9719 at all_loose and all_mid. Detached. Written 6 Nov 2026.
A=$1; SP=$2; shift 2
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
G=$V/goal7m/straight; mkdir -p $G/gait
check() {
  local S=$1 N=fast_${A}_20261102_seed$1
  local F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  until [ -f $F ]; do sleep 60; done
  sleep 10
  SURVIVE_SPEED=$SP nice -n 10 $PY $V/survive_eval.py ${A}_s$S $F $RC $G/${A}_s$S.json medium 0 10 10000:10100 all_loose,all_mid > $G/${A}_s$S.log 2>&1 || echo "FAIL ${A}_s$S" >> $G/fail.log
  SURVIVE_SPEED=$SP nice -n 10 $PY $V/survive_eval.py ${A}_s${S}_hold $F $RC $G/${A}_s${S}_hold.json medium 0 10 10000:10100 all_loose,all_mid hold_gyro > $G/${A}_s${S}_hold.log 2>&1 || echo "FAIL hold ${A}_s$S" >> $G/fail.log
  for X in all_loose all_mid; do
    nice -n 10 $PY $V/gait_report.py gait_${A}_s$S $F $RC $G/gait/${A}_s${S}_$X.json medium 10 9700:9720 $X $SP > $G/gait/${A}_s${S}_$X.log 2>&1 || echo "FAIL gait ${A}_s$S $X" >> $G/fail.log
  done
}
for S in "$@"; do check $S & done
wait
echo finished > $G/done_$A.log
