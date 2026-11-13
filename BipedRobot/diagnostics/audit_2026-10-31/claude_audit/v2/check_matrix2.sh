#!/bin/bash
# Usage: CMDS="0.0 0.15" check_matrix2.sh ARM SEED... As check_matrix.sh, with the commands from CMDS (default 0.0 0.05 0.1). As each 7.2M run of ARM ends: gait_report on test seeds 9001-9100 (100 episodes, medium, mid budget) at command
# 0.0, 0.05 and 0.1, servo duty 1.0 and 0.5, gyro heading hold off and on (the 6 Nov Deep Think evaluation, plus 0.05, the clean arms' trained speed).
# CPU only (CUDA hidden). Detached. Written 6 Nov 2026.
A=$1; shift
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
G=$V/goal7m/matrix; mkdir -p $G
check() {
  local S=$1 N=fast_${A}_20261102_seed$1
  local F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
  until [ -f $F ]; do sleep 60; done
  sleep 10
  for CMD in ${CMDS:-0.0 0.05 0.1}; do for D in 1.0 0.5; do for H in 0 1; do
    T=${A}_s${S}_v${CMD}_d${D}_h$H
    CUDA_VISIBLE_DEVICES="" GAIT_DUTY=$D GAIT_HOLD=$H nice -n 10 $PY $V/gait_report.py $T $F $RC $G/$T.json medium 10 9001:9101 all_mid $CMD > $G/$T.log 2>&1 || echo "FAIL $T" >> $G/fail.log
  done; done; done
}
for S in "$@"; do check $S & done
wait
echo finished > $G/done_$A.log
