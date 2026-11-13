#!/bin/bash
# Usage: check_export.sh ARM SEED SPEED. The 7 Nov plan's pre-export checks on fresh test seeds 9101-9200 (100 episodes, medium, mid budget, CPU only):
# nominal (duty 1.0) at 0 and SPEED; the supported stage's 70% torque (duty 0.7) at 0 and SPEED; worst-case delay (20 ms + one control step = 40 ms) at
# SPEED; low battery (duty 0.85) at SPEED; 30 s in place. Written 7 Nov 2026.
A=$1; S=$2; SP=$3
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
G=$V/goal7m/export; mkdir -p $G
N=fast_${A}_20261102_seed$S; F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
one() { T=$1; CMD=$2; SEC=$3; shift 3
  env CUDA_VISIBLE_DEVICES="" "$@" nice -n 10 $PY $V/gait_report.py ${A}_s${S}_$T $F $RC $G/${A}_s${S}_$T.json medium $SEC 9101:9201 all_mid $CMD > $G/${A}_s${S}_$T.log 2>&1 || echo "FAIL $A $S $T" >> $G/fail.log; }
(one nominal_v0 0.0 10 GAIT_DUTY=1.0; one nominal_v $SP 10 GAIT_DUTY=1.0; one duty07_v0 0.0 10 GAIT_DUTY=0.7) &
(one duty07_v $SP 10 GAIT_DUTY=0.7; one delay40_v $SP 10 GAIT_DUTY=1.0 GAIT_DELAY=0.04; one duty085_v $SP 10 GAIT_DUTY=0.85) &
(one inplace30 0.0 30 GAIT_DUTY=1.0) &
wait
echo finished > $G/done_${A}_s$S.log
