#!/bin/bash
# One stress-test job: stress_job.sh ARM SEED SPEED TAG. TAG picks the condition (the 7 Nov plan's pre-export checks plus stress_all). Seeds 9101-9200.
A=$1; S=$2; SP=$3; T=$4
cd /home/rpuser1/yasho/rl
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
G=$V/goal7m/export; mkdir -p $G
N=fast_${A}_20261102_seed$S; F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt; RC=BipedRobot/runs/$N/seed_$S/resolved_config.yaml
[ -f $G/${A}_s${S}_$T.json ] && exit 0
case $T in
  nominal_v0) E="GAIT_DUTY=1.0"; C=0.0; SEC=10;;
  nominal_v) E="GAIT_DUTY=1.0"; C=$SP; SEC=10;;
  duty07_v0) E="GAIT_DUTY=0.7"; C=0.0; SEC=10;;
  duty07_v) E="GAIT_DUTY=0.7"; C=$SP; SEC=10;;
  delay40_v) E="GAIT_DUTY=1.0 GAIT_DELAY=0.04"; C=$SP; SEC=10;;
  duty085_v) E="GAIT_DUTY=0.85"; C=$SP; SEC=10;;
  inplace30) E="GAIT_DUTY=1.0"; C=0.0; SEC=30;;
  stress_all) E="GAIT_DUTY=0.8 GAIT_DELAY=0.04 GAIT_COM=0.012,0.006"; C=$SP; SEC=10;;
  stress_all10) E="GAIT_DUTY=0.8 GAIT_DELAY=0.04 GAIT_COM=0.012,0.006"; C=0.10; SEC=10;;
  nominal_v10) E="GAIT_DUTY=1.0"; C=0.10; SEC=10;;
esac
env CUDA_VISIBLE_DEVICES="" $E nice -n 15 $PY $V/gait_report.py ${A}_s${S}_$T $F $RC $G/${A}_s${S}_$T.json medium $SEC 9101:9201 all_mid $C > $G/${A}_s${S}_$T.log 2>&1
