#!/bin/bash
# Usage: auto_inspect.sh COMMAND ARM:SEED ... As each 7.2M run ends, gait_anomaly.py on seeds 9300-9319 at COMMAND. Detached. Written 7 Nov 2026.
CMD=$1; shift
cd "$(dirname "$0")/../../../../.."
PY=$(ls -d ~/miniconda3/envs/rl/bin/python)
V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
O=$V/goal7m/anomaly; mkdir -p $O
one() { A=${1%%:*}; S=${1##*:}; N=fast_${A}_20261102_seed$S; F=BipedRobot/checkpoints/$N/seed_$S/sb3_her_final.pt
  until [ -f $F ]; do sleep 60; done; sleep 30
  CUDA_VISIBLE_DEVICES="" nice -n 10 $PY $V/gait_anomaly.py ${A}_s$S $F BipedRobot/runs/$N/seed_$S/resolved_config.yaml $O/${A}_s$S.json 9300:9320 $CMD > $O/${A}_s$S.log 2>&1; }
for x in "$@"; do one $x & done
wait
