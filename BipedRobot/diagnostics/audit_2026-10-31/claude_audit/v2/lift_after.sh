#!/bin/bash
# Usage: lift_after.sh COMMAND ARM:SEED ... As each 7.2M run ends, lift_split.py at COMMAND into goal7m/anomaly/lift_ARM_sSEED.json. Detached. Written 8 Nov 2026.
C=$1; shift; cd "$(dirname "$0")"; R=../../../..; PY=$(ls -d ~/miniconda3/envs/rl/bin/python)
one() { A=${1%%:*}; S=${1##*:}; F=$R/checkpoints/fast_${A}_20261102_seed$S/seed_$S/sb3_her_final.pt; until [ -f $F ]; do sleep 60; done; sleep 30
  CUDA_VISIBLE_DEVICES="" nice -n 10 $PY lift_split.py $A:$S:$C --json goal7m/anomaly/lift_${A}_s$S.json > goal7m/anomaly/lift_${A}_s$S.log 2>&1; }
for x in "$@"; do one $x & done; wait
