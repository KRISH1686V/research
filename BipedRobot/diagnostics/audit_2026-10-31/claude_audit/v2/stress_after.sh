#!/bin/bash
# Usage: stress_after.sh ARM SEED [EXTRA_TAGS]. Waits for the 7.2M run's final checkpoint, then the stress set (stress_job.sh, seeds 9101-9200, at 0.05)
# plus any extra tags (e.g. "stress_all10 nominal_v10"). Detached. Written 8 Nov 2026.
A=$1; S=$2; X=${3:-}
cd "$(dirname "$0")"
F=../../../../checkpoints/fast_${A}_20261102_seed$S/seed_$S/sb3_her_final.pt
until [ -f $F ]; do sleep 60; done; sleep 30
for t in stress_all delay40_v nominal_v nominal_v0 duty07_v duty07_v0 duty085_v inplace30 $X; do echo "$A $S 0.05 $t"; done | xargs -P 4 -L 1 ./stress_job.sh
echo done > goal7m/export/done_stress_${A}_s$S.log
