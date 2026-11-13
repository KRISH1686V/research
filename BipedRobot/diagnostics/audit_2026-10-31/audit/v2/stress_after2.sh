#!/bin/bash
# Usage: stress_after2.sh ARM SEED SPEED. As stress_after.sh, with the stress set's walking conditions at SPEED, for runs trained at that command. Detached. Written 8 Nov 2026.
A=$1; S=$2; SP=$3
cd "$(dirname "$0")"
F=../../../../checkpoints/fast_${A}_20261102_seed$S/seed_$S/sb3_her_final.pt
until [ -f $F ]; do sleep 60; done; sleep 30
for t in stress_all delay40_v nominal_v nominal_v0 duty07_v duty07_v0 duty085_v inplace30; do echo "$A $S $SP $t"; done | xargs -P 4 -L 1 ./stress_job.sh
echo done > goal7m/export/done_stress_${A}_s$S.log
