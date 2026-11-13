#!/bin/bash
# Usage: run_benchmark.sh LABEL CHECKPOINT RESOLVED_CONFIG [PARALLEL]
# Standard v2 suite, 30 seeds each. Run from /home/rpuser1/yasho/rl.
set -u
LABEL=$1; CK=$2; RC=$3; P=${4:-5}
OUT=BipedRobot/diagnostics/audit_2026-10-31/audit/v2/bench/$LABEL
mkdir -p $OUT
PY=/home/rpuser1/miniconda3/envs/rl/bin/python
printf '%s\n' "easy 0.0 -" "easy 0.2 -" "easy 0.3 -" "medium 0.2 -" "full 0.2 --pushes" |
xargs -P $P -L 1 bash -c 'nice -n 5 '$PY' BipedRobot/evaluation_suite_v2.py --checkpoint '$CK' --resolved-config '$RC' --realism $0 --command $1 0 0 --seeds 5000:5030 $( [ "$2" != "-" ] && echo $2 ) --gif-dir '$OUT'/gifs --n-gifs 2 --label '$LABEL' --out '$OUT'/$0_$1$2.json > /dev/null 2>&1 || echo "FAIL '$LABEL' $0 $1"'
