#!/bin/bash
# Starts when the runs named by PID have ended, so at most 4 train at once. Tests the two walking fixes first, then the remaining seeds.
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
cat deploy_runs3.txt | xargs -P 4 -L 1 ./run_budget_deploy.sh
echo "queue 3 finished $(date +%H:%M)"
