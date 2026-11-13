#!/bin/bash
# Starts when the runs named by PID have ended, so at most 4 train at once.
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
cat deploy_runs.txt | xargs -P 4 -L 1 ./run_budget_deploy.sh
echo "deploy runs finished $(date +%H:%M)"
