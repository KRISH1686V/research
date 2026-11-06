#!/bin/bash
# Starts when the runs named by PID have ended, so at most 4 train at once. Reads its list then, so the list can still change until that moment.
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
cat deploy_runs8.txt | xargs -P 4 -L 1 ./run_method.sh
echo "queue 8 finished $(date +%H:%M)"
