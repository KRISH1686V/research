#!/bin/bash
# Usage: run_lane.sh LIST WIDTH [PID ...]. Waits for the named PIDs, then runs LIST WIDTH at a time. Two lanes of width 2 keep at most 4 training at once.
L=$1; W=$2; shift 2
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
cat $L | xargs -P $W -L 1 ./run_method.sh
echo "$L finished $(date +%H:%M)"
