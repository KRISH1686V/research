#!/bin/bash
# Pauses only the given training run if available memory runs low, and resumes it when memory recovers.
# Usage: memory_guard.sh MAIN_PID
MAIN=$1; PIDS="$MAIN $(ps -o pid= --ppid $MAIN | tr '\n' ' ')"; paused=0
while kill -0 $MAIN 2>/dev/null; do
  A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$A" -lt 3000 ] && [ $paused = 0 ]; then kill -STOP $PIDS; paused=1; echo "$(date) paused $MAIN, available ${A} MB"; fi
  if [ "$A" -gt 6000 ] && [ $paused = 1 ]; then kill -CONT $PIDS; paused=0; echo "$(date) resumed $MAIN, available ${A} MB"; fi
  sleep 60
done
echo "$(date) $MAIN exited"
