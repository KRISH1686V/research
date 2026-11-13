#!/bin/bash
# Usage: run_queue.sh QUEUE WIDTH. Keeps at most WIDTH training runs going on this machine in total (any run, not only its own: train.py and train_ppo.py,
# counted by experiment name), taking the first line of QUEUE ("ARM CONFIG SEED", as run_lane.sh) each time a slot frees. QUEUE can be edited while it runs:
# add, remove or reorder lines; lines starting with # are skipped. Each started line moves to QUEUE.started with its start time. Written 11 Nov 2026.
# Set AFTER_PID to a process to wait for first, and AFTER_LANE to a lane file whose run_lane.sh must have ended.
Q=$1; W=$2
cd "$(dirname "$0")"
if [ -n "$AFTER_PID" ]; then while kill -0 $AFTER_PID 2>/dev/null; do sleep 60; done; fi
if [ -n "$AFTER_LANE" ]; then while pgrep -f "run_lane.sh $AFTER_LANE" > /dev/null; do sleep 60; done; fi
echo "queue $Q open $(date '+%d %b %H:%M')"
running() { ps -eo args | grep -E "fast/train(_ppo)?\.py" | grep -v grep | grep -oE "exp_name [^ ]+" | sort -u | wc -l; }
while true; do
  line=$(grep -m1 -vE '^[[:space:]]*(#|$)' "$Q")
  if [ -z "$line" ]; then echo "queue $Q empty $(date '+%d %b %H:%M')"; exit 0; fi
  if [ "$(running)" -lt "$W" ]; then
    awk -v L="$line" 'done || $0 != L {print; next} {done=1}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    echo "$line  # started $(date '+%d %b %H:%M')" >> "$Q.started"
    echo "start $line $(date '+%d %b %H:%M')"
    set -- $line
    (setsid nohup ./run_method.sh "$1" "$2" "$3" >> queue_runs.log 2>&1 &)
    sleep 180
  else
    sleep 60
  fi
done
