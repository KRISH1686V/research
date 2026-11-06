#!/bin/bash
# Runs the UTD grid two cells at a time. Turn and smooth stay paused to respect the 4-run cap and resume on exit, however it ends.
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
PAUSED=""
for M in 3766762 3781345; do kill -0 $M 2>/dev/null && PAUSED="$PAUSED $M $(ps -o pid= --ppid $M | tr '\n' ' ')"; done
trap '[ -n "$PAUSED" ] && kill -CONT $PAUSED 2>/dev/null; echo "$(date) resumed: $PAUSED"' EXIT
[ -n "$PAUSED" ] && kill -STOP $PAUSED
echo "$(date) paused: $PAUSED"
xargs -P 2 -L 1 bash $V/run_grid_cell.sh < $V/grid_cells.txt
echo "$(date) grid finished"
