#!/bin/bash
# Waits for the last grid cells to finish training, then runs the E-A runs two at a time.
for P in  74615; do while kill -0 $P 2>/dev/null; do sleep 20; done; done
echo "$(date) starting E-A runs"
xargs -P 2 -L 1 bash /home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/audit/v2/run_ea_arm.sh < /home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/audit/v2/ea_runs.txt
echo "$(date) all E-A runs trained"
