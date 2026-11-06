#!/bin/bash
# V2 robot: budget-HER first, then baselines; seeds 3, 4, 5; at most 4 at once. Waits for the 4 old-model runs still training.
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
for P in 695928 695931 695933 695938; do while kill -0 $P 2>/dev/null; do sleep 20; done; done
echo "$(date) starting V2 runs"
while read -r SCRIPT REST; do echo "$V/$SCRIPT $REST"; done < $V/v2_runs.txt | xargs -P 4 -L 1 bash
echo "$(date) all V2 runs trained"
