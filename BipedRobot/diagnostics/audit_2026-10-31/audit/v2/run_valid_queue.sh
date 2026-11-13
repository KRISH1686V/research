#!/bin/bash
# Budget-HER pilot first, then its baselines on the corrected simulation; seeds 3, 4, 5; at most 4 runs at once.
# Waits for the 4 old-simulation runs still training.
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/audit/v2
for P in 635809 635810 635811 635813; do while kill -0 $P 2>/dev/null; do sleep 20; done; done
echo "$(date) starting runs"
while read -r SCRIPT REST; do echo "$V/$SCRIPT $REST"; done < $V/valid_runs.txt | xargs -P 4 -L 1 bash
echo "$(date) all runs trained"
