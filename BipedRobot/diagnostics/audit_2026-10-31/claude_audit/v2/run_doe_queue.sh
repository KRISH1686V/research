#!/bin/bash
# Stage 1 screening: 16 designed combinations, 2 seeds each, at most 4 at once. First replicate (seed 3) runs first.
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
echo "$(date) starting screening runs"
while read -r SCRIPT REST; do echo "$V/$SCRIPT $REST"; done < $V/doe_runs.txt | xargs -P 4 -L 1 bash
echo "$(date) all screening runs trained"
