#!/bin/bash
# Speed-matched control (12 Nov 2026, Q1 review): budgets-off policies at lower commands, screening only (20 walks, seeds 9800-9819), one at a time.
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/audit/v2; PY=/home/rpuser1/miniconda3/envs/rl/bin/python
cd /home/rpuser1/yasho/rl
for s in 11 12 13 14; do for c in 0.04 0.06 0.08; do
  CUDA_VISIBLE_DEVICES="" nice -n 15 $PY $V/compare_eval.py deploylong7mc17plain $s --command $c --tag _cmd$c >> $V/compare/speedctl.log 2>&1
done; done
echo "speed control done $(date '+%d %b %H:%M')" >> $V/compare/speedctl.log
