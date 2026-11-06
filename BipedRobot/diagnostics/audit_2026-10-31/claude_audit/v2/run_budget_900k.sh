#!/bin/bash
# Tests the stopped old-robot budget runs at their 900k checkpoint. No training.
cd /home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
for S in 3 4 5; do N=fast_budget_20261102_seed$S; echo "Fast-budget_s$S BipedRobot/checkpoints/$N/seed_$S/sb3_her_model_900000_steps.pt BipedRobot/runs/$N/seed_$S/resolved_config.yaml medium $V/budget/Fast-budget_s${S}_900k_medium.json"; done | xargs -P 3 -L 1 nice -n 5 $PY $V/budget_eval.py
echo "budget 900k tests done $(date +%H:%M)"
