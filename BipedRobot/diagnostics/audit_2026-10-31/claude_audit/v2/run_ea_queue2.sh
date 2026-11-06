#!/bin/bash
# FastSAC fixes and the speed fix, one change at a time against the shaped arm: seeds 3, 4, 5, full realism, 1M steps.
# Waits for the SB3 no-gait run so at most 4 training runs are active, and evaluates it when it ends.
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2; PY=/home/rpuser1/miniconda3/envs/rl/bin/python
while kill -0 4161573 2>/dev/null; do sleep 20; done
echo "$(date) SB3 no-gait run ended"
cd /home/rpuser1/yasho/rl
N=walk_v2_min_nogait_20261101_seed4
nohup nice -n 5 bash $V/run_benchmark.sh Nogait-sb3_s4 BipedRobot/checkpoints/$N/seed_4/sb3_her_final.zip BipedRobot/runs/$N/seed_4/resolved_config.yaml 3 >> $V/ea_evals.log 2>&1 &
echo "$(date) starting runs"
TAG=20261102 xargs -P 4 -L 1 bash $V/run_ea_arm.sh < $V/ea_runs2.txt
echo "$(date) all runs trained"
