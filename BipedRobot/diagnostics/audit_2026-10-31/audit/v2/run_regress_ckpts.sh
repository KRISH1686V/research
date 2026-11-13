#!/bin/bash
# Regression across checkpoints: SB3 vs fast, seeds 3 and 4, 800k and 900k, easy 0.2, 30 seeds.
cd /home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; B=BipedRobot; O=$B/diagnostics/audit_2026-10-31/audit/v2/regress_ckpts
for line in "SB3 3 walk_v2_min_20261031_seed3" "SB3 4 walk_v2_min_20261029_seed4" "Fast 3 fast_min_regress_20261101_seed3" "Fast 4 fast_min_regress_20261101_seed4"; do set -- $line
  for K in 800000 900000; do echo "$1 $2 $3 $K"; done; done |
xargs -P 3 -L 1 bash -c 'nice -n 5 '$PY' '$B'/evaluation_suite_v2.py --checkpoint '$B'/checkpoints/$2/seed_$1/sb3_her_model_$3_steps.zip --resolved-config '$B'/runs/$2/seed_$1/resolved_config.yaml --realism easy --command 0.2 0 0 --seeds 5000:5030 --label $0_s$1 --out '$O'/$0_s$1_$3.json > /dev/null 2>&1 || echo FAIL $0 $1 $3'
echo "regression checkpoints done $(date +%H:%M)"
