#!/bin/bash
# Evaluates the finished SB3 seed 3 full-realism replication the same way as the E-A runs.
cd /home/rpuser1/yasho/rl; B=BipedRobot; V=$B/diagnostics/audit_2026-10-31/audit/v2; PY=/home/rpuser1/miniconda3/envs/rl/bin/python
N=walk_v2_min_full_20261101_seed3; L=Full-realism_s3; D=$B/checkpoints/$N/seed_3; CK=$D/sb3_her_final.zip; RC=$B/runs/$N/seed_3/resolved_config.yaml
bash $V/run_benchmark.sh $L $CK $RC 3
ev() { nice -n 5 $PY $B/evaluation_suite_v2.py --checkpoint $1 --resolved-config $RC --realism medium --command 0.2 0 0 --seeds $2 --label $L --out $3 > /dev/null 2>&1 || echo "FAIL eval $3"; }
ev $CK 5100:5200 $V/ea/${L}_final_medium_0.2_seeds5100-5199.json
for K in 800000 900000; do ev $D/sb3_her_model_${K}_steps.zip 5000:5030 $V/ea/${L}_${K}_medium_0.2.json; done
nice -n 5 $PY $V/jitter.py $L $CK $RC 0.2 $V/jitter/${L}_medium_0.2.json 5000:5010 medium > /dev/null 2>&1 || echo "FAIL jitter"
nice -n 5 $PY $V/paths.py $L $CK $RC medium 0.2 $V/paths/${L}_medium_0.2.json > /dev/null 2>&1 || echo "FAIL paths"
echo "evaluated $L $(date +%H:%M)"
