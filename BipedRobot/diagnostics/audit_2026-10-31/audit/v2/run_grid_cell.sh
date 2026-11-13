#!/bin/bash
# One UTD grid cell: train, then evaluate at 300k, 500k and final with evaluation_suite_v2 (easy 0.2, 30 seeds).
# Memory: evaluations run one at a time, and only one 16-env cell trains at a time.
# Usage: run_grid_cell.sh N_ENVS BATCH UPDATES_PER_ENV_STEP SEED
set -u
E=$1; B=$2; G=$3; S=$4
R=/home/rpuser1/yasho/rl; PY=/home/rpuser1/miniconda3/envs/rl/bin/python; V=BipedRobot/diagnostics/audit_2026-10-31/audit/v2
N=fast_grid2_e${E}_b${B}_g${G}_seed$S
cd $R/BipedRobot
if [ "$E" = 16 ]; then exec 9> $R/$V/grid_e16.lock; flock 9; fi
nice -n 5 $PY fast/train.py --config config/pilot_gait_min_v2.yaml --exp_name $N --seed $S --n_envs $E --batch_size $B --updates_per_env_step $G \
  --total_steps 1000000 --realism easy --mode graph --log_dir runs --checkpoint_dir checkpoints --max_wall_s 7200 > runs/$N.log 2>&1 9>&- || echo "FAIL train $N"
if [ "$E" = 16 ]; then flock -u 9; fi
cd $R
for K in 300000 500000 final; do
  if [ $K = final ]; then C=sb3_her_final.zip; else C=sb3_her_model_${K}_steps.zip; fi
  CK=BipedRobot/checkpoints/$N/seed_$S/$C
  [ -f $CK ] || continue
  nice -n 5 $PY BipedRobot/evaluation_suite_v2.py --checkpoint $CK --resolved-config BipedRobot/runs/$N/seed_$S/resolved_config.yaml --realism easy --command 0.2 0 0 \
    --seeds 5000:5030 --label $N --out $V/grid/${N}_$K.json > /dev/null 2>&1 || echo "FAIL eval $N $K"
done
echo "done $N $(date +%H:%M)"
