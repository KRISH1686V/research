#!/bin/bash
# Runs every validity condition on 5 policies, at most 8 at once.
cd /home/rpuser1/yasho/rl; B=BipedRobot; V=$B/diagnostics/audit_2026-10-31/claude_audit/v2; PY=/home/rpuser1/miniconda3/envs/rl/bin/python
CONDS=$($PY -c "import ast,re;s=open('$V/validity_audit.py').read();print(' '.join(re.findall(r'^    \"([^\"]+)\": dict',s,re.M)))")
for line in "Fast-shaped_s3 fast_full_shaped_20261101_seed3 3" "Fast-shaped_s4 fast_full_shaped_20261101_seed4 4" "Fast-shaped_s5 fast_full_shaped_20261101_seed5 5" "SB3-full_s3 walk_v2_min_full_20261101_seed3 3" "Fast-smooth_s5 fast_full_smooth_20261101_seed5 5"; do set -- $line
  for C in $CONDS; do echo "$1 $B/checkpoints/$2/seed_$3/sb3_her_final.zip $B/runs/$2/seed_$3/resolved_config.yaml $C $V/validity/audit/$1__$C.json"; done
done | xargs -P 8 -L 1 nice -n 5 $PY $V/validity_audit.py
echo "audit done $(date +%H:%M)"
