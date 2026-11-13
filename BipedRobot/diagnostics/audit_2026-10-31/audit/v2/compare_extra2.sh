#!/bin/bash
# Runs compare_eval.py on every comparison run as its final checkpoint appears (12 Nov 2026; grid, knob switch and 2 s windows added 09:40). Two at a time, CPU only. Budget-HER arms are swept over u; the others are evaluated once (their budgets are off).
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/audit/v2; R=/home/rpuser1/yasho/rl/BipedRobot; PY=/home/rpuser1/miniconda3/envs/rl/bin/python
LIST="deploylong7mr17wSF45:11 deploylong7mr17wSF45:12"
for a in ; do for s in 11 12 13 14 15; do LIST="$LIST $a:$s"; done; done
cd /home/rpuser1/yasho/rl
while true; do
  left=0
  for x in $LIST; do a=${x%%:*}; s=${x##*:}; D=$R/checkpoints/fast_${a}_20261102_seed$s/seed_$s
    [ -f $V/compare/${a}_s$s.json ] && continue
    left=$((left+1))
    [ -f $D/sb3_her_final.pt ] || [ -f $D/ppo_final.zip ] || continue
    [ -f $V/compare/${a}_s$s.running ] && continue
    while [ $(ls $V/compare/*.running 2>/dev/null | wc -l) -ge 2 ]; do sleep 30; done
    touch $V/compare/${a}_s$s.running
    # budget arms: a 9-point grid of u (all three together), the robot-like line at 3, and the knob turned mid-walk both ways; the others once (budgets off)
    case $a in *r17w*|*c17ep*) EX="--u 1,1,1 0.875,0.875,0.875 0.75,0.75,0.75 0.625,0.625,0.625 0.5,0.5,0.5 0.375,0.375,0.375 0.25,0.25,0.25 0.125,0.125,0.125 0,0,0 --robot_u 1,1,1 0.5,0.5,0.5 0,0,0 --switch 1,1,1>0,0,0 0,0,0>1,1,1";; *) EX="--u 1,1,1";; esac
    (CUDA_VISIBLE_DEVICES="" nice -n 10 $PY $V/compare_eval.py $a $s $EX --robot > $V/compare/${a}_s$s.log 2>&1; rm -f $V/compare/${a}_s$s.running; echo "done $a s$s $(date '+%d %b %H:%M')") &
    sleep 5
  done
  [ $left -eq 0 ] && { wait; echo "all compared $(date '+%d %b %H:%M')"; exit 0; }
  sleep 120
done
