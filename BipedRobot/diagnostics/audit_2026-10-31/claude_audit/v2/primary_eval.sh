#!/bin/bash
# Primary evaluation (WORKING_RULES 12 Nov 21:0x): runs each arm's 100-episode evaluation on test seeds 10000-10099 as soon as its training ends.
# Outputs go to compare_primary/ and are NOT to be read until every arm is done (blinded); this script only prints completion lines.
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2; R=/home/rpuser1/yasho/rl/BipedRobot; PY=/home/rpuser1/miniconda3/envs/rl/bin/python
O=$V/compare_primary; mkdir -p $O/logs
BUD="r17w:11 r17w:12 r17w:13 r17w:14 r17w:15 r17wS:11 r17wS:12 r17wS:13 r17wS:14 r17wS:15 r17wh:11 r17wh:12 r17wNR:11 r17wNR:12 r17wSF45:11 r17wSF45:12 c17ep:11 c17ep:12 c17ep:13 c17ep:14 c17ep:15"
OFF=""; for a in c17plain c17lagmid c17lagtight c17lagmin c17pen01 c17pen03 c17pen10; do for s in 11 12 13 14 15; do OFF="$OFF $a:$s"; done; done
PPO="ppolong7mc17plain:11 ppolong7mc17plain:12 ppolong7mc17plain:13 ppolong7mc17plain:14 ppolong7mc17plain:15"
UB="1,1,1 0.75,0.75,0.75 0.5,0.5,0.5 0.25,0.25,0.25 0,0,0 0.25,1,1 0,1,1 1,0.25,1 1,0,1 1,1,0.25 1,1,0 0.323,0.285,0.113"
jobs() { ls $O/*.running 2>/dev/null | wc -l; }
start() { # tag arm seed extra...
  local tag=$1 arm=$2 s=$3; shift 3
  [ -f $O/${arm}_s$s$tag.json ] && return 0; [ -f $O/${arm}_s$s$tag.running ] && return 0
  local D=$R/checkpoints/fast_${arm}_20261102_seed$s/seed_$s
  [ -f $D/sb3_her_final.pt ] || [ -f $D/ppo_final.zip ] || return 1
  [ -f $R/runs/fast_${arm}_20261102_seed$s/seed_$s/resolved_config.yaml ] || return 1
  while [ $(jobs) -ge 3 ]; do sleep 60; done
  touch $O/${arm}_s$s$tag.running
  (cd /home/rpuser1/yasho/rl; CUDA_VISIBLE_DEVICES="" nice -n 15 $PY $V/compare_eval.py $arm $s --walks 100 --seed0 10000 --outdir compare_primary --tag "$tag" "$@" > $O/logs/${arm}_s$s$tag.log 2>&1; rm -f $O/${arm}_s$s$tag.running; echo "done $arm s$s$tag $(date '+%d %b %H:%M')") &
  sleep 5; return 0
}
while true; do
  left=0
  for x in $BUD; do a=deploylong7m${x%%:*}; s=${x##*:}; start "" $a $s --u $UB --robot --robot_u 1,1,1 0.5,0.5,0.5 0,0,0 --switch "1,1,1>0,0,0" "0,0,0>1,1,1" || left=$((left+1)); [ -f $O/${a}_s$s.json ] || left=$((left+1)); done
  for x in $OFF; do a=deploylong7m${x%%:*}; s=${x##*:}; start "" $a $s --u 1,1,1 --robot || left=$((left+1)); [ -f $O/${a}_s$s.json ] || left=$((left+1))
    if [ "${x%%:*}" = "c17plain" ]; then for c in 0.04 0.06 0.08; do start "_cmd$c" $a $s --u 1,1,1 --command $c || true; [ -f $O/${a}_s${s}_cmd$c.json ] || left=$((left+1)); done; fi; done
  for x in $PPO; do a=${x%%:*}; s=${x##*:}; start "" $a $s --u 1,1,1 --robot || left=$((left+1)); [ -f $O/${a}_s$s.json ] || left=$((left+1)); done
  [ $left -eq 0 ] && { wait; echo "ALL PRIMARY EVALUATIONS DONE $(date '+%d %b %H:%M')"; exit 0; }
  sleep 300
done
