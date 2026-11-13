#!/bin/bash
# Pulls gpu-13 runs back to the workstation every 10 min (12 Nov 2026): run folders (progress, resolved config with paths rewritten to rpuser1),
# then, once finished, the final checkpoint (copied to a temp name and moved, so waiters never read a partial file) and the deploy evaluations.
# The workstation's gait waiters, wait_more.sh and compare_all.sh then score them as if trained here. Needs SSHPASS (scratchpad ssh_env.sh).
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2; R=/home/rpuser1/yasho/rl/BipedRobot; RR=/home/rpuser2/yasho/rl/BipedRobot; GV=$RR/diagnostics/audit_2026-10-31/claude_audit/v2
RS="timeout 300 sshpass -e rsync -a -e 'ssh -o PubkeyAuthentication=no'"
. /tmp/claude-1009/-home-rpuser1/30f69998-ca43-4838-a45f-92ecfcb4c70e/scratchpad/ssh_env.sh
while true; do
  left=0
  while read -r arm cfg s rest; do
    [ -z "$s" ] && continue; case $arm in \#*) continue;; esac
    N=fast_${arm}_20261102_seed$s; D=$R/checkpoints/$N/seed_$s
    [ -f $D/sb3_her_final.pt ] && [ -f $V/deploy/${arm}_s${s}_medium.json ] && continue
    left=$((left+1))
    mkdir -p $R/runs/$N
    eval $RS rpuser2@192.168.29.76:$RR/runs/$N/ $R/runs/$N/ 2>/dev/null && [ -f $R/runs/$N/seed_$s/resolved_config.yaml ] && sed -i 's#/home/rpuser2#/home/rpuser1#g' $R/runs/$N/seed_$s/resolved_config.yaml
    if timeout 30 sshpass -e ssh -n -o PubkeyAuthentication=no rpuser2@192.168.29.76 "[ -f $RR/checkpoints/$N/seed_$s/sb3_her_final.pt ]"; then
      mkdir -p $D
      if [ ! -f $D/sb3_her_final.pt ]; then eval $RS rpuser2@192.168.29.76:$RR/checkpoints/$N/seed_$s/sb3_her_final.pt $D/.final.tmp && mv $D/.final.tmp $D/sb3_her_final.pt && echo "pulled $N $(date '+%d %b %H:%M')"; fi
      eval $RS --include="${arm}_s${s}_*" --exclude="*" rpuser2@192.168.29.76:$GV/deploy/ $V/deploy/ 2>/dev/null
    fi
  done < <(cat $V/queue_g13.txt; timeout 30 sshpass -e ssh -n -o PubkeyAuthentication=no rpuser2@192.168.29.76 "cat $GV/queue_g13.txt.started" 2>/dev/null)
  [ $left -eq 0 ] && { echo "all gpu-13 runs pulled $(date '+%d %b %H:%M')"; exit 0; }
  sleep 600
done
