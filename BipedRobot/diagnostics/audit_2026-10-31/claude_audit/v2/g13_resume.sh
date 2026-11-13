#!/bin/bash
# Waits for gpu-13 to come back after an outage and resumes its queue (12 Nov 2026, owner: "check gpu-13, if it's idle train on it").
# Every 5 min: try ssh. When it answers: (1) if any other user's process holds more than 2 GB of the GPU, start nothing and keep waiting;
# (2) every started line in queue_g13.txt.started whose run has no final checkpoint and no live process is a run killed by the outage: its run and
# checkpoint folders move to _killed_outage_<time> (never deleted) and the line goes back to the front of queue_g13.txt; arms stopped on purpose
# (r17wSF with floor 0.5, c17lagloose) are never requeued; (3) start the MPS daemon if it is not running; (4) start run_queue.sh (width 3) if not running.
# Logs to g13_resume.log here. Needs SSHPASS (scratchpad ssh_env.sh).
V=/home/rpuser1/yasho/rl/BipedRobot/diagnostics/audit_2026-10-31/claude_audit/v2
. /tmp/claude-1009/-home-rpuser1/30f69998-ca43-4838-a45f-92ecfcb4c70e/scratchpad/ssh_env.sh
SSH="timeout 120 sshpass -e ssh -n -o PubkeyAuthentication=no -o ConnectTimeout=10 rpuser2@192.168.29.76"
log() { echo "$(date '+%d %b %H:%M') $*" >> $V/g13_resume.log; }
log "waiting for gpu-13"
while true; do
  if $SSH true 2>/dev/null; then
    log "gpu-13 answers: $($SSH uptime 2>/dev/null)"
    other=$($SSH 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=", " read pid mem; do u=$(ps -o user= -p $pid 2>/dev/null); [ "$u" != "rpuser2" ] && [ "${mem:-0}" -gt 2048 ] && echo "$pid:$u:$mem"; done')
    if [ -n "$other" ]; then log "GPU used by another user ($other); not starting, will check again"; sleep 600; continue; fi
    $SSH 'R=/home/rpuser2/yasho/rl/BipedRobot; V=$R/diagnostics/audit_2026-10-31/claude_audit/v2; cd $V
      running=$(ps -eo args | grep -E "fast/train.py" | grep -v grep | grep -oE "exp_name [^ ]+" | cut -d" " -f2 | sort -u)
      K=$R/_killed_outage_$(date +%Y%m%d_%H%M); front=""
      while read -r arm cfg s rest; do
        [ -z "$s" ] && continue; case "$arm" in \#*) continue;; deploylong7mr17wSF|deploylong7mc17lagloose) continue;; esac
        N=fast_${arm}_20261102_seed$s
        [ -f $R/checkpoints/$N/seed_$s/sb3_her_final.pt ] && continue
        echo "$running" | grep -qx "$N" && continue
        [ -d $R/runs/$N ] || [ -d $R/checkpoints/$N ] || continue
        mkdir -p $K/runs $K/checkpoints
        [ -d $R/runs/$N ] && mv $R/runs/$N $K/runs/; [ -d $R/checkpoints/$N ] && mv $R/checkpoints/$N $K/checkpoints/; [ -f $R/runs/$N.log ] && mv $R/runs/$N.log $K/runs/
        front="$front$arm $cfg $s\n"; echo "requeued $N"
      done < queue_g13.txt.started
      if [ -n "$front" ]; then { printf "$front"; cat queue_g13.txt; } > queue_g13.txt.new && mv queue_g13.txt.new queue_g13.txt; fi
      export CUDA_MPS_PIPE_DIRECTORY=/home/rpuser2/.biped_mps/pipe CUDA_MPS_LOG_DIRECTORY=/home/rpuser2/.biped_mps/log
      if ! pgrep -x nvidia-cuda-mps-control > /dev/null; then mkdir -p $CUDA_MPS_LOG_DIRECTORY; rm -f $CUDA_MPS_PIPE_DIRECTORY/control; nvidia-cuda-mps-control -d; sleep 2; echo "MPS started"; fi
      if ! pgrep -f "run_queue.sh queue_g1[3]" > /dev/null; then echo "# restarted after an outage $(date "+%d %b %H:%M")" >> queue_g13.log; (setsid nohup ./run_queue.sh queue_g13.txt 3 >> queue_g13.log 2>&1 &); echo "queue restarted"; fi
      echo "queue lines: $(grep -cvE "^[[:space:]]*(#|$)" queue_g13.txt)"' >> $V/g13_resume.log 2>&1
    log "resume done"
    exit 0
  fi
  sleep 300
done
