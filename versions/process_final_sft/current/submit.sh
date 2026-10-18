#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${PF600_WORK:?set PF600_WORK to the prepared WORK}"
: "${PF600_PYTHON:?set PF600_PYTHON to evaluation Python}"
PF600_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PHASE=${1:-pilot}
case "$PHASE" in pilot|all|export) ;; *) echo 'pilot, all or export required'; exit 1;; esac
DEPENDENCY=()
if [ -n "${2:-}" ]; then DEPENDENCY=(--dependency="afterany:$2"); fi
GPU=(--gres=gpu:1)
if [ "$PHASE" = export ]; then GPU=(); fi
export PF600_WORK PF600_PYTHON PF600_PACKAGE PHASE
JOB=$(sbatch --parsable --job-name=pf600-collect --partition=MGPU-TC2 --qos=normal \
  --time=05:45:00 --cpus-per-task=8 --mem=30G "${GPU[@]}" "${DEPENDENCY[@]}" \
  --signal=B:USR1@4500 --output="$PF600_WORK/logs/job_%j.out" --error="$PF600_WORK/logs/job_%j.err" \
  --export=ALL "$PF600_PACKAGE/slurm_entry.sh")
JOB=${JOB%%;*}
printf 'JOB=%s PHASE=%s\nWORK=%s\n' "$JOB" "$PHASE" "$PF600_WORK"
printf '\nwatch -n 60 -t "sacct -X -j %s --format=JobID,State,Elapsed,ExitCode; '\''%s'\'' -B '\''%s/control.py'\'' --work '\''%s'\'' status"\n' "$JOB" "$PF600_PYTHON" "$PF600_PACKAGE" "$PF600_WORK"
echo 'Ctrl+C exits watch only. Pilot does not automatically submit all.'
