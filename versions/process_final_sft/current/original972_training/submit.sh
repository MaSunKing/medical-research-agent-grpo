#!/usr/bin/env bash
set -euo pipefail
PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${PROCESS972_WORK:?Set a unique new run directory, reuse it only to resume this package}"
PHASE=${1:?Usage: bash submit.sh tests|train [dependency-job-id]}
case "$PHASE" in tests|train) ;; *) exit 2;; esac
if [ "$PHASE" = train ]; then
  test -f "$PROCESS972_WORK/tests/ACCEPTANCE.json"
  test "${PROCESS972_RESEARCH_ACK:-}" = yes
fi
mkdir -p "$PROCESS972_WORK/logs"
DEPENDENCY=()
if [ -n "${2:-}" ]; then
  [[ "$2" =~ ^[0-9]+$ ]] || exit 2
  DEPENDENCY=(--dependency="afterany:$2")
fi
JOB=$(sbatch --parsable --job-name="p972-$PHASE" --partition=MGPU-TC2 --qos=normal \
  --time=05:30:00 --gres=gpu:1 --cpus-per-task=8 --mem=30G --signal=B:USR1@900 \
  --output="$PROCESS972_WORK/logs/job_%j.out" --error="$PROCESS972_WORK/logs/job_%j.err" \
  "${DEPENDENCY[@]}" --export=ALL "$PACKAGE/slurm_entry.sh" "$PACKAGE" "$PHASE")
printf 'JOB=%s\nWORK=%s\nPHASE=%s\n' "$JOB" "$PROCESS972_WORK" "$PHASE"
