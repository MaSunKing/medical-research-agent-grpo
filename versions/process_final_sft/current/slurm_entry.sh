#!/usr/bin/env bash
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID PHASE=$PHASE"
if [ "$PHASE" = export ]; then
  exec "$PF600_PYTHON" -B "$PF600_PACKAGE/export_results.py" --work "$PF600_WORK"
fi
exec "$PF600_PYTHON" -B "$PF600_PACKAGE/collect.py" --work "$PF600_WORK" "$PHASE"
