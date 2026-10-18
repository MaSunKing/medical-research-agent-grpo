#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${PROCESS972_BASE:?Set PROCESS972_BASE to the verified local Qwen3-8B directory}"
: "${PROCESS972_PYTHON:?Set PROCESS972_PYTHON to the training Python executable}"
: "${PROCESS972_WORK:?Set PROCESS972_WORK to the new run directory}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
test -x "$PROCESS972_PYTHON"
case "${1:-}" in
  preflight)
    "$PROCESS972_PYTHON" -B "$PACKAGE/contract.py" --package "$PACKAGE" --base "$PROCESS972_BASE"
    ;;
  tests)
    mkdir -p "$PROCESS972_WORK/tests"
    "$PROCESS972_PYTHON" -B "$PACKAGE/train.py" --base "$PROCESS972_BASE" --output "$PROCESS972_WORK/tests/smoke" --smoke
    "$PROCESS972_PYTHON" -B "$PACKAGE/train.py" --base "$PROCESS972_BASE" --output "$PROCESS972_WORK/tests/reference" --diagnostic-stop 2
    "$PROCESS972_PYTHON" -B "$PACKAGE/train.py" --base "$PROCESS972_BASE" --output "$PROCESS972_WORK/tests/resumed" --diagnostic-stop 1
    "$PROCESS972_PYTHON" -B "$PACKAGE/train.py" --base "$PROCESS972_BASE" --output "$PROCESS972_WORK/tests/resumed" --diagnostic-stop 2
    "$PROCESS972_PYTHON" -B "$PACKAGE/acceptance.py" --package "$PACKAGE" --base "$PROCESS972_BASE" --tests "$PROCESS972_WORK/tests"
    ;;
  train)
    test "${PROCESS972_RESEARCH_ACK:-}" = yes || { echo 'Set PROCESS972_RESEARCH_ACK=yes after reviewing audit caveats'; exit 2; }
    exec "$PROCESS972_PYTHON" -B "$PACKAGE/train.py" --base "$PROCESS972_BASE" --output "$PROCESS972_WORK/training" \
      --acceptance "$PROCESS972_WORK/tests/ACCEPTANCE.json" --seconds 18000 \
      --acknowledge-research-data-not-full-medical-review
    ;;
  *) echo 'Usage: bash run_phase.sh preflight|tests|train'; exit 2;;
esac
