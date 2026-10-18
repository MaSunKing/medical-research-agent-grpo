#!/usr/bin/env bash
set -euo pipefail
# The batch shell forwards the advance warning to the Python child via its shell.
child=''
forward() { if [ -n "$child" ]; then kill -USR1 -- "-$child" 2>/dev/null || true; fi; }
trap forward USR1 TERM INT
setsid bash "$1/run_phase.sh" "$2" &
child=$!
set +e
wait "$child"
rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
exit "$rc"
