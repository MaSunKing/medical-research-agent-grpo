#!/bin/bash
# Login-shell submission only. Checks and work run on allocated Slurm nodes.
# Usage: bash submit.sh PHASE PACKAGE WORK PYTHON [EXTRA]
# cpu EXTRA=optional CONFIG for fresh prepare; grade EXTRA=--allow-judge-api;
# export EXTRA=new external export directory; smoke/run take no EXTRA.
set -euo pipefail
umask 077

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ];then
 echo 'Usage: bash submit.sh cpu|smoke|run|grade-smoke|grade|export PACKAGE WORK PYTHON [EXTRA]' >&2
 exit 1
fi
PE_PHASE=$1
PE_PACKAGE=$(cd "$2" && pwd -P)
PE_WORK=$(realpath -m -- "$3")
PE_PY=$(realpath -e -- "$4")
PE_EXTRA=${5:-}
PE_SECRETS=
test -x "$PE_PY"
test -f "$PE_PACKAGE/FILES.sha256.json"
test -f "$PE_PACKAGE/eval/offline.py"
case "$PE_WORK" in /|'') echo 'A dedicated evaluation WORK directory is required.' >&2;exit 1;; esac
case "$PE_WORK/" in "$PE_PACKAGE/"*) echo 'WORK must be outside the immutable package.' >&2;exit 1;; esac
for VALUE in "$PE_PACKAGE" "$PE_WORK" "$PE_PY" "$PE_EXTRA";do
 case "$VALUE" in *$'\n'*|*$'\r'*) echo 'Newlines are not allowed in task paths.' >&2;exit 1;; esac
done

OPTIONS=()
CPUS=1
MEM=1G
case "$PE_PHASE" in
 cpu)
  TIME=00:30:00
  if [ -n "$PE_EXTRA" ];then
   PE_EXTRA=$(realpath -e -- "$PE_EXTRA")
   test -f "$PE_EXTRA"
   test ! -e "$PE_WORK" || {
    echo 'cpu with CONFIG prepares fresh WORK; omit CONFIG for offline-only recheck.' >&2;exit 1;
   }
  fi;;
 smoke|run)
  test -z "$PE_EXTRA"
  test -f "$PE_WORK/IDENTITY.json"
  test -f "$PE_WORK/CONFIG.json"
  TIME=05:45:00;CPUS=8;MEM=28G
  OPTIONS+=(--gres=gpu:1 '--exclude=TC2N[03-08]');;
 grade|grade-smoke)
  test "$PE_EXTRA" = --allow-judge-api || {
   echo 'grade requires explicit fifth argument --allow-judge-api.' >&2;exit 1;
  }
  : "${EVAL_JUDGE_SECRETS:?Set EVAL_JUDGE_SECRETS to the external credential-file path only.}"
  PE_SECRETS=$(realpath -e -- "$EVAL_JUDGE_SECRETS")
  test -f "$PE_SECRETS"
  case "$PE_SECRETS/" in "$PE_PACKAGE/"*|"$PE_WORK/"*)
   echo 'Credential file must be outside package and WORK; never export it.' >&2;exit 1;;
  esac
  test -f "$PE_WORK/IDENTITY.json"
  TIME=05:45:00;;
 export)
  test -n "$PE_EXTRA"
  PE_EXTRA=$(realpath -m -- "$PE_EXTRA")
  test ! -e "$PE_EXTRA"
  case "$PE_EXTRA/" in "$PE_PACKAGE/"*|"$PE_WORK/"*)
   echo 'Export output must be a fresh external directory.' >&2;exit 1;;
  esac
  test -f "$PE_WORK/IDENTITY.json"
  TIME=02:00:00;;
 *) echo 'Unknown phase; no submission.' >&2;exit 1;;
esac

# Sibling logs preserve fresh-WORK preparation and locked export snapshots.
PE_OPS="$PE_WORK.ops"
mkdir -p "$PE_OPS/logs"
exec 9>"$PE_OPS/.submit.lock"
flock -n 9 || { echo 'Another submission is in progress.' >&2;exit 1; }
export PE_PHASE PE_PACKAGE PE_WORK PE_PY PE_EXTRA PE_SECRETS PE_OPS
JOB=$(sbatch --parsable --job-name="newbench-eval-$PE_PHASE" \
 --partition=MGPU-TC2 --qos=normal --nodes=1 --ntasks=1 \
 --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
 --signal=B:USR1@180 \
 --output="$PE_OPS/logs/${PE_PHASE}_%j.out" \
 --error="$PE_OPS/logs/${PE_PHASE}_%j.err" \
 --export=ALL "${OPTIONS[@]}" <<'SH'
#!/bin/bash
set -euo pipefail
umask 077
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID"
echo "PHASE=$PE_PHASE"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false
if [ "$PE_PHASE" = run ] || [ "$PE_PHASE" = smoke ];then
 export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
 export CUBLAS_WORKSPACE_CONFIG=:4096:8
else
 export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fi
cd "$PE_OPS"
echo 'STAGE=OFFLINE_PACKAGE_CHECKS'
"$PE_PY" -B "$PE_PACKAGE/eval/offline.py" --package "$PE_PACKAGE"

case "$PE_PHASE" in
 cpu)
  echo 'STAGE=OFFLINE_TESTS'
  "$PE_PY" -B -m unittest discover -s "$PE_PACKAGE/eval" -p 'test_*.py' -q
  if [ -n "$PE_EXTRA" ];then
   echo 'STAGE=PREPARE_FRESH_EVALUATION_WORK'
   "$PE_PY" -B "$PE_PACKAGE/eval/evaluate.py" prepare \
    --package "$PE_PACKAGE" --config "$PE_EXTRA" --work "$PE_WORK"
  fi
  echo 'CPU_CHECKS_COMPLETE; MODEL=0; RETRIEVAL_API=0; JUDGE_API=0; no automatic next phase'
  exit 0;;
 smoke|run)
  ARGS=("$PE_PACKAGE/eval/evaluate.py" run --package "$PE_PACKAGE" \
   --work "$PE_WORK" --real-gpu --seconds 19800)
  if [ "$PE_PHASE" = smoke ];then ARGS+=(--indices 1 51 --pilot-only);fi
  echo 'STAGE=PROCESS_EVALUATION; new_arms=base,sft,grpo13,grpo25; Final=frozen_base';;
 grade|grade-smoke)
  test "$PE_EXTRA" = --allow-judge-api
  ARGS=("$PE_PACKAGE/eval/grading.py" --package "$PE_PACKAGE" \
   --work "$PE_WORK" --output "$PE_WORK/grades" --real-judge-api \
   --secrets-file "$PE_SECRETS" --seconds 19800)
  if [ "$PE_PHASE" = grade-smoke ];then ARGS+=(--indices 1 51);fi
  echo 'STAGE=BENCHMARK_GRADING; replacement_judge; explicit_API_opt_in';;
 export)
  ARGS=("$PE_PACKAGE/eval/review_export.py" export --package "$PE_PACKAGE" \
   --work "$PE_WORK" --output "$PE_EXTRA")
  echo 'STAGE=EXPORT_AND_VERIFY; no model or API';;
esac
"$PE_PY" -B "${ARGS[@]}" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true' USR1 TERM INT
set +e
wait "$CHILD";CODE=$?
if kill -0 "$CHILD" 2>/dev/null;then wait "$CHILD";CODE=$?;fi
if [ "$CODE" -eq 0 ] && [ "$PE_PHASE" = smoke ];then
 "$PE_PY" -B "$PE_PACKAGE/eval/ops.py" pilot-check --package "$PE_PACKAGE" --work "$PE_WORK"
fi
if [ "$CODE" -eq 0 ] && [ "$PE_PHASE" = grade-smoke ];then
 "$PE_PY" -B "$PE_PACKAGE/eval/ops.py" judge-check --package "$PE_PACKAGE" --work "$PE_WORK"
fi
exit "$CODE"
SH
)
JOB=${JOB%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]]
printf '%s\n' "$JOB" > "$PE_OPS/${PE_PHASE}_LATEST_JOB.txt"
printf 'JOB=%s\nPHASE=%s\nLOG_DIR=%s\n' "$JOB" "$PE_PHASE" "$PE_OPS/logs"
MONITOR='sacct -X -j "$1" --format=JobID,State,Elapsed,AllocCPUS,NodeList
echo "=== Saved activity / counts; may be stale ==="
for F in "$3/ACTIVE.json" "$3/STATUS.json" "$3/grades/PASS_STATUS.json";do
 if [ -f "$F" ];then
  grep -E '\''"(arm|index|stage|finished|expected|complete|process_failure|final_failure|engineering_failure|interrupted|missing|scored|deferred|unavailable|engineering_hold)"'\'' "$F" | tr "\n" " " | cut -c1-170
  echo
 fi
done
echo "=== Recent progress ==="
grep -E '\''^(NODE=|PHASE=|STAGE=|PREPARED|QUESTION_FINISHED|GRADE |GRADING_PASS_COMPLETE|EVALUATION_SLICE_COMPLETE|PAUSED_|ARCHIVE=|SHA256_FILE=|EXPORT_|CPU_CHECKS_COMPLETE)'\'' "$2.out" 2>/dev/null | tail -n 4 | cut -c1-170
echo "=== Errors ==="
tail -n 2 "$2.err" 2>/dev/null | cut -c1-170'
printf '\nwatch -n 60 -t --exec bash -c %q -- %q %q %q\n' \
 "$MONITOR" "$JOB" "$PE_OPS/logs/${PE_PHASE}_$JOB" "$PE_WORK"
echo 'Ctrl+C exits watch only; submitted job continues. No automatic next phase.'
