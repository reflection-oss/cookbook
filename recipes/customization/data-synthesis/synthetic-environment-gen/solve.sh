#!/usr/bin/env bash
# Stage 3: the no-op control and the Beam solve runs, both through Harbor.
# Usage: bash solve.sh [harbor_dir] [jobs_dir]
# Env: ATTEMPTS (default 2), TRIALS (parallel trials, default 8), MODEL (default Beam-1.0-501B-A23B),
#      API_BASE / API_KEY (default: the REFLECTION_* variables from .env), LANES ("control solve" by default;
#      "control" runs only the no-op check, which needs no API)
set -euo pipefail
cd "$(dirname "$0")"
for env in .env ../../../../.env; do
  # shellcheck disable=SC1090,SC1091
  [[ -f "$env" ]] && set -a && source "$env" && set +a
done
HARBOR_DIR="${1:-output/harbor}"
JOBS_DIR="${2:-output/jobs}"
ATTEMPTS="${ATTEMPTS:-2}"
TRIALS="${TRIALS:-8}"
LANES="${LANES:-control solve}"
MODEL="${MODEL:-${REFLECTION_MODEL:-reflection/Beam-1.0-501B-A23B}}"
# mini-swe-agent talks to the model through litellm, which has no "reflection/" provider yet: the
# Reflection API (OpenAI-compatible) is reached through its "openai/" provider and the two variables
# below. Accept the model name with either prefix, or none, and pass litellm exactly one.
MODEL="${MODEL#reflection/}"
MODEL="${MODEL#openai/}"
LITELLM_MODEL="openai/$MODEL"
export OPENAI_BASE_URL="${API_BASE:-${REFLECTION_API_BASE:-https://api.reflection.ai/openai/v1}}"
export OPENAI_API_KEY="${API_KEY:-${REFLECTION_API_KEY:-}}"
export HARBOR_TELEMETRY=0 ATTEMPTS
mkdir -p "$JOBS_DIR" "$JOBS_DIR/_stage"

# A job that stopped without its .exit marker (the shell or the machine died) would keep its tasks
# in a "running" bucket forever. If no harbor process is running it any more, mark it finished; its
# unfinished trials then count as lost and are rerun.
for job in "$JOBS_DIR"/control-*/ "$JOBS_DIR"/solve-*/; do
  [[ -d "$job" ]] || continue
  job="${job%/}"
  [[ -e "$job.exit" ]] || pgrep -f -- "--job-name $(basename "$job")" >/dev/null || echo interrupted >"$job.exit"
done

# stage <lane> <bucket>: copy the tasks in <bucket> into a fresh dataset folder with the limits from
# config/task_overrides.toml written into each task.toml; prints its path, or nothing. Harbor's -k
# is per job, so a task that already has one completed attempt may get ATTEMPTS more on a rerun;
# export.py keeps at most MAX_PER_TASK traces per task regardless.
stage() {
  local dir="" t
  while read -r t; do
    [[ -n "$t" ]] || continue
    if [[ -z "$dir" ]]; then
      dir="$JOBS_DIR/_stage/$1-$(date +%Y%m%d-%H%M%S)"
      mkdir -p "$dir"
    fi
    cp -r "$HARBOR_DIR/$t" "$dir/"
    python3 state.py limits "$dir/$t/task.toml" # agent, verifier, build limits and resources from config/task_overrides.toml
  done < <(python3 state.py "$HARBOR_DIR" "$JOBS_DIR" list "$2")
  [[ -n "$dir" ]] && echo "$dir"
  return 0
}

# run <lane> <dataset> <harbor args...>: one Harbor job; the .exit file marks it finished for state.py.
run() {
  local lane="$1" dataset="$2"
  shift 2
  local job
  job="$lane-$(date +%Y%m%d-%H%M%S)"
  echo "solve: $lane on $(find "$dataset" -mindepth 1 -maxdepth 1 -type d | wc -l) tasks -> $JOBS_DIR/$job"
  local rc=0
  harbor run -e docker -p "$dataset" -n "$TRIALS" -o "$JOBS_DIR" --job-name "$job" --quiet "$@" || rc=$?
  echo "$rc" >"$JOBS_DIR/$job.exit"
}

if [[ " $LANES " == *" control "* ]] && dataset=$(stage control need_control) && [[ -n "$dataset" ]]; then
  run control "$dataset" -a nop -k 1
fi
if [[ " $LANES " == *" solve "* ]] && dataset=$(stage solve need_solve) && [[ -n "$dataset" ]]; then
  run solve "$dataset" -a mini-swe-agent -k "$ATTEMPTS" -m "$LITELLM_MODEL" \
    --ak version=2.4.6 --ak "config=$(cat config/agent.json)" --agent-setup-timeout-multiplier 3
fi
python3 state.py "$HARBOR_DIR" "$JOBS_DIR"
