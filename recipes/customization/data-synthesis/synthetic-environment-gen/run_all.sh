#!/usr/bin/env bash
# Generate tasks with TMax, convert them for Harbor, run the control and solve jobs, export the traces.
# Every stage skips what an earlier run finished, so rerun this script to resume or to grade new tasks.
# Needs TMAX_DIR (a TMax checkout, installed as its README says) and REFLECTION_API_KEY (see .env.example).
# Any value in config/generate.env can be overridden from the shell: NUM_TASKS=10 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
for env in .env ../../../../.env; do
  # shellcheck disable=SC1090,SC1091
  [[ -f "$env" ]] && set -a && source "$env" && set +a
done
while IFS='=' read -r key value; do
  [[ -n "$key" ]] && export "$key=${!key:-$value}" # the environment wins over the file
done < <(sed -e 's/[[:space:]]*#.*//' -e '/^[[:space:]]*$/d' config/generate.env)
: "${TMAX_DIR:?set TMAX_DIR to a TMax checkout}"
# TMax and mini-swe-agent talk to the model through litellm, which has no "reflection/" provider yet:
# the Reflection API (OpenAI-compatible) is reached through litellm's "openai/" provider and these
# two variables. solve.sh does the same mapping for Harbor.
export OPENAI_API_BASE="${REFLECTION_API_BASE:-https://api.reflection.ai/openai/v1}"
export OPENAI_API_KEY="$REFLECTION_API_KEY"
export OUT_DIR="$PWD/$OUT_DIR"

# 1. Generate: TMax's four stages, checkpointed under $OUT_DIR (_intermediates.jsonl, _stage4_done.jsonl).
(cd "$TMAX_DIR" && MODEL="openai/${MODEL#reflection/}" bash rl_data/scripts/generate_tasks/run_generate_tasks.sh)

# 2. Convert the tasks that are not in output/harbor yet with TMax's own Harbor converter.
mkdir -p output/harbor
new_tasks=$(mktemp -d)
for task in "$OUT_DIR"/task_*/; do
  [[ -d "$task" ]] || continue # no tasks at all: the glob did not expand
  name=$(basename "$task")
  [[ -f "output/harbor/$name/task.toml" ]] || ln -s "$task" "$new_tasks/$name"
done
if [[ -n "$(ls -A "$new_tasks")" ]]; then
  "$TMAX_DIR/.venv/bin/python" "$TMAX_DIR/rl_data/scripts/analyze/convert_to_harbor.py" \
    --src "$new_tasks" --dst output/harbor
fi
rm -rf "$new_tasks"

# 3. Control and solve through Harbor (task limits from config/task_overrides.toml are applied here).
bash solve.sh output/harbor output/jobs

# 4. Export the verified traces.
python3 export.py output/harbor output/jobs output
