#!/usr/bin/env bash
# Generate and score one registered model on BFCL v4: single-turn,
# multi-turn and memory. Waits until the endpoint answers a tool-call probe
# three times in a row, since a server that is still loading or restarting
# would fail entries.
#
#   bash eval_run.sh <project_root> <registry model name>
#
# The project root holds a `.env` with REMOTE_OPENAI_BASE_URL (and
# REMOTE_OPENAI_API_KEY if the server needs one). Results land in
# <project_root>/result and scores in <project_root>/score. Run one
# `bfcl generate` per project root at a time.
set -euo pipefail

ROOT="${1:?project root}"
MODEL="${2:?registry model name}"
CATS="${CATS:-single_turn,multi_turn,memory}"
export BFCL_PROJECT_ROOT="$ROOT"
cd "$ROOT"
set -a
# shellcheck source=/dev/null
source .env
set +a

probe='{"model":"MODEL","messages":[{"role":"user","content":"Weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":256}'
ok=0
until ((ok >= 3)); do
  served="$(curl -sf "$REMOTE_OPENAI_BASE_URL/models" \
    -H "Authorization: Bearer ${REMOTE_OPENAI_API_KEY:-EMPTY}" |
    python -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])' ||
    true)"
  reply="$(curl -sf -m 180 "$REMOTE_OPENAI_BASE_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${REMOTE_OPENAI_API_KEY:-EMPTY}" \
    -d "${probe/MODEL/$served}" || true)"
  if [[ "$reply" == *'"tool_calls"'* ]]; then ok=$((ok + 1)); else ok=0; fi
  ((ok >= 3)) || {
    echo "waiting for $REMOTE_OPENAI_BASE_URL"
    sleep 20
  }
done
echo "ENDPOINT READY $(date -Is)"

echo "GENERATE $MODEL $(date -Is)"
bfcl generate --model "$MODEL" --test-category "$CATS" \
  --num-threads "${THREADS:-32}"
echo "EVALUATE $MODEL $(date -Is)"
bfcl evaluate --model "$MODEL" --test-category "$CATS"
echo "DONE $MODEL $(date -Is)"
