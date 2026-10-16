#!/usr/bin/env bash
# Build the training set end to end with the recipe's settings.
#
#   bash run.sh              # the full set: about 59M teacher tokens
#   LIMIT=20 bash run.sh     # a first run on 20 conversations per source
#
# WORK is the output folder (default work/). WORKERS is the number of
# concurrent teacher requests (default 16). The teacher stage is resumable:
# run the script again after an interruption and it continues.
set -euo pipefail
cd "$(dirname "$0")"

WORK="${WORK:-work}"
LIMIT="${LIMIT:-}"
WORKERS="${WORKERS:-16}"
PYTHON="${PYTHON:-python}"
mkdir -p "$WORK"

limit=()
if [[ -n "$LIMIT" ]]; then
  limit=(--limit "$LIMIT")
fi

echo "== select"
"$PYTHON" main.py select-toolace --out "$WORK/toolace_rows.jsonl" \
  ${limit[@]+"${limit[@]}"}
"$PYTHON" main.py select-nemotron --out "$WORK/nemotron_rows.jsonl" \
  --pool "$WORK/nemotron_pool.jsonl" ${limit[@]+"${limit[@]}"}

echo "== enrich"
"$PYTHON" main.py enrich --rows "$WORK/toolace_rows.jsonl" \
  --out "$WORK/teacher_toolace.jsonl" --workers "$WORKERS"
"$PYTHON" main.py enrich --rows "$WORK/nemotron_rows.jsonl" \
  --out "$WORK/teacher_nemotron.jsonl" --workers "$WORKERS"

echo "== audit"
"$PYTHON" main.py audit --rows "$WORK/toolace_rows.jsonl" \
  --records "$WORK/teacher_toolace.jsonl" --out-prefix "$WORK/audit_toolace"
"$PYTHON" main.py audit --rows "$WORK/nemotron_rows.jsonl" \
  --records "$WORK/teacher_nemotron.jsonl" --out-prefix "$WORK/audit_nemotron"

echo "== assemble"
"$PYTHON" main.py assemble --rows "$WORK/toolace_rows.jsonl" \
  --records "$WORK/teacher_toolace.jsonl" \
  --exclude "$WORK/audit_toolace_bad_turns.json" \
  --out "$WORK/toolace_messages.jsonl" --no-ask
"$PYTHON" main.py assemble --rows "$WORK/nemotron_rows.jsonl" \
  --records "$WORK/teacher_nemotron.jsonl" \
  --exclude "$WORK/audit_nemotron_bad_turns.json" \
  --out "$WORK/nemotron_messages.jsonl" --system-text toolace

echo "== mix"
"$PYTHON" main.py mix --toolace "$WORK/toolace_messages.jsonl" \
  --nemotron "$WORK/nemotron_messages.jsonl" --out "$WORK/train.jsonl"
echo "done: $WORK/train.jsonl"
