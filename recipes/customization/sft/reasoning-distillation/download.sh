#!/usr/bin/env bash
# Download a Gemma 4 instruction-tuned checkpoint at its pinned revision and
# check that it is complete. MODEL picks the size: 12B (default), E4B or
# E2B. The weights are public; no Hugging Face token is needed.
#
#   MODEL=12B bash download.sh
set -euo pipefail

MODEL="${MODEL:-12B}"
case "$MODEL" in
  12B) REVISION=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 ;;
  E4B) REVISION=ee0ef6023621cff504d758262d4e04895a5af4a2 ;;
  E2B) REVISION=3e22461f65e89153144f8adb70e3b8c2cc9845a7 ;;
  *)
    echo "MODEL must be 12B, E4B or E2B, got '$MODEL'" >&2
    exit 1
    ;;
esac
BASE_MODEL="${BASE_MODEL:-/data/gemma-4-$MODEL-it}"

hf download "google/gemma-4-$MODEL-it" --revision "$REVISION" \
  --local-dir "$BASE_MODEL"

python - "$BASE_MODEL" <<'PY'
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
index = base / 'model.safetensors.index.json'
if index.exists():
  with index.open(encoding='utf-8') as f:
    shards = sorted(set(json.load(f)['weight_map'].values()))
else:
  shards = ['model.safetensors']
names = ['config.json', 'tokenizer_config.json', 'chat_template.jinja', *shards]
missing = [n for n in names if not (base / n).exists()]
if missing:
  sys.exit(f'missing files in {base}: {missing}')
print(f'{base}: {len(shards)} weight file(s) complete')
PY
echo "Gemma 4 $MODEL weights complete in $BASE_MODEL (revision $REVISION)"
