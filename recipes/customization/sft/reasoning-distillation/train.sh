#!/usr/bin/env bash
# Prepare the data, fine-tune and export, with the recipe's defaults.
#
#   BASE_MODEL=/data/gemma-4-12B-it DATA=/data/train.jsonl bash train.sh
#
# BASE_MODEL (see download.sh) and DATA are required. RUN_DIR holds
# dataset/, checkpoints/ and export/ (default /data/gemma4-tool-reasoning);
# MAX_LEN (12288), PREP_SEED (2) and NUM_GPUS (8) are optional. Arguments
# are passed on to train.py, for example `bash train.sh --seed 44`.
set -euo pipefail

: "${BASE_MODEL:?set BASE_MODEL to the base checkpoint directory}"
: "${DATA:?set DATA to the training JSONL}"
RUN_DIR="${RUN_DIR:-/data/gemma4-tool-reasoning}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ -e "$RUN_DIR/checkpoints" ]]; then
  echo "$RUN_DIR/checkpoints exists; use a fresh RUN_DIR so the export cannot pick up an earlier run's checkpoint" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

# FSDP wraps one decoder-layer class; the dense 12B and the E-series differ.
MODEL_TYPE="$(python -c 'import json, sys
with open(sys.argv[1] + "/config.json", encoding="utf-8") as f:
  print(json.load(f)["model_type"])' "$BASE_MODEL")"
case "$MODEL_TYPE" in
  *unified*) WRAP=Gemma4UnifiedTextDecoderLayer ;;
  *) WRAP=Gemma4TextDecoderLayer ;;
esac
sed -e "s/__DECODER_LAYER_CLASS__/$WRAP/" \
  -e "s/^num_processes: .*/num_processes: ${NUM_GPUS:-8}/" \
  "$HERE/fsdp.yaml" >"$RUN_DIR/fsdp.yaml"
echo "START base=$BASE_MODEL model_type=$MODEL_TYPE wrap=$WRAP $(date -Is)"

python "$HERE/prepare.py" --input "$DATA" --model "$BASE_MODEL" \
  --output "$RUN_DIR/dataset" --max-len "${MAX_LEN:-12288}" \
  --seed "${PREP_SEED:-2}"
accelerate launch --config_file "$RUN_DIR/fsdp.yaml" "$HERE/train.py" \
  --model "$BASE_MODEL" --dataset "$RUN_DIR/dataset" \
  --eval-dataset "$RUN_DIR/dataset_val" --output "$RUN_DIR/checkpoints" "$@"
echo "TRAIN_DONE $(date -Is)"

# The epoch checkpoint is the one with the highest step number.
LAST=0
for ckpt in "$RUN_DIR"/checkpoints/checkpoint-*; do
  step="${ckpt##*checkpoint-}"
  if ((step > LAST)); then LAST="$step"; fi
done
python "$HERE/export.py" --src "$RUN_DIR/checkpoints/checkpoint-$LAST" \
  --base "$BASE_MODEL" --out "$RUN_DIR/export" --gpu 0
echo "EXPORT_OK $RUN_DIR/export $(date -Is)"
