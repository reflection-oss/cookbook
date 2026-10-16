#!/usr/bin/env bash
# Prepare, train and export. TRACES lists the sources as NAME=DIR pairs.
#   TRACES="corpus=traces/corpus rollouts=traces/rollouts" NUM_GPUS=8 bash train.sh
# Extra arguments go to train_sft.py, for example `bash train.sh --epochs 3`.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_DIR="${MODEL_DIR:-models/gemma-4-E4B-it}"
TRACES="${TRACES:?NAME=DIR pairs of trace folders}"
RUN_DIR="${RUN_DIR:-runs/e4b-distil}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_LEN="${MAX_LEN:-65536}"   # longer conversations are left out
GRAD_ACCUM=$((32 / NUM_GPUS)) # effective batch of 32 conversations
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR"

# FSDP wraps one decoder layer at a time; the class name depends on the
# model family: E2B/E4B are `gemma4`, 12B/31B are `gemma4_unified`.
model_type=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model_type'])" "$MODEL_DIR/config.json")
layer_cls=Gemma4TextDecoderLayer
[[ "$model_type" == *unified* ]] && layer_cls=Gemma4UnifiedTextDecoderLayer
sed "s/fsdp_transformer_layer_cls_to_wrap: .*/fsdp_transformer_layer_cls_to_wrap: $layer_cls/" \
  fsdp.yaml >"$RUN_DIR/fsdp.yaml"

trace_args=()
for pair in $TRACES; do trace_args+=(--traces "$pair"); done

echo "== prepare $(date -Is)"
python prepare.py --model "$MODEL_DIR" --out "$RUN_DIR" --max-len "$MAX_LEN" "${trace_args[@]}"

echo "== train on $NUM_GPUS GPUs, wrap $layer_cls $(date -Is)"
accelerate launch --config_file "$RUN_DIR/fsdp.yaml" --num_processes "$NUM_GPUS" \
  train_sft.py --model "$MODEL_DIR" --data "$RUN_DIR" \
  --output "$RUN_DIR/checkpoints" --grad-accum "$GRAD_ACCUM" "$@"

last=$(printf '%s\n' "$RUN_DIR"/checkpoints/checkpoint-* | sort -V | tail -1)
echo "== export $last $(date -Is)"
python export.py --base "$MODEL_DIR" --src "$last" --out "$RUN_DIR/export"
