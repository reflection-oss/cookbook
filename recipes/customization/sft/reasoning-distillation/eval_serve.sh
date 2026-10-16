#!/usr/bin/env bash
# Serve an exported checkpoint, or the stock model, with SGLang for the
# evaluation. Gemma 4's native tool-call and reasoning parsers are enabled,
# so the OpenAI-compatible API returns structured `tool_calls` and the
# model's thoughts in `reasoning_content`.
#
#   bash eval_serve.sh <model_dir_or_hub_id> [served_model_name] [port]
set -euo pipefail

MODEL="${1:?model directory or hub id}"
python -m sglang.launch_server --model-path "$MODEL" \
  --served-model-name "${2:-$MODEL}" --port "${3:-30000}" \
  --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --mem-fraction-static 0.85 --trust-remote-code
