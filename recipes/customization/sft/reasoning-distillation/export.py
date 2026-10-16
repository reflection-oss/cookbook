"""Export the epoch checkpoint to a bf16 directory for serving and test it.

The base model's config files are copied in verbatim: newer `transformers`
versions rewrite Gemma 4's config on save in a way older serving images
cannot load. The export then generates a tool call with and without
thinking; `SMOKE_OK` is written only when both generations read as text
and the thinking one opens a thought channel, and both call the probe tool.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time

import torch
import transformers

COPIED_FROM_BASE = (
  'config.json',
  'generation_config.json',
  'tokenizer_config.json',
  'chat_template.jinja',
  'processor_config.json',
)
PROBE_TOOLS = [
  {
    'type': 'function',
    'function': {
      'name': 'get_weather',
      'description': 'Get weather',
      'parameters': {
        'type': 'object',
        'properties': {'city': {'type': 'string'}},
        'required': ['city'],
      },
    },
  }
]
PROBE_MESSAGES = [{'role': 'user', 'content': 'What is the weather in Paris?'}]
THOUGHT_OPEN = '<|channel>thought'
TOOL_CALL = '<|tool_call>call:get_weather'


def looks_sane(text: str) -> bool:
  """Whether a generation reads as text rather than as a scrambled tensor."""
  text = text.strip()
  if len(text) < 20:
    return False
  letters = sum(ch.isascii() and (ch.isalpha() or ch == ' ') for ch in text)
  top = max(text.count(ch) for ch in set(text))
  return letters / len(text) > 0.5 and top / len(text) < 0.3


def generate(model_dir: str, gpu: int) -> dict[bool, str]:
  """Generate the probe reply with thinking on and off."""
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
  model = transformers.AutoModelForCausalLM.from_pretrained(
    model_dir, dtype=torch.bfloat16, device_map={'': gpu}
  )
  generations = {}
  for thinking in (True, False):
    inputs = tokenizer.apply_chat_template(
      PROBE_MESSAGES,
      tools=PROBE_TOOLS,
      add_generation_prompt=True,
      enable_thinking=thinking,
      return_tensors='pt',
      return_dict=True,
    ).to(gpu)
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    prompt_len = inputs['input_ids'].shape[1]
    generations[thinking] = tokenizer.decode(out[0][prompt_len:])
    print(f'generation (thinking={thinking}):', repr(generations[thinking]))
  return generations


def main() -> None:
  """Export the checkpoint, then run the generation smoke test."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--src', required=True, help='trainer checkpoint')
  parser.add_argument('--base', required=True, help='base model directory')
  parser.add_argument('--out', required=True, help='export directory')
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--no-test', action='store_true')
  args = parser.parse_args()
  start = time.time()
  model = transformers.AutoModelForCausalLM.from_pretrained(
    args.src, dtype=torch.bfloat16, device_map='cpu'
  )
  n_params = sum(p.numel() for p in model.parameters()) / 1e9
  print(
    f'loaded {type(model).__name__} from {args.src} in '
    f'{time.time() - start:.0f}s; {n_params:.2f}B params'
  )
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  for marker in ('SMOKE_OK', 'SMOKE_FAIL'):
    (out / marker).unlink(missing_ok=True)
  model.save_pretrained(out, safe_serialization=True, max_shard_size='5GB')
  transformers.AutoTokenizer.from_pretrained(args.src).save_pretrained(out)
  for name in COPIED_FROM_BASE:
    if pathlib.Path(args.base, name).exists():
      shutil.copy2(pathlib.Path(args.base, name), out)
  (out / 'training_args.bin').unlink(missing_ok=True)
  del model
  if args.no_test:
    return
  generations = generate(str(out), args.gpu)
  ok = (
    looks_sane(generations[True])
    and looks_sane(generations[False])
    and THOUGHT_OPEN in generations[True]
    and TOOL_CALL in generations[True]
    and TOOL_CALL in generations[False]
  )
  marker = out / ('SMOKE_OK' if ok else 'SMOKE_FAIL')
  marker.write_text(
    'thinking: ' + generations[True] + '\n\nno thinking: ' + generations[False],
    encoding='utf-8',
  )
  print(marker.name)
  if not ok:
    sys.exit(3)


if __name__ == '__main__':
  main()
