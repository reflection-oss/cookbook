"""Export a checkpoint in bf16 and check that it generates.

The base model's config and tokenizer files are copied over the export:
newer Transformers releases rewrite Gemma 4's config in a format that older
serving stacks cannot read, and the weights do not depend on it. Then the
export is loaded on one GPU and asked a tool-calling question with thinking
on. The run fails if the answer is not readable text or does not open the
thought channel; a corrupted save has normal tensor statistics and only
shows up here.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_FILES = (
  'config.json',
  'generation_config.json',
  'tokenizer_config.json',
  'chat_template.jinja',
  'processor_config.json',
)
TOOL = {
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


def main() -> None:
  """Export the checkpoint and run the generation check."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--src', type=pathlib.Path, required=True)
  parser.add_argument('--base', type=pathlib.Path, required=True)
  parser.add_argument('--out', type=pathlib.Path, required=True)
  args = parser.parse_args()

  model = AutoModelForCausalLM.from_pretrained(
    args.src, dtype=torch.bfloat16, device_map='cpu'
  )
  model.save_pretrained(args.out, safe_serialization=True, max_shard_size='5GB')
  AutoTokenizer.from_pretrained(args.src).save_pretrained(args.out)
  for name in BASE_FILES:
    if (args.base / name).exists():
      shutil.copy2(args.base / name, args.out / name)
  (args.out / 'training_args.bin').unlink(missing_ok=True)
  del model

  tokenizer = AutoTokenizer.from_pretrained(args.out)
  model = AutoModelForCausalLM.from_pretrained(
    args.out, dtype=torch.bfloat16, device_map={'': 0}
  )
  inputs = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'What is the weather in Paris?'}],
    tools=[TOOL],
    add_generation_prompt=True,
    enable_thinking=True,
    return_tensors='pt',
    return_dict=True,
  ).to(0)
  output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
  text = tokenizer.decode(
    output[0][inputs['input_ids'].shape[1] :], skip_special_tokens=False
  )
  print(f'generation: {text!r:.400}')
  letters = sum(c.isascii() and (c.isalpha() or c == ' ') for c in text)
  repeated = max(text.count(c) for c in set(text)) if text else 0
  sane = (
    len(text) > 20
    and letters / len(text) > 0.5
    and repeated / len(text) < 0.3
    and '<|channel>thought' in text
  )
  if not sane:
    raise SystemExit('EXPORT_FAILED: the export does not generate sane text')
  print('EXPORT_OK', args.out)


if __name__ == '__main__':
  main()
