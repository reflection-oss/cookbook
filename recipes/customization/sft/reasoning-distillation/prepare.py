"""Render the training file through Gemma 4's chat template and mask labels.

Reads the data-synthesis recipe's JSONL and writes a Hugging Face dataset
with `input_ids`, `attention_mask` and `labels`, plus a `<output>_val`
split. Labels are -100 except on what the model generates. One line of the
chat template changes for training: the stock gate renders a thought only
on the last round and on tool-call turns; the training gate renders it on
every assistant turn, so the model also learns to think before a text reply.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics

import datasets
import transformers

MODEL_TURN = '<|turn>model\n'
TURN_END = '<turn|>'
TOOL_RESP_START = '<|tool_response>'
TOOL_RESP_END = '<tool_response|>'
GATE_STOCK = (
  '{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or '
  "(preserve_thinking and message.get('tool_calls')) -%}"
)
GATE_ALL = (
  '{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or '
  'preserve_thinking -%}'
)


def training_template(template: str) -> str:
  """The chat template with the thinking gate open on every turn."""
  if GATE_STOCK not in template:
    raise RuntimeError(
      'stock thinking gate not found in the chat template; '
      'use the model revision pinned in download.sh'
    )
  return template.replace(GATE_STOCK, GATE_ALL)


def template_messages(messages: list[dict]) -> list[dict]:
  """Copy the messages with tool-call arguments as objects, not JSON text."""
  out = json.loads(json.dumps(messages))
  for message in out:
    for call in message.get('tool_calls') or []:
      arguments = call['function'].get('arguments')
      if isinstance(arguments, str):
        call['function']['arguments'] = (
          json.loads(arguments) if arguments else {}
        )
  return out


def trainable_spans(text: str) -> list[tuple[int, int]]:
  """The character spans the model generates in a rendered example.

  Each model turn counts, minus the tool responses inside it. The model
  emits the first `<|tool_response>` tag after its calls as its stop signal,
  so that tag is trained; the response body and the tags that follow are
  environment output.
  """
  spans: list[tuple[int, int]] = []
  pos = 0
  while (start := text.find(MODEL_TURN, pos)) != -1:
    start += len(MODEL_TURN)
    end = text.find(TURN_END, start)
    end = len(text) if end == -1 else end + len(TURN_END)
    cur = start
    while True:
      resp = text.find(TOOL_RESP_START, cur, end)
      if resp == -1:
        spans.append((cur, end))
        break
      after_call = text[max(0, resp - 12) : resp].endswith('<tool_call|>')
      keep = resp + len(TOOL_RESP_START) if after_call else resp
      if keep > cur:
        spans.append((cur, keep))
      resp_end = text.find(TOOL_RESP_END, resp, end)
      cur = end if resp_end == -1 else resp_end + len(TOOL_RESP_END)
      if cur >= end:
        break
    pos = end
  return spans


def encode(
  tokenizer: transformers.PreTrainedTokenizerBase,
  text: str,
  spans: list[tuple[int, int]],
  max_len: int,
) -> dict[str, list[int]] | None:
  """Tokenize one rendered example; None when it is longer than `max_len`."""
  enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
  ids, offsets = enc['input_ids'], enc['offset_mapping']
  if len(ids) > max_len:
    return None
  labels = []
  span_idx = 0
  for token_id, (char_start, _) in zip(ids, offsets, strict=True):
    while span_idx < len(spans) and spans[span_idx][1] <= char_start:
      span_idx += 1
    inside = (
      span_idx < len(spans)
      and spans[span_idx][0] <= char_start < spans[span_idx][1]
    )
    labels.append(token_id if inside else -100)
  return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def main() -> None:
  """Render, tokenize and split the training file."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', required=True, help='training JSONL')
  parser.add_argument('--model', required=True, help='base model directory')
  parser.add_argument(
    '--output', required=True, help='dataset directory; <output>_val too'
  )
  parser.add_argument('--max-len', type=int, default=12288)
  parser.add_argument('--val-size', type=int, default=200)
  parser.add_argument('--seed', type=int, default=2)
  args = parser.parse_args()
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
  template = training_template(tokenizer.chat_template)
  with pathlib.Path(args.input).open(encoding='utf-8') as f:
    rows = [json.loads(line) for line in f if line.strip()]
  examples: list[dict] = []
  lengths: list[int] = []
  ratios: list[float] = []
  dropped_len = dropped_err = 0
  for row in rows:
    try:
      text = tokenizer.apply_chat_template(
        template_messages(row['messages']),
        tools=row['tools'] or None,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        preserve_thinking=True,
        chat_template=template,
      )
      spans = trainable_spans(text)
    except Exception as exc:  # noqa: BLE001 - one bad row must not stop the run
      dropped_err += 1
      if dropped_err <= 3:
        print(f'render error on {row.get("id")}: {type(exc).__name__}: {exc}')
      continue
    item = encode(tokenizer, text, spans, args.max_len)
    if item is None:
      dropped_len += 1
      continue
    item['id'] = row.get('id')
    examples.append(item)
    lengths.append(len(item['input_ids']))
    trainable = sum(1 for label in item['labels'] if label != -100)
    ratios.append(trainable / len(item['labels']))
  random.Random(args.seed).shuffle(examples)
  n_val = min(args.val_size, len(examples) // 10)
  val, train = examples[:n_val], examples[n_val:]
  datasets.Dataset.from_list(train).save_to_disk(args.output)
  if n_val:
    datasets.Dataset.from_list(val).save_to_disk(args.output + '_val')
  stats = {
    'input_rows': len(rows),
    'train': len(train),
    'val': n_val,
    'dropped_too_long': dropped_len,
    'dropped_render_error': dropped_err,
    'tokens_mean': round(statistics.mean(lengths)) if lengths else 0,
    'tokens_p95': sorted(lengths)[int(0.95 * len(lengths)) - 1]
    if lengths
    else 0,
    'tokens_max': max(lengths) if lengths else 0,
    'trainable_ratio_mean': round(statistics.mean(ratios), 3) if ratios else 0,
    'total_tokens': sum(lengths),
    'max_len': args.max_len,
  }
  print(json.dumps(stats, indent=2))
  pathlib.Path(args.output, 'prep_stats.json').write_text(
    json.dumps(stats, indent=2), encoding='utf-8'
  )


if __name__ == '__main__':
  main()
