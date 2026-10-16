"""Turn teacher traces into the tokenized training set.

Input: one folder per source, one folder per environment inside it, with the
passing mini-swe-agent conversations as `trace_<n>.json` files holding
`{"task": "<env_id>", "messages": [...]}` in the OpenAI chat format the agent
records (system, user, assistant with `reasoning_content` and `tool_calls`,
tool).

Steps: keep at most `--per-env` traces per environment; drop the agent's
bookkeeping messages; truncate every thought to `--thought-cap` characters at
a sentence boundary; replace the prompts with the ones the student is served
with; shuffle all sources together and write `train.jsonl`; render every
conversation with Gemma 4's chat template, thinking on; label only what the
model generates (thought, tool call, the `<|tool_response>` stop tag, the
turn end) and mask the prompts and tool outputs; hold out a tenth for the
eval loss; save `dataset` and `dataset_val`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import shutil
import statistics

from datasets import Dataset
from transformers import AutoTokenizer

STANDARD_SYSTEM_PROMPT = (
  'You are a helpful assistant that can interact with a computer.'
)
TASK_MARKER = 'Please solve this issue:'
BASH_TOOL = {
  'type': 'function',
  'function': {
    'name': 'bash',
    'description': 'Execute a bash command',
    'parameters': {
      'type': 'object',
      'properties': {
        'command': {
          'type': 'string',
          'description': 'The bash command to execute',
        }
      },
      'required': ['command'],
    },
  },
}
SUBMIT_NOTE = 'action was not executed'
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
# Gemma 4 control tags. A trace whose text contains one (a command that
# printed template markup) would corrupt the rendering and the loss mask.
CONTROL_TAG = re.compile(r'<\|[a-z_"]+>|<[a-z_"]+\|>')

MODEL_TURN = '<|turn>model\n'
TURN_END = '<turn|>'
TOOL_CALL_END = '<tool_call|>'
TOOL_RESPONSE_START = '<|tool_response>'
TOOL_RESPONSE_END = '<tool_response|>'
# The stock template renders the thought only on tool-call turns and the last
# round; the training template renders it on every assistant turn.
GATE_STOCK = (
  '{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or '
  "(preserve_thinking and message.get('tool_calls')) -%}"
)
GATE_ALL = (
  '{%- set thinking_gate = (loop.index0 > ns_turn.last_user_idx) or '
  'preserve_thinking -%}'
)


def cap_thought(text: str, cap: int) -> str:
  """Cut a thought at the last sentence end before `cap`, keep its last one.

  The result can run up to about 400 characters past `cap`: the thought's
  final sentence is kept so the decision it reaches survives the cut.
  """
  if len(text) <= cap:
    return text
  head = text[:cap]
  cut = head.rfind('. ')
  if cut > cap // 2:
    head = head[: cut + 1]
  last = SENTENCE_END.split(text.strip())[-1]
  return f'{head.rstrip()} … {last[:400]}'.strip()


def convert(trace: list[dict], thought_cap: int) -> list[dict]:
  """Keep the chat messages, cap the thoughts, drop the submit observation."""
  out: list[dict] = []
  for index, message in enumerate(trace):
    role = message['role']
    if role in ('system', 'user'):
      out.append({'role': role, 'content': message['content']})
    elif role == 'assistant':
      entry = {
        'role': 'assistant',
        'content': message.get('content') or '',
        'reasoning_content': cap_thought(
          message.get('reasoning_content') or '', thought_cap
        ),
      }
      if message.get('tool_calls'):
        entry['tool_calls'] = [
          {
            'id': call['id'],
            'type': 'function',
            'function': {
              'name': call['function']['name'],
              'arguments': call['function']['arguments'],
            },
          }
          for call in message['tool_calls']
        ]
      out.append(entry)
    elif role == 'tool':
      content = message.get('content') or ''
      if SUBMIT_NOTE in content and index >= len(trace) - 2:
        continue  # mini-swe-agent's note after the submit, never executed
      out.append({
        'role': 'tool',
        'tool_call_id': message.get('tool_call_id'),
        'name': 'bash',
        'content': content,
      })
  return out


def normalize_prompts(messages: list[dict]) -> None:
  """Use the evaluation prompts: standard system message, task text only.

  The teacher may have been run with extra instructions in front of the
  task; the student is served with mini-swe-agent's plain prompts, so it has
  to learn the teacher's habits under those. Assistant turns are not touched.
  """
  messages[0]['content'] = STANDARD_SYSTEM_PROMPT
  content = messages[1]['content']
  index = content.find(TASK_MARKER)
  if messages[0]['role'] != 'system' or messages[1]['role'] != 'user':
    raise ValueError('expected a system message and a user message first')
  if index == -1:
    raise ValueError(f'task marker {TASK_MARKER!r} not found')
  messages[1]['content'] = content[index:]


def chars(messages: list[dict]) -> int:
  """Count the characters the template will render."""
  return sum(
    len(m.get('content') or '')
    + len(m.get('reasoning_content') or '')
    + sum(len(c['function']['arguments']) for c in m.get('tool_calls') or [])
    for m in messages
  )


def has_control_tag(messages: list[dict]) -> bool:
  """Return whether any message text contains a Gemma 4 control tag."""
  return any(
    CONTROL_TAG.search(m.get(field) or '')
    for m in messages
    for field in ('content', 'reasoning_content')
  )


def load_rows(
  sources: dict[str, pathlib.Path],
  per_env: int,
  thought_cap: int,
  max_chars: int,
  seed: int,
) -> list[dict]:
  """Read, convert, filter and shuffle the traces of every source."""
  rows = []
  for name, root in sources.items():
    for env_dir in sorted(p for p in root.iterdir() if p.is_dir()):
      for path in sorted(env_dir.glob('trace_*.json'))[:per_env]:
        with path.open(encoding='utf-8') as f:
          messages = convert(json.load(f)['messages'], thought_cap)
        if chars(messages) > max_chars or has_control_tag(messages):
          continue
        normalize_prompts(messages)
        rows.append({
          'id': f'{name}/{env_dir.name}/{path.stem}',
          'env_id': env_dir.name,
          'source': name,
          'tools': [BASH_TOOL],
          'messages': messages,
        })
  random.Random(seed).shuffle(rows)
  return rows


def trainable_spans(text: str) -> list[tuple[int, int]]:
  """Character spans the model generates: model turns minus tool results.

  Inside a model turn the model writes its thought, text and tool calls and
  then emits the first `<|tool_response>` tag, the server's stop signal. That
  tag is trained; the result that follows and later tags are masked.
  """
  spans: list[tuple[int, int]] = []
  position = 0
  while (start := text.find(MODEL_TURN, position)) != -1:
    start += len(MODEL_TURN)
    end = text.find(TURN_END, start)
    end = len(text) if end == -1 else end + len(TURN_END)
    cursor = start
    while (response := text.find(TOOL_RESPONSE_START, cursor, end)) != -1:
      before = text[max(0, response - len(TOOL_CALL_END)) : response]
      keep = response + len(TOOL_RESPONSE_START) * (before == TOOL_CALL_END)
      if keep > cursor:
        spans.append((cursor, keep))
      response_end = text.find(TOOL_RESPONSE_END, response, end)
      cursor = (
        end if response_end == -1 else response_end + len(TOOL_RESPONSE_END)
      )
    if cursor < end:
      spans.append((cursor, end))
    position = end
  return spans


def render(
  tokenizer: AutoTokenizer, rows: list[dict], max_len: int
) -> list[dict]:
  """Render and tokenize the rows, labelling only the trainable spans."""
  if GATE_STOCK not in tokenizer.chat_template:
    raise RuntimeError('thinking gate not found in the chat template')
  template = tokenizer.chat_template.replace(GATE_STOCK, GATE_ALL)
  examples = []
  for row in rows:
    messages = json.loads(json.dumps(row['messages']))
    for message in messages:  # the template wants arguments as objects
      for call in message.get('tool_calls') or []:
        call['function']['arguments'] = json.loads(
          call['function']['arguments'] or '{}'
        )
    text = tokenizer.apply_chat_template(
      messages,
      tools=row['tools'],
      tokenize=False,
      add_generation_prompt=False,
      enable_thinking=True,
      preserve_thinking=True,
      chat_template=template,
    )
    spans = trainable_spans(text)
    encoded = tokenizer(
      text, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = encoded['input_ids']
    if len(ids) > max_len:
      continue
    labels, span = [], 0
    for token_id, (offset, _) in zip(
      ids, encoded['offset_mapping'], strict=True
    ):
      while span < len(spans) and spans[span][1] <= offset:
        span += 1
      inside = span < len(spans) and spans[span][0] <= offset
      labels.append(token_id if inside else -100)
    examples.append({
      'input_ids': ids,
      'attention_mask': [1] * len(ids),
      'labels': labels,
    })
  return examples


def main() -> None:
  """Write train.jsonl, dataset and dataset_val under --out."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--traces',
    action='append',
    required=True,
    metavar='NAME=DIR',
    help='a source label and its trace folder; repeat per source',
  )
  parser.add_argument('--model', required=True, help='base model folder')
  parser.add_argument('--out', type=pathlib.Path, required=True)
  parser.add_argument('--per-env', type=int, default=2)
  parser.add_argument('--thought-cap', type=int, default=2000)
  parser.add_argument('--max-chars', type=int, default=220_000)
  parser.add_argument('--max-len', type=int, default=65536)
  parser.add_argument('--val-size', type=int, default=200)
  parser.add_argument('--seed', type=int, default=9)
  args = parser.parse_args()

  sources = {}
  for spec in args.traces:
    name, _, folder = spec.partition('=')
    if not folder:
      parser.error(f'--traces expects NAME=DIR, got {spec!r}')
    sources[name] = pathlib.Path(folder)
  rows = load_rows(
    sources, args.per_env, args.thought_cap, args.max_chars, args.seed
  )
  args.out.mkdir(parents=True, exist_ok=True)
  with (args.out / 'train.jsonl').open('w', encoding='utf-8') as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False) + '\n')

  tokenizer = AutoTokenizer.from_pretrained(args.model)
  examples = render(tokenizer, rows, args.max_len)
  n_val = min(args.val_size, len(examples) // 10)
  for name in ('dataset', 'dataset_val'):  # never leave an earlier run's split
    shutil.rmtree(args.out / name, ignore_errors=True)
  Dataset.from_list(examples[n_val:]).save_to_disk(str(args.out / 'dataset'))
  if n_val:
    Dataset.from_list(examples[:n_val]).save_to_disk(
      str(args.out / 'dataset_val')
    )
  lengths = [len(e['input_ids']) for e in examples]
  print(
    json.dumps({
      'rows': len(rows),
      'environments': len({r['env_id'] for r in rows}),
      'dropped_too_long': len(rows) - len(examples),
      'train': len(examples) - n_val,
      'val': n_val,
      'tokens_mean': round(statistics.mean(lengths)),
      'tokens_max': max(lengths),
      'total_tokens': sum(lengths),
    })
  )


if __name__ == '__main__':
  main()
