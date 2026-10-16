"""Whole-conversation rewrite: the teacher adds a thought to every turn.

One request per conversation. The teacher returns the conversation unchanged
apart from a ``<thought>...</thought>`` block at the start of every assistant
message. The reply is verified turn by turn (calls must survive verbatim),
cleaned (restated calls and references to the rewriting task removed) and
flagged. Records are appended to a JSONL file, one per assistant turn, and
the run is resumable.
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import os
import pathlib
import random
import re
import threading
import time

import config
import formats
import openai
import tqdm

Record = dict[str, object]

REWRITE_PROMPT = (
  'You are a highly intelligent AI with an IQ of 170, and your job is to '
  'enhance existing conversation examples. Remember to return the entire '
  'conversation as is, BUT\n'
  '\n'
  'BUT, we will add Chain of Thought and planning to EVERY "Assistant" '
  'message: before a tool call, before a clarifying question or a refusal '
  '(when no function fits or a required parameter is missing), and before '
  'a final answer that follows tool results.\n'
  '\n'
  'Each assistant message must start with the thinking inside '
  '<thought></thought> tags, followed by the original message content '
  'unchanged: tool calls stay inside their <tool></tool> tags exactly as '
  'they are, and text replies stay verbatim.\n'
  '\n'
  "The thought is the assistant's own reasoning, in first person, before it "
  'acts:\n'
  '- before a tool call: what the user needs, which function fits and why '
  '(and why alternatives do not), and why each parameter gets its value;\n'
  '- after tool results: what the results say, how they answer (or fail to '
  'answer) the request, and what to do next;\n'
  '- before a clarifying question or a refusal: why no available function '
  'applies, or which required information is missing.\n'
  'Never mention that you are enhancing an example.\n'
  '\n'
  'Please keep in mind that we are not modifying anything in the example, '
  'nor are we changing what it does. We are only adding thoughts to every '
  'assistant message.\n'
  '\n'
  '<<thought_style>>\n'
  '\n'
  'For example, if you see:\n'
  '\n'
  '"from": "assistant",\n'
  '"value": "<tool>[Some API(param=\\"value\\")]</tool>"\n'
  '\n'
  'Change it to:\n'
  '\n'
  '"from": "assistant",\n'
  '"value": "<thought>Let me think about this request. I need to gather X '
  'information using Tool Y. To do this, I need to set the parameter to '
  "'value' because of reason Z.</thought><tool>[Some API(param="
  '\\"value\\")]</tool>"\n'
  '\n'
  'And if you see:\n'
  '\n'
  '"from": "assistant",\n'
  '"value": "Here is the information you asked for: ..."\n'
  '\n'
  'Change it to:\n'
  '\n'
  '"from": "assistant",\n'
  '"value": "<thought>The tool returned ..., which answers the user\'s '
  'question about ... Nothing else is needed, so I will summarize it for '
  'the user.</thought>Here is the information you asked for: ..."\n'
  '\n'
  'Return the enhanced conversation as a JSON array of {"from": ..., '
  '"value": ...} objects with exactly the same messages in the same order, '
  'inside a ```json code block, and nothing else.\n'
  '\n'
  'BEGIN WORK NOW. Enhance every assistant message with detailed Chain of '
  'Thought reasoning:\n'
  '<<conversations>>\n'
)

_TOOL_BLOCK = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)
_THOUGHT_BLOCK = re.compile(
  r'^\s*<thought>(.*?)</thought>\s*', re.DOTALL | re.IGNORECASE
)
_JSON_FENCE = re.compile(r'```(?:json)?\s*(\[.*\])\s*```', re.DOTALL)
_BANNED = re.compile(
  r'\b(gold|the next message|next message|the example|given call|'
  r'provided call|provided answer|the answer provided|as shown|shown to me|'
  r'the expected call)\b',
  re.IGNORECASE,
)
_TOOL_TOKENS = re.compile(
  r'<\|tool_call>|<tool_call\|>|<\|channel>|<channel\|>|<tool>|</tool>|```'
)
_WRAPPER_TAGS = re.compile(r'</?reasoning>', re.IGNORECASE)
_TRAILING_CALL = re.compile(
  r'\s*\[[^\[\]]*\([^\[\]]*\)\s*(?:,\s*[^\[\]]*\([^\[\]]*\)\s*)*\]\s*$',
  re.DOTALL,
)
_SENTENCE = re.compile(r'(?<=[.!?])\s+')


def build_prompt(row: formats.Row, thought_style: str) -> str:
  """The teacher prompt for one conversation."""
  wrapped = json.dumps(wrap_conversation(row), ensure_ascii=False, indent=2)
  style = config.THOUGHT_STYLES[thought_style]
  return REWRITE_PROMPT.replace('<<thought_style>>', style).replace(
    '<<conversations>>', wrapped
  )


def wrap_conversation(row: formats.Row) -> list[dict[str, str]]:
  """System prompt first; tool calls wrapped in <tool></tool> tags."""
  wrapped = [{'from': 'system', 'value': row['system']}]
  for m in row['conversations']:
    v = m['value']
    if m['from'] == 'assistant' and formats.is_tool_call(v):
      v = f'<tool>{v.strip()}</tool>'
    wrapped.append({'from': m['from'], 'value': v})
  return wrapped


def parse_rewrite(content: str) -> list[dict[str, str]] | None:
  """The JSON array in the teacher's reply, or None when there is none."""
  m = _JSON_FENCE.search(content)
  candidates = [m.group(1)] if m else []
  a, b = content.find('['), content.rfind(']')
  if a != -1 and b > a:
    candidates.append(content[a : b + 1])
  for c in candidates:
    try:
      out = json.loads(c)
    except ValueError:
      continue
    if isinstance(out, list) and all(
      isinstance(x, dict) and 'from' in x and 'value' in x for x in out
    ):
      return out
  return None


def _norm(s: str) -> str:
  return re.sub(r'\s+', ' ', s).strip()


_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')


def _norm_call(s: str) -> str:
  """Collapse whitespace outside quoted strings; keep it exact inside."""
  parts: list[str] = []
  last = 0
  for m in _QUOTED.finditer(s):
    parts.append(_norm(s[last : m.start()]))
    parts.append(m.group(0))
    last = m.end()
  parts.append(_norm(s[last:]))
  return ''.join(parts).strip()


def verify_rewrite(
  wrapped: list[dict[str, str]], rewritten: list[dict[str, str]]
) -> tuple[dict[int, str], list[str]]:
  """Thoughts by wrapped index, and the failures that reject the reply."""
  if not isinstance(rewritten, list) or len(rewritten) != len(wrapped):
    return {}, ['len_mismatch']
  thoughts: dict[int, str] = {}
  failures: list[str] = []
  for idx, (o, n) in enumerate(zip(wrapped, rewritten, strict=True)):
    if o['from'] != n.get('from'):
      failures.append(f'role_mismatch@{idx}')
    elif o['from'] == 'assistant':
      nv = str(n.get('value', ''))
      m = _THOUGHT_BLOCK.match(nv)
      if not m or len(m.group(1).strip()) < config.MIN_THOUGHT_CHARS:
        failures.append(f'missing_thought@{idx}')
        continue
      blocks = _TOOL_BLOCK.findall(o['value'])
      if blocks:
        nblocks = _TOOL_BLOCK.findall(nv[m.end() :])
        if len(nblocks) != 1 or _norm_call(nblocks[0]) != _norm_call(blocks[0]):
          failures.append(f'call_modified@{idx}')
          continue
      thoughts[idx] = m.group(1).strip()
  return thoughts, failures


def clean_and_flag(
  reasoning: str, kind: str, gold: str
) -> tuple[str, list[str]]:
  """Strip wrapper tags, restated calls and task references; return flags."""
  flags: list[str] = []
  text = _WRAPPER_TAGS.sub('', reasoning)
  if gold.strip() and gold.strip() in text:
    text = text.replace(gold.strip(), ' ')
    flags.append('restated_call')
  if _TRAILING_CALL.search(text):
    text = _TRAILING_CALL.sub('', text)
    flags.append('restated_call')
  if _BANNED.search(text):
    text = ' '.join(s for s in _SENTENCE.split(text) if not _BANNED.search(s))
    flags.append('meta_sentence_removed')
  text = re.sub(r'[ \t]+\n', '\n', text).strip()
  flags = sorted(set(flags))
  if not text:
    return text, ['empty', *flags]
  if len(text) < config.MIN_THOUGHT_CHARS:
    flags.append('too_short')
  if len(text) > config.MAX_THOUGHT_CHARS:
    flags.append('too_long')
  if _BANNED.search(text):
    flags.append('meta_reference')
  if _TOOL_TOKENS.search(text):
    flags.append('tool_tokens')
  if kind == 'call':
    calls, ok = formats.parse_calls(gold)
    names = [c['name'] for c in calls] if ok else []
    if names and not any(n.lower() in text.lower() for n in names):
      flags.append('function_not_mentioned')
  return text, flags


class TeacherClient:
  """One streamed chat request at a time against the Reflection API."""

  def __init__(self, cfg: config.TeacherConfig) -> None:
    api_key = os.environ.get('REFLECTION_API_KEY', '')
    if not api_key:
      raise SystemExit(
        'REFLECTION_API_KEY is not set. Copy .env.example to .env and fill '
        'it in, or export the variable.'
      )
    self.cfg = cfg
    self.model = os.environ.get('REFLECTION_MODEL') or config.MODEL
    self.client = openai.OpenAI(
      base_url=os.environ.get('REFLECTION_API_BASE') or config.API_BASE,
      api_key=api_key,
      timeout=cfg.timeout_s,
      max_retries=0,
    )

  def complete(self, prompt: str, max_tokens: int) -> Record:
    """Stream one reply; return its text, stop reason and token usage."""
    stream = self.client.chat.completions.create(
      model=self.model,
      messages=[{'role': 'user', 'content': prompt}],
      temperature=self.cfg.temperature,
      top_p=self.cfg.top_p,
      max_tokens=max_tokens,
      reasoning_effort=self.cfg.reasoning_effort,
      stream=True,
      stream_options={'include_usage': True},
    )
    content: list[str] = []
    finish_reason = None
    usage = None
    for chunk in stream:
      usage = chunk.usage or usage
      for choice in chunk.choices:
        if choice.delta is not None and choice.delta.content:
          content.append(choice.delta.content)
        finish_reason = choice.finish_reason or finish_reason
    return {
      'content': ''.join(content),
      'finish_reason': finish_reason,
      'usage': {
        'prompt_tokens': usage.prompt_tokens if usage else 0,
        'completion_tokens': usage.completion_tokens if usage else 0,
      },
    }


def rewrite_conversation(
  client: TeacherClient,
  row_idx: int,
  row: formats.Row,
  cfg: config.TeacherConfig,
) -> list[Record]:
  """Rewrite one conversation: one record per assistant turn, or one failure."""
  wrapped = wrap_conversation(row)
  prompt = build_prompt(row, cfg.thought_style)
  max_tokens, error = cfg.max_tokens, 'no attempt'
  usage = {'prompt_tokens': 0, 'completion_tokens': 0}
  for attempt in range(1, cfg.retries + 1):
    try:
      out = client.complete(prompt, max_tokens)
    except Exception as e:  # noqa: BLE001 - any client error is retried
      error = f'{type(e).__name__}: {str(e)[:200]}'
      time.sleep(min(60, 2**attempt) + random.random())
      continue
    for k in usage:
      usage[k] += out['usage'][k]
    if out['finish_reason'] == 'length' and max_tokens < cfg.max_tokens_cap:
      max_tokens = min(cfg.max_tokens_cap, max_tokens * 2)
      error = f'length-exhausted, retrying with {max_tokens}'
      continue
    rewritten = parse_rewrite(out['content'])
    if rewritten is None:
      error = 'unparseable_json'
      continue
    thoughts, failures = verify_rewrite(wrapped, rewritten)
    if failures:
      error = 'verify_failed: ' + ','.join(failures)
      continue
    records: list[Record] = []
    for idx, text in thoughts.items():
      j = idx - 1  # wrapped[0] is the system prompt
      gold = row['conversations'][j]['value']
      kind = 'call' if formats.is_tool_call(gold) else 'text'
      cleaned, flags = clean_and_flag(text, kind, gold)
      records.append({
        'row': row_idx,
        'id': row['id'],
        'msg': j,
        'kind': kind,
        'ok': bool(cleaned) and 'tool_tokens' not in flags,
        'reasoning': cleaned,
        'flags': flags,
        'finish_reason': out['finish_reason'],
        'attempts': attempt,
        'usage': usage,
        'teacher': {'model': client.model, 'style': cfg.thought_style},
      })
    return records
  failure = {
    'row': row_idx,
    'id': row['id'],
    'msg': -1,
    'kind': 'conversation',
    'ok': False,
  }
  return [
    {
      **failure,
      'reasoning': '',
      'flags': ['error'],
      'error': error,
      'attempts': cfg.retries,
      'usage': usage,
    }
  ]


def load_done(path: pathlib.Path, rows: list[formats.Row]) -> set[str]:
  """Ids with a record for every assistant turn; the rest are (re)sent."""
  expected: dict[str, int] = {}
  for row in rows:
    expected[row['id']] = sum(
      1 for m in row['conversations'] if m['from'] == 'assistant'
    )
  seen: dict[str, set[int]] = collections.defaultdict(set)
  if path.exists():
    with path.open(encoding='utf-8') as f:
      for line in f:
        r = json.loads(line)
        if r.get('msg', -1) != -1 and r.get('id') in expected:
          seen[r['id']].add(r['msg'])
  return {i for i, msgs in seen.items() if len(msgs) >= expected[i]}


def enrich(
  rows: list[formats.Row], output: str, cfg: config.TeacherConfig
) -> dict[str, int]:
  """Rewrite every row that has no successful records in ``output`` yet."""
  out_path = pathlib.Path(output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  done = load_done(out_path, rows)
  todo = [
    i
    for i, row in enumerate(rows)
    if row['id'] not in done
    and any(m['from'] == 'assistant' for m in row['conversations'])
  ]
  print(
    f'teacher: {len(todo)} conversations to rewrite '
    f'(style={cfg.thought_style}) -> {out_path}',
    flush=True,
  )
  stats = {'ok': 0, 'bad': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
  if not todo:
    return stats
  client = TeacherClient(cfg)
  print(
    f'teacher model: {client.model} | temperature={cfg.temperature} | '
    f'max_tokens={cfg.max_tokens} | workers={cfg.workers}',
    flush=True,
  )
  lock = threading.Lock()
  with (
    out_path.open('a', encoding='utf-8') as fout,
    concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as pool,
  ):
    futures = [
      pool.submit(rewrite_conversation, client, i, rows[i], cfg) for i in todo
    ]
    for fut in tqdm.tqdm(
      concurrent.futures.as_completed(futures),
      total=len(futures),
      mininterval=1.0,
    ):
      records = fut.result()
      with lock:
        # One write per conversation, so an interruption never leaves a
        # conversation half recorded.
        fout.write(
          ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
        )
        fout.flush()
        for rec in records:
          stats['ok' if rec['ok'] else 'bad'] += 1
        for k in ('prompt_tokens', 'completion_tokens'):
          stats[k] += records[0]['usage'][k]
  print(
    f'teacher done: ok_turns={stats["ok"]} bad={stats["bad"]} '
    f'prompt_tokens={stats["prompt_tokens"]:,} '
    f'completion_tokens={stats["completion_tokens"]:,}',
    flush=True,
  )
  return stats
