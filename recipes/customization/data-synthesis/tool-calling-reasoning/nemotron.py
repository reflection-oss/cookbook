"""Nemotron-SFT-Agentic-v2 (NVIDIA, CC-BY-4.0): the agentic half of the data.

Only the ``interactive_agent`` file is used: customer-service trajectories
with a policy prompt and 10 to 30 tools. Selection rule ("look up, then
act"): from the first rows of the file, keep a conversation when it has no
``filter_reason`` and is not a duplicate, contains at least two tool calls
that are not all transfers or escalations, has no assistant turn that mixes
prose and a call, fits in a character budget, and has at least one call
argument value that appears in an earlier tool result and nowhere in the
user's text, so the model acted on state it discovered. Then a seeded
uniform sample of the qualifying conversations.
"""

from __future__ import annotations

import collections
import collections.abc
import hashlib
import json
import random

import config
import formats
import requests

RawRow = dict[str, object]


def stream_interactive_agent(n_rows: int) -> collections.abc.Iterator[RawRow]:
  """Yield the first ``n_rows`` rows of the pinned file from the Hub."""
  url = (
    f'https://huggingface.co/datasets/{config.NEMOTRON_REPO}/resolve/'
    f'{config.NEMOTRON_REVISION}/{config.NEMOTRON_FILE}'
  )
  with requests.get(url, stream=True, timeout=120) as resp:
    resp.raise_for_status()
    for k, line in enumerate(resp.iter_lines(decode_unicode=True)):
      if k >= n_rows:
        break
      if line:
        yield json.loads(line)


def _values(obj: object, out: set[str]) -> None:
  if isinstance(obj, dict):
    for v in obj.values():
      _values(v, out)
  elif isinstance(obj, list):
    for v in obj:
      _values(v, out)
  elif isinstance(obj, (str, int, float)) and not isinstance(obj, bool):
    s = str(obj).strip()
    if len(s) >= 3 and s.lower() not in ('true', 'false', 'none', 'null'):
      out.add(s.lower())


def _threaded_values(messages: list[dict[str, object]]) -> int:
  """Call argument values that came from a tool result, not from the user."""
  user_text, tool_text, threaded = '', '', 0
  for m in messages:
    if m['role'] == 'user':
      user_text += ' ' + (m.get('content') or '').lower()
    elif m['role'] == 'tool':
      tool_text += ' ' + (m.get('content') or '').lower()
    elif m['role'] == 'assistant':
      for c in m.get('tool_calls') or []:
        try:
          args = json.loads(c['function'].get('arguments') or '{}')
        except ValueError:
          args = {}
        vs: set[str] = set()
        _values(args, vs)
        threaded += sum(1 for v in vs if v in tool_text and v not in user_text)
  return threaded


def select_state_dependent(
  pool: list[RawRow],
) -> tuple[list[RawRow], dict[str, int]]:
  """Apply the rule above and sample ``config.NEMOTRON_N`` conversations."""
  stats: collections.Counter[str] = collections.Counter()
  seen: set[str] = set()
  keep: list[RawRow] = []
  for r in pool:
    stats['all'] += 1
    msgs = r['messages']
    key = (r.get('metadata') or {}).get('uuid') or json.dumps(msgs[:2])[:300]
    names: list[str] = []
    for m in msgs:
      if m['role'] == 'assistant':
        for c in m.get('tool_calls') or []:
          names.append(c['function']['name'])
    size = len(json.dumps(msgs)) + len(json.dumps(r.get('tools') or []))
    if r.get('filter_reason'):
      reason = 'filter_reason'
    elif key in seen:
      reason = 'duplicate'
    elif len(names) < 2:
      reason = 'fewer_than_2_calls'
    elif all('transfer' in n or 'escalat' in n for n in names):
      reason = 'transfer_only'
    elif any(
      m['role'] == 'assistant'
      and m.get('tool_calls')
      and (m.get('content') or '').strip()
      for m in msgs
    ):
      reason = 'mixed_text_and_call_turn'
    elif _threaded_values(msgs) == 0:
      reason = 'no_state_dependence'
    elif size > config.NEMOTRON_MAX_CHARS:
      reason = 'too_long'
    else:
      reason = None
    if not r.get('filter_reason'):
      seen.add(key)
    if reason:
      stats[reason] += 1
    else:
      keep.append(r)
  stats['qualifying'] = len(keep)
  random.Random(config.NEMOTRON_SEED).shuffle(keep)
  keep = keep[: config.NEMOTRON_N]
  stats['selected'] = len(keep)
  return keep, dict(stats)


def _render_calls(calls: list[dict[str, object]]) -> str | None:
  """Render OpenAI tool calls as a ToolACE call list, or None if malformed."""
  parts = []
  for c in calls:
    fn = c['function']
    args = fn.get('arguments') or '{}'
    if isinstance(args, str):
      try:
        args = json.loads(args)
      except ValueError:
        return None
    if not isinstance(args, dict):
      return None
    rendered = ', '.join(f'{k}={v!r}' for k, v in args.items())
    parts.append(f'{fn["name"]}({rendered})')
  return '[' + ', '.join(parts) + ']'


def uuid_of(raw: RawRow) -> str:
  """The dataset's conversation uuid, or a hash of its opening when missing."""
  uid = (raw.get('metadata') or {}).get('uuid')
  if uid:
    return str(uid)
  opening = json.dumps(raw['messages'][:2])[:300].encode('utf-8')
  return hashlib.sha1(opening, usedforsecurity=False).hexdigest()[:16]


def to_toolace_row(raw: RawRow, row_id: str) -> formats.Row | None:
  """Convert an OpenAI-style row to the ToolACE row format.

  The shipped ``reasoning_content`` is dropped: the teacher writes every
  thought from scratch so both halves of the mix share one thought style.
  """
  msgs = raw['messages']
  tool_list = [
    {
      'name': t['function']['name'],
      'description': t['function'].get('description', ''),
      'parameters': t['function'].get('parameters')
      or {'type': 'object', 'properties': {}},
    }
    for t in raw.get('tools') or []
    if t.get('type') == 'function'
  ]
  system = (
    next((m['content'] for m in msgs if m['role'] == 'system'), '').strip()
    + '\n\nHere is a list of functions in JSON format that you can invoke:\n'
    + json.dumps(tool_list, ensure_ascii=False)
  )
  conv: list[dict[str, str]] = []
  for m in msgs:
    role, text, calls = m['role'], (m.get('content') or '').strip(), None
    if role == 'assistant' and m.get('tool_calls'):
      if text:
        return None
      calls = _render_calls(m['tool_calls'])
      if calls is None:
        return None
    if role == 'system':
      continue
    if role not in ('user', 'assistant', 'tool'):
      return None
    conv.append({'from': role, 'value': calls or text})
  if any(not m['value'] for m in conv):
    return None
  return {
    'id': row_id,
    'system': system,
    'conversations': conv,
    'domain': raw.get('domain'),
    'uuid': (raw.get('metadata') or {}).get('uuid'),
  }
