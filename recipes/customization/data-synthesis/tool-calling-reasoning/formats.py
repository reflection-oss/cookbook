"""The ToolACE conversation format and its conversion to OpenAI messages.

A row is ``{"id", "system", "conversations"}``: the system prompt embeds the
function list as a JSON array, and the conversation alternates ``user``,
``assistant`` and ``tool`` turns with calls written as ``[Name(a=1, b="x")]``.
Nemotron conversations are converted into the same shape (``nemotron.py``) so
one teacher prompt, one audit and one assembly serve both sources.
"""

from __future__ import annotations

import ast
import collections.abc
import json
import re

Row = dict[str, object]
Message = dict[str, object]

_TYPE_MAP = {
  'dict': 'object',
  'int': 'integer',
  'float': 'number',
  'bool': 'boolean',
  'list': 'array',
  'str': 'string',
  'double': 'number',
  'long': 'integer',
  'tuple': 'array',
}


def is_tool_call(value: str) -> bool:
  """Whether an assistant turn is a bracketed call list ``[Name(...), ...]``."""
  v = value.strip()
  return v.startswith('[') and v.endswith(']') and '(' in v


def _split_top_level(s: str) -> list[str]:
  """Split on commas outside quotes and brackets."""
  items: list[str] = []
  buf: list[str] = []
  depth, quote, i = 0, None, 0
  while i < len(s):
    ch = s[i]
    if quote:
      buf.append(ch)
      if ch == '\\' and i + 1 < len(s):
        buf.append(s[i + 1])
        i += 1
      elif ch == quote:
        quote = None
    elif ch in '"\'':
      quote = ch
      buf.append(ch)
    elif ch == ',' and depth == 0:
      items.append(''.join(buf))
      buf = []
    else:
      depth += (ch in '([{') - (ch in ')]}')
      buf.append(ch)
    i += 1
  if ''.join(buf).strip():
    items.append(''.join(buf))
  return [x.strip() for x in items]


def _literal(v: str) -> object:
  v = v.strip()
  try:
    return ast.literal_eval(v)
  except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
    pass
  if v.lower() in ('true', 'false'):
    return v.lower() == 'true'
  if v.lower() in ('null', 'none'):
    return None
  try:
    return json.loads(v)
  except ValueError:
    return v


def parse_calls(value: str) -> tuple[list[dict[str, object]], bool]:
  """Parse ``[Name(a=1), Other(b="x")]``; ``ok`` is False on a malformed item."""
  v = value.strip()
  if not (v.startswith('[') and v.endswith(']')):
    return [], False
  calls: list[dict[str, object]] = []
  ok = True
  for item in _split_top_level(v[1:-1]):
    if not item:
      continue
    p = item.find('(')
    if p == -1 or not item.endswith(')'):
      ok = False
      continue
    args: dict[str, object] = {}
    for kv in _split_top_level(item[p + 1 : -1]):
      if not kv:
        continue
      if '=' not in kv:
        ok = False
        continue
      k, val = kv.split('=', 1)
      args[k.strip()] = _literal(val)
    calls.append({'name': item[:p].strip(), 'arguments': args})
  return calls, ok and len(calls) > 0


def extract_tools_from_system(
  system: str,
) -> tuple[list[dict[str, object]] | None, str, str]:
  """The JSON function list in the system prompt, with the text around it."""
  start = system.find('[')
  decoder = json.JSONDecoder()
  while start != -1:
    try:
      tools, end = decoder.raw_decode(system[start:])
    except ValueError:
      tools, end = None, 0
    if (
      isinstance(tools, list)
      and tools
      and all(isinstance(t, dict) and 'name' in t for t in tools)
    ):
      return tools, system[:start].strip(), system[start + end :].strip()
    start = system.find('[', start + 1)
  return None, system, ''


def sanitize_name(name: str) -> str:
  """Restrict a function name to the characters tool-call APIs accept."""
  return re.sub(r'[^A-Za-z0-9_.\-]', '_', name.strip())


def _normalize_schema(x: object) -> object:
  """Map ToolACE type names (dict, int, float, ...) to JSON Schema types."""
  if isinstance(x, dict):
    return {
      k: _TYPE_MAP.get(v.strip().lower(), v)
      if k == 'type' and isinstance(v, str)
      else _normalize_schema(v)
      for k, v in x.items()
    }
  if isinstance(x, list):
    return [_normalize_schema(v) for v in x]
  return x


def openai_tools(
  tools: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, str]]:
  """Function declarations in OpenAI format, and the raw-to-clean name map."""
  name_map = {t['name']: sanitize_name(t['name']) for t in tools}
  out: list[dict[str, object]] = []
  for t in tools:
    params = _normalize_schema(
      t.get('parameters') or {'type': 'object', 'properties': {}}
    )
    if isinstance(params, dict) and params.get('type') is None:
      params['type'] = 'object'
    out.append({
      'type': 'function',
      'function': {
        'name': name_map[t['name']],
        'description': t.get('description', ''),
        'parameters': params,
      },
    })
  return out, name_map


def to_openai_messages(
  row: Row,
  row_id: str,
  reasoning: dict[int, str] | None = None,
  system_text: str = 'none',
  call_id_prefix: str | None = None,
) -> Row | None:
  """Convert a row into ``{"id", "tools", "messages"}``; None if it cannot be.

  ``reasoning`` maps assistant-turn index to its thought (stored as
  ``reasoning_content``). ``system_text="toolace"`` keeps the text around the
  function list as a system message; ``"none"`` sends the tools only, as the
  benchmark harness does. Tool-call ids are ``call_<prefix>_<turn>_<k>``.
  """
  tools, pre_text, post_text = extract_tools_from_system(row['system'])
  if tools is None:
    return None
  oai_tools, name_map = openai_tools(tools)
  prefix = row_id if call_id_prefix is None else call_id_prefix
  messages: list[Message] = []
  if system_text == 'toolace':
    content = (pre_text + '\n' + post_text).strip()
    messages.append({'role': 'system', 'content': content})
  pending: list[tuple[str, str]] = []
  for j, m in enumerate(row['conversations']):
    if m['from'] == 'user':
      messages.append({'role': 'user', 'content': m['value']})
    elif m['from'] == 'assistant':
      msg: Message = {'role': 'assistant', 'content': ''}
      if reasoning and j in reasoning:
        msg['reasoning_content'] = reasoning[j]
      if is_tool_call(m['value']):
        calls, ok = parse_calls(m['value'])
        if not ok:
          return None
        pending = []
        tool_calls: list[dict[str, object]] = []
        for k, c in enumerate(calls):
          cid = f'call_{prefix}_{j}_{k}'
          pending.append((cid, c['name']))
          tool_calls.append({
            'id': cid,
            'type': 'function',
            'function': {
              'name': name_map.get(c['name'], sanitize_name(c['name'])),
              'arguments': json.dumps(c['arguments'], ensure_ascii=False),
            },
          })
        msg['tool_calls'] = tool_calls
      else:
        msg['content'] = m['value']
        pending = []  # a text reply closes the open calls
      messages.append(msg)
    elif m['from'] == 'tool':
      if not pending:
        return None  # a result with no call to answer
      try:
        results = json.loads(m['value'])
      except ValueError:
        results = None
      if (
        isinstance(results, list)
        and len(results) == len(pending)
        and all(isinstance(r, dict) for r in results)
      ):
        bodies = [r.get('results', r) for r in results]
        owners, pending = pending, []
      else:
        # One result message per call, in call order.
        bodies = [m['value']]
        owners = [pending.pop(0)]
      for (cid, cname), body in zip(owners, bodies, strict=True):
        messages.append({
          'role': 'tool',
          'tool_call_id': cid,
          'name': name_map.get(cname, sanitize_name(cname)),
          'content': body
          if isinstance(body, str)
          else json.dumps(body, ensure_ascii=False),
        })
  return {'id': row_id, 'tools': oai_tools, 'messages': messages}


def conversation_kind(messages: list[Message]) -> str:
  """text_only, single_call, call_result_answer or continuation_chain."""
  calls = [
    i
    for i, m in enumerate(messages)
    if m['role'] == 'assistant' and m.get('tool_calls')
  ]
  if not calls:
    return 'text_only'
  if any(messages[i - 1]['role'] == 'tool' for i in calls if i > 0):
    return 'continuation_chain'
  if any(m['role'] == 'tool' for m in messages):
    return 'call_result_answer'
  return 'single_call'


def has_clarification_turn(messages: list[Message]) -> bool:
  """Whether the assistant replies with text that does not follow a result."""
  return any(
    m['role'] == 'assistant'
    and not m.get('tool_calls')
    and (i == 0 or messages[i - 1]['role'] != 'tool')
    for i, m in enumerate(messages)
  )


def read_jsonl(path: str) -> list[Row]:
  """Read the JSON object on every non-empty line of `path`."""
  with open(path, encoding='utf-8') as f:
    return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: collections.abc.Iterable[Row]) -> int:
  """Write one JSON object per line; returns the number of rows written."""
  n = 0
  with open(path, 'w', encoding='utf-8') as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + '\n')
      n += 1
  return n
