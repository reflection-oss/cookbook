"""Build the training file from the source rows and the teacher's thoughts.

A conversation is written only when every assistant turn has an accepted
thought: a missing turn, an audited-out turn, or a thought carrying a drop
flag removes the whole conversation. With ``no_ask`` set, conversations that
contain a clarification or refusal turn are removed as well.
"""

from __future__ import annotations

import collections
import json

import config
import formats
import teacher

Record = dict[str, object]
TurnKey = tuple[int, int]


def assemble(
  rows: list[formats.Row],
  records: dict[TurnKey, Record],
  excluded: set[TurnKey],
  system_text: str = 'none',
  no_ask: bool = False,
) -> tuple[list[formats.Row], dict[str, int]]:
  """Return the training rows and the counts of what was kept and dropped."""
  recs = {k: v for k, v in records.items() if k not in excluded}
  stats: collections.Counter[str] = collections.Counter()
  out: list[formats.Row] = []
  for i in sorted({r for r, _ in recs}):
    row, conv = rows[i], rows[i]['conversations']
    stats['rows_seen'] += 1
    targets = [j for j, m in enumerate(conv) if m['from'] == 'assistant']
    if any((i, j) not in recs for j in targets):
      stats['rows_dropped_missing_turn'] += 1
      continue
    thoughts: dict[int, str] = {}
    dropped = False
    for j in targets:
      kind = 'call' if formats.is_tool_call(conv[j]['value']) else 'text'
      cleaned, flags = teacher.clean_and_flag(
        recs[(i, j)]['reasoning'], kind, conv[j]['value']
      )
      drop = set(flags) & config.DROP_FLAGS
      if drop:
        for f in drop:
          stats[f'drop_flag:{f}'] += 1
        dropped = True
        break
      thoughts[j] = cleaned
    if dropped:
      stats['rows_dropped_flags'] += 1
      continue
    converted = formats.to_openai_messages(
      row, row['id'], thoughts, system_text=system_text, call_id_prefix=str(i)
    )
    if converted is None:
      stats['rows_dropped_unconvertible'] += 1
      continue
    if no_ask and formats.has_clarification_turn(converted['messages']):
      stats['rows_dropped_clarification_turn'] += 1
      continue
    out.append(converted)
    stats['rows_written'] += 1
    stats['thought_chars'] += sum(len(t) for t in thoughts.values())
    stats['thoughts'] += len(thoughts)
  summary = dict(stats)
  summary['mean_thought_chars'] = round(
    stats['thought_chars'] / max(1, stats['thoughts'])
  )
  return out, summary


def load_excluded(path: str | None) -> set[TurnKey]:
  """The (row, msg) turns listed by the audit, or an empty set."""
  if not path:
    return set()
  with open(path, encoding='utf-8') as f:
    return {(e['row'], e['msg']) for e in json.load(f)}
