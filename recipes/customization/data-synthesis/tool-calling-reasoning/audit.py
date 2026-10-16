"""Audit the teacher's thoughts before training.

A thought is dropped, and with it its conversation, when it leaks a token
that only exists in the tool result that follows the call (the teacher saw
the whole conversation), refers to the rewriting task, contains tool tokens
or markdown, is too short or too long, or is mostly non-ASCII.
"""

from __future__ import annotations

import collections
import json
import re

import config
import formats

Record = dict[str, object]
TurnKey = tuple[int, int]

_META = re.compile(
  r'\b(the example|this example|conversation example|given call|'
  r'provided call|provided answer|the answer provided|gold answer|gold call|'
  r'the assistant message|chain of thought|i was (asked|told) to|'
  r'the next message|rewrit(e|ing) the|'
  r'enhancing the (conversation|example|assistant)|'
  r'(keep|keeping|leave|leaving) the original|'
  r'original (content|message|reply|response|text|wording)|'
  r'content unchanged|unchanged content|verbatim)\b',
  re.I,
)
_FORMAT = re.compile(
  r'<\|[a-z_]+>|<[a-z_]+\|>|<tool>|</tool>|```|^#{1,6}\s', re.I | re.M
)
_TOKEN = re.compile(r'[A-Za-z0-9_.\-]{6,}')


def _tokens(s: str) -> set[str]:
  return {
    t.lower()
    for t in _TOKEN.findall(s)
    if not t.replace('.', '').replace('-', '').isdigit() or len(t) >= 4
  }


def _specific(t: str) -> bool:
  """A token specific enough to count as leaked from a later tool result."""
  return (
    sum(ch.isdigit() for ch in t) >= 2
    or '_' in t
    or (len(t) >= 10 and not t.isalpha())
  )


def load_records(
  path: str, rows: list[formats.Row]
) -> tuple[dict[TurnKey, Record], int, int]:
  """Accepted records keyed by (row index, msg), joined to the rows by id.

  Returns the records, how many conversations only ever failed, and how
  many records belong to no conversation in ``rows`` (they are skipped).
  """
  index_of = {row['id']: i for i, row in enumerate(rows)}
  recs: dict[TurnKey, Record] = {}
  failed_ids: set[str] = set()
  done_ids: set[str] = set()
  unknown = 0
  with open(path, encoding='utf-8') as f:
    for line in f:
      r = json.loads(line)
      i = index_of.get(r.get('id'))
      if i is None:
        unknown += 1
        continue
      if r.get('msg', -1) == -1:
        failed_ids.add(r['id'])
        continue
      done_ids.add(r['id'])
      if r.get('ok'):
        recs[(i, r['msg'])] = r
  return recs, len(failed_ids - done_ids), unknown


def _issues(row: formats.Row, j: int, text: str) -> list[str]:
  """The issues of one thought; any of them drops the turn."""
  conv = row['conversations']
  issues: list[str] = []
  if _META.search(text):
    issues.append('meta_reference')
  if _FORMAT.search(text):
    issues.append('format_tokens')
  words = len(text.split())
  if words < config.AUDIT_MIN_WORDS:
    issues.append('too_short')
  if words > config.AUDIT_MAX_WORDS:
    issues.append('too_long')
  if sum(1 for ch in text if ord(ch) > 127) > 0.3 * max(1, len(text)):
    issues.append('non_ascii_heavy')
  results: list[str] = []
  for later in conv[j + 1 :]:
    if later['from'] != 'tool':
      break
    results.append(later['value'])
  if results:
    before = _tokens(row['system']) | _tokens(
      ' '.join(m['value'] for m in conv[: j + 1])
    )
    if any(
      _specific(t) and t in text.lower() and not any(t in b for b in before)
      for t in _tokens(' '.join(results)) - before
    ):
      issues.append('future_leak')
  return issues


def audit(
  rows: list[formats.Row], records: dict[TurnKey, Record]
) -> tuple[dict[str, object], list[TurnKey]]:
  """Return the summary and the (row, msg) turns to drop."""
  counts: collections.Counter[str] = collections.Counter()
  bad_turns: list[TurnKey] = []
  for (row, j), r in records.items():
    issues = _issues(rows[row], j, r['reasoning'])
    counts.update(issues)
    if issues:
      bad_turns.append((row, j))
  bad_turns.sort()
  summary: dict[str, object] = {
    'enriched_turns': len(records),
    'conversations': len({r for r, _ in records}),
    'turn_kinds': dict(
      collections.Counter(r['kind'] for r in records.values())
    ),
    'issue_counts': dict(counts.most_common()),
    'turns_to_drop': len(bad_turns),
  }
  return summary, bad_turns
