"""Combine the two halves into one shuffled training file."""

from __future__ import annotations

import random
import statistics

import config
import formats


def mix(toolace_file: str, nemotron_file: str, output: str) -> dict[str, int]:
  """Prefix the Nemotron ids, shuffle both halves together and write them."""
  toolace = formats.read_jsonl(toolace_file)
  nemotron = formats.read_jsonl(nemotron_file)
  for r in nemotron:
    r['id'] = config.NEMOTRON_ID_PREFIX + str(r['id'])
  rows = toolace + nemotron
  random.Random(config.MIX_SEED).shuffle(rows)
  formats.write_jsonl(output, rows)
  chars: list[int] = []
  for r in rows:
    for m in r['messages']:
      if m['role'] == 'assistant':
        chars.append(len(m.get('reasoning_content') or ''))
  return {
    'toolace_rows': len(toolace),
    'nemotron_rows': len(nemotron),
    'total_rows': len(rows),
    'mean_thought_chars': round(statistics.mean(chars)) if chars else 0,
  }
