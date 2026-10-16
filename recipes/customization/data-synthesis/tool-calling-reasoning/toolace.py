"""ToolACE (Team-ACE/ToolACE, Apache-2.0): the API-grounding half of the data.

Selection rule: convert every row to OpenAI-style messages (rows without a
JSON function list or with an unparseable call are left out); keep every
conversation that contains a tool result, because it shows the model what a
call returns; add a seeded random sample of single-call conversations, the
bulk of ToolACE, so the source does not dominate the mix. Conversations
without any call are not selected: every one of them answers in text where
the act-don't-ask rule below would drop it, so sending them to the teacher
only costs tokens. After enrichment, ``assemble --no-ask`` drops
conversations with a clarification or refusal turn: ToolACE's convention is
to point out a missing parameter, and a model trained on that asks the user
for things it should look up.
"""

from __future__ import annotations

import collections
import random

import config
import datasets
import formats


def load_toolace() -> list[formats.Row]:
  """All 11,300 ToolACE rows from the Hub, in dataset order."""
  ds = datasets.load_dataset(
    config.TOOLACE_REPO, split='train', revision=config.TOOLACE_REVISION
  )
  return [
    {
      'id': f'toolace_{i}',
      'system': r['system'],
      'conversations': list(r['conversations']),
    }
    for i, r in enumerate(ds)
  ]


def select_rows(rows: list[formats.Row]) -> tuple[list[int], dict[str, int]]:
  """Indices of the rows to send to the teacher, with a breakdown by kind."""
  kinds: dict[int, str] = {}
  for i, row in enumerate(rows):
    converted = formats.to_openai_messages(row, row['id'])
    if converted is not None:
      kinds[i] = formats.conversation_kind(converted['messages'])
  with_result = ('continuation_chain', 'call_result_answer')
  keep = [i for i, k in kinds.items() if k in with_result]
  single = [i for i, k in kinds.items() if k == 'single_call']
  rng = random.Random(config.TOOLACE_SEED)
  keep += rng.sample(
    single, min(config.TOOLACE_SINGLE_CALL_SAMPLE, len(single))
  )
  keep.sort()
  return keep, dict(collections.Counter(kinds[i] for i in keep))
