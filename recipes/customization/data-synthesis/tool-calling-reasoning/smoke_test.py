"""Check every stage on conversations built here, and the teacher if a key is set.

The offline part builds small tool-calling conversations and the replies a
teacher would give for them, then exercises every stage after the teacher:
parse, verification, cleaning, audit, assemble, mix, the selection rule,
resume bookkeeping and the parallel-call converter. The live part runs when
``REFLECTION_API_KEY`` is set: it sends one conversation to the teacher and
checks that both assistant turns received an accepted thought.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import assemble
import audit
import config
import dotenv
import formats
import mix
import teacher
import toolace

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = [
  {
    'name': 'get_weather',
    'description': 'Current weather for a city.',
    'parameters': {
      'type': 'object',
      'properties': {'city': {'type': 'string'}},
      'required': ['city'],
    },
  }
]
ROW = {
  'id': 'smoke_0',
  'system': (
    'You are an assistant with access to functions. Here is a list of '
    'functions in JSON format that you can invoke:\n' + json.dumps(TOOLS)
  ),
  'conversations': [
    {'from': 'user', 'value': 'Is it raining in Paris right now?'},
    {'from': 'assistant', 'value': '[get_weather(city="Paris")]'},
    {
      'from': 'tool',
      'value': '[{"name": "get_weather", "results": {"condition": "rain"}}]',
    },
    {'from': 'assistant', 'value': 'Yes, it is raining in Paris right now.'},
  ],
}
THOUGHTS = (
  'The user asks about the current weather in Paris. get_weather is the '
  'only function and takes a city, so I call it with city set to Paris.',
  'The tool reported rain for Paris, which answers the question directly, '
  'so I state that it is raining and make no further call.',
)


class FakeTeacher:
  """Returns the reply a teacher gives: the conversation plus thoughts."""

  model = 'fake-teacher'

  def complete(self, prompt: str, max_tokens: int) -> teacher.Record:
    """Ignore the prompt; wrap the known conversation with thoughts."""
    del prompt, max_tokens
    rewritten = teacher.wrap_conversation(ROW)
    for turn, thought in zip(rewritten[2::2], THOUGHTS, strict=True):
      turn['value'] = f'<thought>{thought}</thought>\n{turn["value"]}'
    return {
      'content': json.dumps(rewritten, indent=2),
      'finish_reason': 'stop',
      'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
    }


def _row(row_id: str, turns: list[tuple[str, str]]) -> formats.Row:
  """A ToolACE-format row over the weather tool."""
  return {
    'id': row_id,
    'system': ROW['system'],
    'conversations': [{'from': f, 'value': v} for f, v in turns],
  }


def _records(work: pathlib.Path, name: str, records: list[dict]) -> str:
  path = work / name
  path.write_text(
    ''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8'
  )
  return str(path)


def offline(work: pathlib.Path) -> None:
  """Run every stage after the teacher on the conversation built above."""
  records = teacher.rewrite_conversation(
    FakeTeacher(), 0, ROW, config.TeacherConfig()
  )
  assert [r['ok'] for r in records] == [True, True], records
  assert [r['reasoning'] for r in records] == list(THOUGHTS), records
  assert all(r['id'] == 'smoke_0' for r in records), records
  by_turn, failed, unknown = audit.load_records(
    _records(work, 'records.jsonl', records), [ROW]
  )
  assert (len(by_turn), failed, unknown) == (2, 0, 0), (failed, unknown)
  summary, bad_turns = audit.audit([ROW], by_turn)
  assert not bad_turns and summary['conversations'] == 1, (summary, bad_turns)
  rows, stats = assemble.assemble([ROW], by_turn, set(), 'none', no_ask=True)
  assert stats['rows_written'] == 1, stats
  assistant = [m for m in rows[0]['messages'] if m['role'] == 'assistant']
  assert [m['reasoning_content'] for m in assistant] == list(THOUGHTS)
  call = assistant[0]['tool_calls'][0]['function']
  assert call['name'] == 'get_weather', call
  assert json.loads(call['arguments']) == {'city': 'Paris'}, call
  assert [t['function']['name'] for t in rows[0]['tools']] == ['get_weather']
  part = work / 'messages.jsonl'
  part.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
  mixed = mix.mix(str(part), str(part), str(work / 'train.jsonl'))
  assert mixed['total_rows'] == 2, mixed
  print(
    'ok: parse, verify, clean, audit, assemble and mix on one conversation '
    f'({len(records)} thoughts, {mixed["total_rows"]} rows mixed)'
  )
  checks(work)


def checks(work: pathlib.Path) -> None:
  """The focused cases: parallel results, selection, resume, ids, calls."""
  parallel = _row(
    'smoke_parallel',
    [
      ('user', 'Weather in Paris and Rome?'),
      ('assistant', '[get_weather(city="Paris"), get_weather(city="Rome")]'),
      ('tool', '[{"name": "get_weather", "results": {"condition": "rain"}}]'),
      ('tool', '[{"name": "get_weather", "results": {"condition": "sun"}}]'),
      ('assistant', 'Rain in Paris, sun in Rome.'),
    ],
  )
  converted = formats.to_openai_messages(parallel, parallel['id'])
  assert converted is not None
  calls = [m for m in converted['messages'] if m.get('tool_calls')][0]
  results = [m for m in converted['messages'] if m['role'] == 'tool']
  assert [r['tool_call_id'] for r in results] == [
    c['id'] for c in calls['tool_calls']
  ], results
  assert 'rain' in results[0]['content'] and 'sun' in results[1]['content']
  orphan = _row(
    'smoke_orphan',
    [
      ('user', 'Hi'),
      ('assistant', 'Hello, how can I help?'),
      ('tool', '[{"name": "get_weather", "results": {}}]'),
    ],
  )
  assert formats.to_openai_messages(orphan, orphan['id']) is None
  print(
    'ok: parallel results land on their calls; a result with no call is rejected'
  )

  text_only = _row(
    'smoke_text', [('user', 'Hi'), ('assistant', 'Hello, how can I help?')]
  )
  keep, kinds = toolace.select_rows([ROW, text_only, parallel])
  assert keep == [0, 2] and 'text_only' not in kinds, (keep, kinds)
  print('ok: the ToolACE rule skips conversations without a call')

  first_turn_only = teacher.rewrite_conversation(
    FakeTeacher(), 0, ROW, config.TeacherConfig()
  )[:1]
  partial = pathlib.Path(_records(work, 'partial.jsonl', first_turn_only))
  assert teacher.load_done(partial, [ROW]) == set()
  complete = pathlib.Path(
    _records(
      work,
      'complete.jsonl',
      teacher.rewrite_conversation(
        FakeTeacher(), 0, ROW, config.TeacherConfig()
      ),
    )
  )
  assert teacher.load_done(complete, [ROW]) == {'smoke_0'}
  print('ok: a conversation with a missing turn is sent again on resume')

  foreign = [{**r, 'id': 'someone_else'} for r in first_turn_only]
  by_turn, _, unknown = audit.load_records(
    _records(work, 'foreign.jsonl', foreign), [ROW]
  )
  assert not by_turn and unknown == 1, (by_turn, unknown)
  print('ok: a record whose id matches no conversation is skipped')

  ny = _row(
    'smoke_ny',
    [
      ('user', 'Weather in New York?'),
      ('assistant', '[get_weather(city="New York", units="c")]'),
    ],
  )
  wrapped = teacher.wrap_conversation(ny)
  thought = '<thought>' + THOUGHTS[0] + '</thought>\n'
  inside = [dict(t) for t in wrapped]
  inside[2]['value'] = (
    thought + '<tool>[get_weather(city="New  York", units="c")]</tool>'
  )
  _, failures = teacher.verify_rewrite(wrapped, inside)
  assert failures == ['call_modified@2'], failures
  between = [dict(t) for t in wrapped]
  between[2]['value'] = (
    thought + '<tool>[get_weather(city="New York",\n    units="c")]</tool>'
  )
  thoughts, failures = teacher.verify_rewrite(wrapped, between)
  assert not failures and 2 in thoughts, failures
  print(
    'ok: whitespace inside an argument value must match; line breaks '
    'between arguments may differ'
  )


def live(work: pathlib.Path) -> None:
  """Send the conversation to the teacher and check both thoughts."""
  rows_file = work / 'live_rows.jsonl'
  rows_file.write_text(json.dumps(ROW) + '\n', encoding='utf-8')
  out = work / 'live_records.jsonl'
  subprocess.run(
    [sys.executable, 'main.py', 'enrich', '--rows', rows_file, '--out', out],
    cwd=HERE,
    check=True,
    stdout=subprocess.DEVNULL,
  )
  with out.open(encoding='utf-8') as f:
    records = [json.loads(line) for line in f if line.strip()]
  failures = [r for r in records if r.get('msg') == -1]
  assert not failures, f'teacher failed: {failures[0].get("error")}'
  accepted = [r for r in records if r.get('ok')]
  assert len(accepted) == 2, f'{len(accepted)} accepted thoughts, expected 2'
  print(
    f'ok: {accepted[0]["teacher"]["model"]} wrote {len(accepted)} accepted '
    f'thoughts for 2 assistant turns '
    f'({accepted[0]["usage"]["completion_tokens"]} completion tokens)'
  )


def main() -> None:
  """Run the offline check, then the live check when a key is available."""
  dotenv.load_dotenv(HERE / '.env')
  with tempfile.TemporaryDirectory() as tmp:
    offline(pathlib.Path(tmp))
    if os.environ.get('REFLECTION_API_KEY'):
      live(pathlib.Path(tmp))
    else:
      print('skipped: live teacher check (REFLECTION_API_KEY is not set)')


if __name__ == '__main__':
  main()
