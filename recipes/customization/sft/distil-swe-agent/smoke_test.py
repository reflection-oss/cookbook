"""Check the trace conversion and the loss mask on invented input.

Runs on a CPU in a second and needs no model. It writes five traces for
three environments: one environment with three plain traces, one with a
teacher-side rules prompt and a long thought, and one whose tool output
contains a Gemma control tag. It builds the training rows and checks the
per-environment limit, the prompt normalization, the thought cap and the
control-tag filter; then checks the loss mask on a hand-written rendering.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import prepare

TASK = 'Please solve this issue: `mean([])` raises; it should return 0.0.'
SUBMIT_NOTE = 'The action was not executed; the task was marked as complete.'
FIX = "sed -i 's#/ len(xs)#/ len(xs) if xs else 0.0#' m.py"


def _step(i: int, thought: str, command: str, output: str) -> list[dict]:
  """One assistant turn with a bash call, and the tool result after it."""
  call = {'name': 'bash', 'arguments': json.dumps({'command': command})}
  return [
    {
      'role': 'assistant',
      'content': '',
      'reasoning_content': thought,
      'tool_calls': [{'id': f'c{i}', 'type': 'function', 'function': call}],
    },
    {'role': 'tool', 'tool_call_id': f'c{i}', 'content': output},
  ]


def trace(system: str, task: str, thought: str, output: str = 'ok') -> dict:
  """A two-step invented trajectory: fix, then submit."""
  return {
    'task': 'x',
    'messages': [
      {'role': 'system', 'content': system},
      {'role': 'user', 'content': task},
      *_step(1, thought, FIX, output),
      *_step(2, 'Submitting.', 'echo COMPLETE_TASK_AND_SUBMIT', SUBMIT_NOTE),
    ],
  }


def check_rows(root: pathlib.Path) -> None:
  """Write the traces, build the rows and inspect them."""
  standard = prepare.STANDARD_SYSTEM_PROMPT
  long_thought = ' '.join(f'Hypothesis {i}: no guard.' for i in range(120))
  long_thought += ' The fix is a guard for the empty list.'
  rules_task = 'HARD RULES: sed only.\n' + TASK
  traces = {
    'env_a': [trace(standard, TASK, f'Try {i}.') for i in range(3)],
    'env_b': [trace('You are an expert.', rules_task, long_thought)],
    'env_c': [trace(standard, TASK, 'Print.', 'cat: <turn|> seen')],
  }
  for env, items in traces.items():
    (root / env).mkdir(parents=True)
    for i, item in enumerate(items):
      (root / env / f'trace_{i}.json').write_text(json.dumps(item))
  rows = prepare.load_rows({'sample': root}, 2, 2000, 220_000, 9)
  assert len(rows) == 3, f'expected 3 rows (2 of 3, 1, 0), got {len(rows)}'
  assert not any(r['env_id'] == 'env_c' for r in rows), 'control tag kept'
  for row in rows:
    messages = row['messages']
    assert messages[0]['content'] == standard
    assert messages[1]['content'] == TASK, messages[1]['content']
    assert messages[-1]['role'] == 'assistant', 'submit note was kept'
  capped = next(r for r in rows if r['env_id'] == 'env_b')['messages'][2]
  assert len(capped['reasoning_content']) < len(long_thought)
  assert capped['reasoning_content'].endswith('for the empty list.')
  print(
    'ok: rows: per-env limit, prompts normalized, thought capped, tag dropped'
  )


def check_mask() -> None:
  """The trained text is the thoughts, the calls, the stop tags, the end."""
  text = (
    '<|turn>user\nfix it<turn|>\n<|turn>model\n'
    '<|channel>thought\nLook.<channel|>'
    '<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>'
    '<|tool_response>response:bash{output:<|"|>a.py<|"|>}<tool_response|>'
    '<|channel>thought\nDone.<channel|>'
    '<|tool_call>call:bash{command:<|"|>echo DONE<|"|>}<tool_call|>'
    '<|tool_response>response:bash{output:<|"|>ok<|"|>}<tool_response|>'
    '<turn|>\n'
  )
  trained = ''.join(text[a:b] for a, b in prepare.trainable_spans(text))
  assert trained == (
    '<|channel>thought\nLook.<channel|>'
    '<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|><|tool_response>'
    '<|channel>thought\nDone.<channel|>'
    '<|tool_call>call:bash{command:<|"|>echo DONE<|"|>}<tool_call|>'
    '<|tool_response><turn|>'
  ), trained
  print('ok: loss mask covers thoughts, calls, stop tags and the turn end')


if __name__ == '__main__':
  with tempfile.TemporaryDirectory() as tmp:
    check_rows(pathlib.Path(tmp))
  check_mask()
  print('ok: smoke test passed')
