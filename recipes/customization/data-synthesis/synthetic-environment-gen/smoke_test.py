"""Check the export stage's filters and trace layout on a synthetic Harbor job.

Builds a tiny jobs folder in a temporary directory (one task with two passing
attempts, one of which reads the hidden test and must be rejected), runs
export.py on it, and checks the files it writes. Needs no Docker or API.
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SYSTEM = 'You are a helpful assistant that can interact with a computer.'
TASK = 'Please solve this issue: write hello to /home/user/out.txt'


def trajectory(commands: list[str]) -> dict:
  """A mini-swe-agent trajectory that runs `commands` and submits."""
  messages = [
    {'role': 'system', 'content': SYSTEM},
    {'role': 'user', 'content': TASK},
  ]
  for i, command in enumerate(commands):
    messages.append({
      'role': 'assistant',
      'content': '',
      'reasoning_content': f'step {i}',
      'tool_calls': [
        {
          'id': f'call_{i}',
          'type': 'function',
          'function': {
            'name': 'bash',
            'arguments': json.dumps({'command': command}),
          },
        }
      ],
    })
    messages.append({
      'role': 'tool',
      'tool_call_id': f'call_{i}',
      'content': '{"returncode": 0, "output": ""}',
    })
  return {'messages': messages, 'info': {'exit_status': 'Submitted'}}


def write_trial(jobs: pathlib.Path, name: str, commands: list[str]) -> None:
  """Write one passing Harbor trial for task_000000_smoke."""
  trial = jobs / 'solve-smoke' / f'task_000000_smoke__{name}'
  (trial / 'agent').mkdir(parents=True)
  (trial / 'config.json').write_text(
    json.dumps({'task': {'path': 'harbor/task_000000_smoke'}}),
    encoding='utf-8',
  )
  (trial / 'result.json').write_text(
    json.dumps({'verifier_result': {'rewards': {'reward': 1.0}}}),
    encoding='utf-8',
  )
  (trial / 'agent' / 'mini-swe-agent.trajectory.json').write_text(
    json.dumps(trajectory(commands)), encoding='utf-8'
  )


with tempfile.TemporaryDirectory() as tmp:
  jobs = pathlib.Path(tmp) / 'jobs'
  submit = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
  write_trial(jobs, 'a', ['ls', 'echo hello > /home/user/out.txt', submit])
  write_trial(jobs, 'b', ['echo hello > /home/user/out.txt', submit])
  write_trial(jobs, 'c', ['cat /tests/test_final_state.py', submit])
  write_trial(jobs, 'd', ['cd /tests && cat *.py', submit])
  (jobs / 'solve-smoke.exit').write_text('0\n', encoding='utf-8')
  run = subprocess.run(
    [sys.executable, 'export.py', 'harbor', str(jobs), tmp],
    cwd=HERE,
    capture_output=True,
    text=True,
    check=False,
  )
  assert run.returncode == 0, run.stderr
  files = sorted(pathlib.Path(tmp).glob('traces/*/*/trace_*.json'))
  assert [f.name for f in files] == ['trace_0.json', 'trace_1.json'], files
  for path in files:
    record = json.loads(path.read_text(encoding='utf-8'))
    assert record['task'] == path.parent.name == 'task_000000_smoke'
    roles = [m['role'] for m in record['messages']]
    assert roles[:2] == ['system', 'user'] and 'tool' in roles, roles
    assert 'Please solve this issue:' in record['messages'][1]['content']
  shortest = json.loads(files[0].read_text(encoding='utf-8'))
  assert sum(1 for m in shortest['messages'] if m['role'] == 'tool') == 2
  assert "'rejected: reads the hidden test': 2" in run.stdout, run.stdout
  manifest = (pathlib.Path(tmp) / 'manifest.tsv').read_text(encoding='utf-8')
  assert manifest.count('\n') == 2, manifest
  # A re-export with a tighter limit must leave no stale trace file behind.
  again = subprocess.run(
    [sys.executable, 'export.py', 'harbor', str(jobs), tmp],
    cwd=HERE,
    capture_output=True,
    text=True,
    check=False,
    env={**os.environ, 'MAX_PER_TASK': '1'},
  )
  assert again.returncode == 0, again.stderr
  left = sorted(
    f.name for f in pathlib.Path(tmp).glob('traces/*/*/trace_*.json')
  )
  assert left == ['trace_0.json'], left
  print(f'ok: 1 task, 2 traces | {run.stdout.strip().splitlines()[0]}')
