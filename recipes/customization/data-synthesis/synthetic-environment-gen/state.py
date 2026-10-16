"""Per-task state of the control and solve runs, read from Harbor's job output.

Usage: state.py [harbor_dir] [jobs_dir] [status | list <bucket>]
       state.py limits <task.toml>   (write config/task_overrides.toml into it)

Buckets: need_control, control_running, control_failed (the verifier passed on the untouched
environment), verifier_broken (the verifier timed out), env_broken (the environment keeps failing
to start), need_solve, solve_running, solved, failed (Beam failed every attempt).

An attempt that ends in AgentTimeoutError counts as a completed, failed attempt: the agent had
the whole budget and did not submit. Attempts that die of anything else (an API outage, an image
that did not start) are infrastructure failures and are retried, up to MAX_INFRA per task.
"""

import collections
import glob
import json
import os
import pathlib
import re
import sys
import tomllib

ATTEMPTS = int(os.environ.get('ATTEMPTS', '2'))
MAX_INFRA = (
  3  # trials that died of infrastructure errors before a task is given up
)


def load(path: str) -> dict:
  """Read a JSON file, or return {} if it is missing or unreadable."""
  try:
    with open(path, encoding='utf-8') as f:
      return json.load(f)
  except (OSError, ValueError):
    return {}


def trials(
  jobs_dir: str, prefix: str
) -> dict[str, list[tuple[str, float | None]]]:
  """{task: [(status, reward)]} over every trial of jobs named <prefix>-*."""
  out = collections.defaultdict(list)
  for job in sorted(glob.glob(f'{jobs_dir}/{prefix}-*')):
    if not os.path.isdir(job):
      continue
    running = not os.path.exists(job + '.exit')
    for trial in glob.glob(job + '/*/'):
      config = load(trial + 'config.json')
      task = os.path.basename(
        ((config.get('task') or {}).get('path') or '').rstrip('/')
      )
      if not task:
        continue
      if not os.path.exists(trial + 'result.json'):
        out[task].append(('running' if running else 'lost', None))
        continue
      result = load(trial + 'result.json')
      exception = (result.get('exception_info') or {}).get('exception_type')
      reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get(
        'reward'
      )
      if exception == 'VerifierTimeoutError':
        out[task].append(('verifier_timeout', 0.0))
      elif exception and exception != 'AgentTimeoutError':
        out[task].append(('infra', 0.0))
      else:
        out[task].append(('done', float(reward or 0.0)))
  return out


def buckets(harbor_dir: str, jobs_dir: str) -> dict[str, list[str]]:
  """Sort every Harbor task into one bucket from its trials."""
  tasks = sorted(
    d
    for d in os.listdir(harbor_dir)
    if os.path.exists(f'{harbor_dir}/{d}/task.toml')
  )
  control, solve = trials(jobs_dir, 'control'), trials(jobs_dir, 'solve')
  result = collections.defaultdict(list)
  for task in tasks:
    c, s = control.get(task, []), solve.get(task, [])
    if any(status == 'running' for status, _ in c):
      result['control_running'].append(task)
    elif not any(status == 'done' for status, _ in c):
      result[
        'env_broken'
        if sum(1 for status, _ in c if status in ('infra', 'lost')) >= MAX_INFRA
        else 'need_control'
      ].append(task)
    elif any(status == 'done' and reward >= 1 for status, reward in c):
      result['control_failed'].append(task)
    elif any(status == 'running' for status, _ in s):
      result['solve_running'].append(task)
    elif any(status == 'done' and reward >= 1 for status, reward in s):
      result['solved'].append(task)
    elif any(status == 'verifier_timeout' for status, _ in c + s):
      # The verifier hung at least once and never completed with a pass: the
      # test itself is the problem, so do not spend more attempts on it.
      result['verifier_broken'].append(task)
    elif sum(1 for status, _ in s if status == 'done') >= ATTEMPTS:
      result['failed'].append(task)
    elif sum(1 for status, _ in s if status in ('infra', 'lost')) >= MAX_INFRA:
      result['env_broken'].append(task)
    else:
      result['need_solve'].append(task)
  return result


def apply_limits(toml_path: str, overrides_path: str) -> None:
  """Write every key of the overrides file into the task.toml, section by section."""
  overrides = tomllib.loads(
    pathlib.Path(overrides_path).read_text(encoding='utf-8')
  )
  text = pathlib.Path(toml_path).read_text(encoding='utf-8')
  for section, keys in overrides.items():
    block = re.search(rf'(\[{section}\]\s*\n)([^\[]*)', text)
    if not block:
      text = text.rstrip('\n') + f'\n\n[{section}]\n'
      block = re.search(rf'(\[{section}\]\s*\n)([^\[]*)', text)
    body = block.group(2)
    for key, value in keys.items():
      rendered = (
        'true'
        if value is True
        else 'false'
        if value is False
        else repr(value)
        if isinstance(value, (int, float))
        else f'"{value}"'
      )
      if re.search(rf'(?m)^\s*{key}\s*=', body):
        body = re.sub(
          rf'(?m)^(\s*{key}\s*=\s*)[^\n]+', rf'\g<1>{rendered}', body
        )
      else:
        body = body.rstrip('\n') + f'\n{key} = {rendered}\n'
    text = text[: block.start(2)] + body + text[block.end(2) :]
  pathlib.Path(toml_path).write_text(text, encoding='utf-8')


ORDER = [
  'need_control',
  'control_running',
  'control_failed',
  'verifier_broken',
  'env_broken',
  'need_solve',
  'solve_running',
  'solved',
  'failed',
]


def main() -> None:
  """Print the bucket counts, or the tasks of one bucket."""
  args = sys.argv[1:]
  if args and args[0] == 'limits':
    here = pathlib.Path(__file__).resolve().parent
    apply_limits(args[1], str(here / 'config' / 'task_overrides.toml'))
    return
  harbor_dir = (
    args.pop(0) if args and os.path.isdir(args[0]) else 'output/harbor'
  )
  jobs_dir = args.pop(0) if args and os.path.isdir(args[0]) else 'output/jobs'
  os.makedirs(jobs_dir, exist_ok=True)
  b = buckets(harbor_dir, jobs_dir)
  if args and args[0] == 'list':
    print('\n'.join(b.get(args[1], [])))
    return
  print(' | '.join(f'{k} {len(b[k])}' for k in ORDER))


if __name__ == '__main__':
  main()
