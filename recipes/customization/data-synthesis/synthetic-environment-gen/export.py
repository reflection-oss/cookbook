"""Stage 4: collect the verified trajectories.

For every task that Beam solved, the passing trajectories that survive the acceptance filters are
written as <out_dir>/traces/<SOURCE>/<task>/trace_<n>.json, at most MAX_PER_TASK per task, the
shortest first (trace_0 is the one to train on if you keep one). Each file holds the task name and
the messages exactly as mini-swe-agent recorded them: system, user, assistant turns with content,
reasoning_content and tool_calls, and tool results. This is the layout the SFT recipes read.
<out_dir>/manifest.tsv has one row per task. Usage: export.py [harbor_dir] [jobs_dir] [out_dir]
"""

import collections
import glob
import itertools
import json
import operator
import os
import re
import shutil
import statistics
import sys

MAX_COMMANDS = int(os.environ.get('MAX_COMMANDS', '250'))
MAX_TOKENS = int(
  os.environ.get('MAX_TRACE_TOKENS', '128000')
)  # estimated as characters / 3.5
MAX_REPEAT = 5  # identical consecutive commands
MAX_PER_TASK = int(os.environ.get('MAX_PER_TASK', '2'))
SOURCE = os.environ.get(
  'SOURCE', 'beam'
)  # folder name under traces/ for this run


def commands(messages) -> list[str]:
  """Return the command of every tool call, in order."""
  out = []
  for m in messages:
    if m.get('role') != 'assistant':
      continue
    for call in m.get('tool_calls') or []:
      args = (call.get('function') or {}).get('arguments') or '{}'
      try:
        args = json.loads(args) if isinstance(args, str) else args
      except json.JSONDecodeError:
        args = {'command': args}
      out.append(str(args.get('command', args)))
  return out


def est_tokens(messages) -> int:
  """Estimate the token count of the messages as characters / 3.5."""
  chars = 0
  for m in messages:
    chars += len(str(m.get('content') or '')) + len(
      m.get('reasoning_content') or ''
    )
    for call in m.get('tool_calls') or []:
      chars += len(json.dumps(call))
  return int(chars / 3.5)


_HIDDEN_TEST = re.compile(r"""test_final_state|(^|[\s;&|='"(])/tests(/|\b)""")


def _touches_hidden_test(command: str) -> bool:
  """True if the command names the hidden test or the /tests folder it lives in."""
  return bool(_HIDDEN_TEST.search(command))


def reject_reason(trajectory) -> str | None:
  """Why a passing trajectory is not exported, or None if it is accepted."""
  info = trajectory.get('info') or {}
  messages = trajectory.get('messages') or []
  cmds = commands(messages)
  if info.get('exit_status') != 'Submitted':
    return f'ended with {info.get("exit_status")}, not a submit'
  if len(cmds) > MAX_COMMANDS:
    return f'{len(cmds)} commands'
  if est_tokens(messages) > MAX_TOKENS:
    return f'{est_tokens(messages)} estimated tokens'
  if any(_touches_hidden_test(c) for c in cmds):
    return 'reads the hidden test'
  run = 1
  for a, b in itertools.pairwise(cmds):
    run = run + 1 if a == b else 1
    if run >= MAX_REPEAT:
      return f'{MAX_REPEAT} identical consecutive commands'
  return None


def _read_json(path: str) -> dict:
  with open(path, encoding='utf-8') as f:
    return json.load(f)


def passing_trials(jobs_dir: str) -> dict[str, list[tuple[str, dict]]]:
  """{task: [(trial_dir, trajectory)]} for every solve trial with reward 1."""
  out = collections.defaultdict(list)
  for job in glob.glob(f'{jobs_dir}/solve-*/'):
    for trial in glob.glob(job + '*/'):
      config_path, result_path = trial + 'config.json', trial + 'result.json'
      if not (os.path.exists(config_path) and os.path.exists(result_path)):
        continue
      result = _read_json(result_path)
      reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get(
        'reward'
      ) or 0
      if result.get('exception_info') or reward < 1:
        continue
      task = os.path.basename(
        _read_json(config_path)['task']['path'].rstrip('/')
      )
      trajectory_path = trial + 'agent/mini-swe-agent.trajectory.json'
      if os.path.exists(trajectory_path):
        out[task].append((trial, _read_json(trajectory_path)))
  return out


def main() -> None:
  """Export the accepted traces and write the manifest."""
  # sys.argv[1] is the harbor dir, accepted for a uniform command line but
  # not read: everything comes from the jobs dir.
  jobs_dir = sys.argv[2] if len(sys.argv) > 2 else 'output/jobs'
  out_dir = sys.argv[3] if len(sys.argv) > 3 else 'output'
  traces_dir = os.path.join(out_dir, 'traces', SOURCE)
  # The traces folder is derived from the jobs: rebuild it from scratch so a
  # re-export with tighter limits leaves no trace file behind.
  shutil.rmtree(traces_dir, ignore_errors=True)
  os.makedirs(traces_dir)
  stats = collections.Counter()
  rows = []
  for task, trials in sorted(passing_trials(jobs_dir).items()):
    accepted = []
    for trial, trajectory in trials:
      reason = reject_reason(trajectory)
      stats['accepted' if reason is None else f'rejected: {reason}'] += 1
      if reason is None:
        accepted.append((
          len(commands(trajectory['messages'])),
          trial,
          trajectory,
        ))
    if not accepted:
      stats['solved task with no accepted trace'] += 1
      continue
    accepted.sort(key=operator.itemgetter(0))
    accepted = accepted[:MAX_PER_TASK]
    task_dir = os.path.join(traces_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    for i, (_, _, trajectory) in enumerate(accepted):
      path = os.path.join(task_dir, f'trace_{i}.json')
      with open(path, 'w', encoding='utf-8') as f:
        json.dump({'task': task, 'messages': trajectory['messages']}, f)
    model = (
      ((accepted[0][2].get('info') or {}).get('config') or {}).get('model')
      or {}
    ).get('model_name', '')
    reasoning = sum(
      len(m.get('reasoning_content') or '')
      for m in accepted[0][2]['messages']
      if m.get('role') == 'assistant'
    )
    rows.append((
      task,
      len(trials),
      len(accepted),
      accepted[0][0],
      est_tokens(accepted[0][2]['messages']),
      reasoning,
      model,
      accepted[0][1],
    ))
  with open(f'{out_dir}/manifest.tsv', 'w', encoding='utf-8') as manifest:
    manifest.write(
      'task\tpassing_attempts\texported_traces\ttrace_0_commands\ttrace_0_est_tokens\ttrace_0_reasoning_chars\tmodel\ttrace_0_trial\n'
    )
    for row in rows:
      manifest.write('\t'.join(str(x) for x in row) + '\n')
  n_traces = sum(r[2] for r in rows)
  print(
    f'export: {len(rows)} tasks, {n_traces} traces -> {traces_dir}/ | {dict(stats)}'
  )
  if rows:
    print(
      f'trace_0: median {statistics.median(r[3] for r in rows):.0f} commands, '
      f'median {statistics.median(r[5] for r in rows):.0f} reasoning characters'
    )


if __name__ == '__main__':
  main()
