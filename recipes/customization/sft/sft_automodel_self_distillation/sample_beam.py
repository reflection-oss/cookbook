"""Run Beam on the rebuilt task corpus and keep the verified answers (self-distillation data generation).

For every task: up to --attempts samples from an OpenAI-compatible endpoint (temperature 1, top_p 1, reasoning effort
max, up to --max-tokens tokens, streamed). The candidate kernel is parsed out of the answer and graded by verify_gpu.py
on a local GPU; the task closes at its first PASS. Resumable: re-run with the same --out.

Endpoint or network errors do not consume attempts. The same attempt is retried up to --endpoint-retries times with a
60 s pause; when the retries run out the task is reported as an endpoint failure and the run exits with code 1 so the
outage is visible. An attempt counts only once its result is persisted.

Output layout:
  OUT/raw/<id>/s<k>.json      {id, k, prompt_sha256, response{reasoning, content, usage, finish_reason}, parse_mode, status, verdict}
  OUT/results/attempts.jsonl  one line per graded sample; status PASS|FAIL|RUNTIME_ERROR|TORCH_FALLBACK|...|NO_ANSWER|TRUNCATED

The verifier executes model-generated code. It restricts imports, disables sockets and caps memory and time, but it
does not isolate the filesystem: run this on a dedicated machine or container that holds no credentials or data you
care about.

usage: sample_beam.py --corpus DIR/corpus.jsonl --prompts DIR/prompts --out RUN --base-url http://host:8000/v1 --model beam
       [--attempts 16] [--concurrency 32] [--verify-workers 8] [--gpus 8] [--max-tokens 160000] [--effort max] [--ids a,b]
       [--verifier verify_gpu.py] [--verify-python verifier_dir/.venv/bin/python] [--api-key KEY] [--endpoint-retries 5]
       [--dry-run]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time

HDR = re.compile(r'^###\s+(\S+)\s*$', re.M)
FENCE = re.compile(r'```[^\n]*\n(.*?)```', re.S)
RETRY_PAUSE_S = 60.0


def parse_answer(tid: str, text: str) -> tuple[str, str | None]:
  """Same rules as our sampling run: the fenced block under `### <id>` (exact or descriptive heading), else the last
  fenced block that defines ModelNew. Returns (mode, code|None)."""
  heads = list(HDR.finditer(text))
  parts = {}
  for i, m in enumerate(heads):
    seg = text[
      m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)
    ]
    f = FENCE.search(seg)
    parts[m.group(1).strip('`')] = f.group(1).rstrip('\n') if f else None
  if parts.get(tid):
    return 'exact_header', parts[tid]
  heads = list(re.finditer(r'^#{1,4}[^\n]*$', text, re.M))
  for i, m in enumerate(heads):
    if tid in m.group(0):
      seg = text[
        m.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)
      ]
      f = FENCE.search(seg)
      if f:
        return 'desc_header', f.group(1).rstrip('\n')
  for f in FENCE.finditer(text):
    if 'class ModelNew' in f.group(1):
      return 'lone_block', f.group(1).rstrip('\n')
  return 'none', None


class Sampler:
  """Samples tasks from the endpoint, grades the candidates and records every attempt under the run directory."""

  def __init__(self, a: argparse.Namespace, prompts: dict[str, str]) -> None:
    """Prepare the run directory, the verifier settings and the per-task progress from a previous run."""
    self.a = a
    self.prompts = prompts
    self.out = pathlib.Path(a.out)
    (self.out / 'raw').mkdir(parents=True, exist_ok=True)
    (self.out / 'results').mkdir(exist_ok=True)
    (self.out / 'scratch').mkdir(exist_ok=True)
    self.attempts_file = self.out / 'results' / 'attempts.jsonl'
    self.lock = threading.Lock()
    self.verify_python = a.verify_python or str(
      pathlib.Path(a.verifier).resolve().parent / '.venv' / 'bin' / 'python'
    )
    self.vsem = threading.Semaphore(a.verify_workers)
    self.slot_lock = threading.Lock()
    self.slots = list(range(a.verify_workers))
    self.done: dict[str, int] = {}
    self.solved: set[str] = set()
    if self.attempts_file.exists():
      with open(self.attempts_file, encoding='utf-8') as fh:
        for line in fh:
          if line.strip():
            r = json.loads(line)
            self.done[r['id']] = max(self.done.get(r['id'], -1), r['k'])
            if r['status'] == 'PASS':
              self.solved.add(r['id'])

  def todo(self, ids: list[str]) -> list[str]:
    """Tasks that are neither solved nor out of attempts."""
    return [
      tid
      for tid in ids
      if tid not in self.solved and self.done.get(tid, -1) + 1 < self.a.attempts
    ]

  def grade(self, tid: str, code: str) -> dict:
    """Grade one candidate kernel with verify_gpu.py on a GPU slot; returns the verdict dict.

    The candidate and the verifier's sandbox live in a temporary directory that is removed afterwards, whatever
    the verdict; the stderr tail of a grader failure is kept in the verdict message.
    """
    a = self.a
    with self.vsem:
      with self.slot_lock:
        slot = self.slots.pop()
      try:
        with tempfile.TemporaryDirectory(
          prefix=f'{tid}_', dir=self.out / 'scratch'
        ) as d:
          cand = pathlib.Path(d) / 'candidate.py'
          cand.write_text(code + '\n', encoding='utf-8')
          env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(slot % a.gpus),
            TRITON_F32_DEFAULT='ieee',
          )
          p = subprocess.run(
            [
              self.verify_python,
              a.verifier,
              'single',
              '--corpus',
              a.corpus,
              '--task-id',
              tid,
              '--candidate',
              str(cand),
              '--scratch',
              d,
              '--mode',
              'full',
              '--soft-timeout',
              str(a.soft_timeout),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=a.soft_timeout * 4 + 300,
            check=False,
          )
          m = re.search(r'\{.*\}', p.stdout, re.S)
          return (
            json.loads(m.group(0))
            if m
            else {
              'status': 'GRADER_ERROR',
              'msg': (p.stderr or p.stdout)[-500:],
            }
          )
      except Exception as e:  # noqa: BLE001  a grader failure is a verdict, not a crash of the run
        return {'status': 'GRADER_ERROR', 'msg': str(e)[:500]}
      finally:
        with self.slot_lock:
          self.slots.append(slot)

  def request(self, tid: str) -> tuple[str, str, dict | None, str | None]:
    """One streamed chat completion; returns (reasoning, content, usage, finish_reason). Raises on endpoint errors."""
    import requests  # noqa: PLC0415  (pip install requests)

    a = self.a
    body = dict(
      model=a.model,
      messages=[{'role': 'user', 'content': self.prompts[tid]}],
      max_tokens=a.max_tokens,
      temperature=a.temperature,
      top_p=a.top_p,
      reasoning_effort=a.effort,
      stream=True,
      stream_options={'include_usage': True},
    )
    reasoning, content, usage, finish = [], [], None, None
    with requests.post(
      f'{a.base_url.rstrip("/")}/chat/completions',
      json=body,
      headers={'Authorization': f'Bearer {a.api_key}'},
      stream=True,
      timeout=(30, a.read_timeout),
    ) as r:
      r.raise_for_status()
      for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith('data:'):
          continue
        d = line[5:].strip()
        if d == '[DONE]':
          break
        j = json.loads(d)
        if j.get('usage'):
          usage = j['usage']
        for ch in j.get('choices') or []:
          dl = ch.get('delta') or {}
          if dl.get('reasoning_content'):
            reasoning.append(dl['reasoning_content'])
          if dl.get('content'):
            content.append(dl['content'])
          if ch.get('finish_reason'):
            finish = ch['finish_reason']
    return ''.join(reasoning), ''.join(content), usage, finish

  def sample(self, tid: str, k: int) -> str:
    """Draw sample k for a task, grade it and persist it; returns the status. Raises on endpoint errors before
    anything is written, so the caller can retry the same k."""
    a = self.a
    t0 = time.time()
    reasoning, text, usage, finish = self.request(tid)
    mode, code = parse_answer(tid, text)
    if code is None:
      verdict = {
        'status': 'TRUNCATED' if finish == 'length' else 'NO_ANSWER',
        'msg': 'no answer block',
      }
    else:
      verdict = self.grade(tid, code)
    rec = dict(
      id=tid,
      k=k,
      model=a.model,
      effort=a.effort,
      prompt_sha256=hashlib.sha256(self.prompts[tid].encode()).hexdigest(),
      seconds=round(time.time() - t0, 1),
      response=dict(
        reasoning=reasoning, content=text, usage=usage, finish_reason=finish
      ),
      parse_mode=mode,
      status=verdict.get('status'),
      verdict=verdict,
    )
    (self.out / 'raw' / tid).mkdir(exist_ok=True)
    (self.out / 'raw' / tid / f's{k}.json').write_text(
      json.dumps(rec), encoding='utf-8'
    )
    with self.lock, open(self.attempts_file, 'a', encoding='utf-8') as f:
      f.write(
        json.dumps(
          dict(
            id=tid,
            k=k,
            status=rec['status'],
            parse_mode=mode,
            seconds=rec['seconds'],
            t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          )
        )
        + '\n'
      )
    return rec['status']

  def run_task(self, tid: str) -> str:
    """Sample a task until its first PASS or the attempt budget.

    Returns 'PASS', 'EXHAUSTED' (budget spent without a PASS) or 'ENDPOINT_ERROR' (the endpoint kept failing on one
    attempt; nothing was recorded for it and a later run resumes at the same k).
    """
    a = self.a
    k = self.done.get(tid, -1) + 1
    while k < a.attempts:
      status = None
      for retry in range(a.endpoint_retries + 1):
        try:
          status = self.sample(tid, k)
          break
        except Exception as e:  # noqa: BLE001  endpoint/network error: nothing was persisted, retry the same k
          left = a.endpoint_retries - retry
          print(
            f'{tid} s{k}: endpoint error {str(e)[:120]}; {left} retries left',
            flush=True,
          )
          if left:
            time.sleep(RETRY_PAUSE_S)
      if status is None:
        print(
          f'{tid} s{k}: endpoint failed {a.endpoint_retries + 1} times, giving up on this task',
          flush=True,
        )
        return 'ENDPOINT_ERROR'
      print(f'{time.strftime("%H:%M:%S")} {status:16s} {tid} s{k}', flush=True)
      if status == 'PASS':
        return 'PASS'
      k += 1
    return 'EXHAUSTED'


def parse_args(argv: list[str] | None) -> argparse.Namespace:
  """Command line of the sampler."""
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument('--corpus', required=True)
  ap.add_argument('--prompts', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--base-url', required=True)
  ap.add_argument('--model', default='beam')
  ap.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY', 'none'))
  ap.add_argument('--attempts', type=int, default=16)
  ap.add_argument('--concurrency', type=int, default=32)
  ap.add_argument('--verify-workers', type=int, default=8)
  ap.add_argument('--gpus', type=int, default=8)
  ap.add_argument('--max-tokens', type=int, default=160000)
  ap.add_argument('--effort', default='max')
  ap.add_argument('--temperature', type=float, default=1.0)
  ap.add_argument('--top-p', type=float, default=1.0)
  ap.add_argument('--ids')
  ap.add_argument(
    '--verifier',
    default=str(pathlib.Path(__file__).resolve().parent / 'verify_gpu.py'),
  )
  ap.add_argument('--verify-python')
  ap.add_argument('--soft-timeout', type=float, default=120.0)
  ap.add_argument('--read-timeout', type=float, default=1800.0)
  ap.add_argument(
    '--endpoint-retries',
    type=int,
    default=5,
    help='retries of the same attempt after an endpoint or network error, 60 s apart',
  )
  ap.add_argument(
    '--dry-run',
    action='store_true',
    help='parse + grade nothing: just list what would run',
  )
  return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  """Sample every selected task; returns 0, or 1 when any task hit the endpoint retry limit."""
  a = parse_args(argv)
  tasks = {}
  with open(a.corpus, encoding='utf-8') as fh:
    for line in fh:
      if line.strip():
        r = json.loads(line)
        tasks[r['id']] = r
  ids = sorted(a.ids.split(',')) if a.ids else sorted(tasks)
  prompts = {
    tid: (pathlib.Path(a.prompts) / f'{tid}.md').read_text(encoding='utf-8')
    for tid in ids
  }
  sampler = Sampler(a, prompts)
  todo = sampler.todo(ids)
  print(
    f'tasks {len(ids)} | solved {len(sampler.solved)} | to run {len(todo)} | '
    f'attempts/task {a.attempts} | model {a.model} effort {a.effort}',
    flush=True,
  )
  if a.dry_run:
    return 0
  with cf.ThreadPoolExecutor(a.concurrency) as ex:
    outcomes = list(ex.map(sampler.run_task, todo))
  n_pass = 0
  if sampler.attempts_file.exists():
    with open(sampler.attempts_file, encoding='utf-8') as fh:
      n_pass = sum(1 for line in fh if '"status": "PASS"' in line)
  endpoint_failures = [
    tid for tid, o in zip(todo, outcomes) if o == 'ENDPOINT_ERROR'
  ]
  print(
    f'ALL_DONE solved_now={outcomes.count("PASS")} total_tasks_with_PASS={n_pass} '
    f'endpoint_failures={len(endpoint_failures)} attempts_file={sampler.attempts_file}',
    flush=True,
  )
  if endpoint_failures:
    print(
      f'endpoint failures, re-run to resume: {endpoint_failures[:10]}',
      file=sys.stderr,
    )
    return 1
  return 0


if __name__ == '__main__':
  sys.exit(main())
