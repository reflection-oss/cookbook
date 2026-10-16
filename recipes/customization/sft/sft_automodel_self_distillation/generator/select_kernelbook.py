# Research script kept as run (formatting only); lint exemptions below.
# ruff: noqa: ANN201,ARG001,B007,BLE001,D103,E741,FURB118,N806,PLC0415,SIM102
"""Family K selection from KernelBook (GPUMODE/KernelBook, dataset_permissive.parquet), PyTorch side only.
Filters: non-synthetic (human GitHub code), <= 40 lines, imports only torch/math/typing, no control flow or
randomness in the module body, forward returns a single finite floating tensor, no input mutation / aliasing,
deterministic, CPU runtime < 2 s. Runs each surviving module in sharded, niced subprocesses with a per-module alarm.
Usage: select_kernelbook.py static | shard <i> <n> | merge
"""

import ast
import json
import pathlib
import random
import re
import signal
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / 'data'
ALLOWED = {'torch', 'math', 'typing'}
RAND_RE = re.compile(
  r'dropout|Dropout|\brand|multinomial|bernoulli|uniform_|normal_|randperm|manual_seed|\.cuda\(|cuda|print\(|numpy|\bnp\.'
)


def static_filter(row):
  src = row['python_code']
  if row.get('synthetic'):
    return None
  if src.count('\n') + 1 > 40:
    return None
  try:
    tree = ast.parse(src)
  except SyntaxError:
    return None
  for n in ast.walk(tree):
    if isinstance(n, ast.Import):
      if any(a.name.split('.')[0] not in ALLOWED for a in n.names):
        return None
    elif isinstance(n, ast.ImportFrom):
      if n.level or (n.module or '').split('.')[0] not in ALLOWED:
        return None
  classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
  entry = [c for c in classes if c.name == row['entry_point']]
  if not entry:
    return None
  cls = entry[0]
  fwd = [
    n
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == 'forward'
  ]
  if not fwd:
    return None
  for n in ast.walk(cls):
    if isinstance(
      n,
      (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.ListComp,
        ast.GeneratorExp,
        ast.IfExp,
        ast.With,
      ),
    ):
      return None
  fnames = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
  if not {'get_inputs', 'get_init_inputs'} <= fnames:
    return None
  cls_src = ast.get_source_segment(src, cls) or ''
  if RAND_RE.search(cls_src) or '.cuda' in src or 'device=' in src:
    return None
  ops = sorted(
    {
      n.attr
      for n in ast.walk(cls)
      if isinstance(n, ast.Attribute)
      and not (isinstance(n.value, ast.Name) and n.value.id == 'self')
    }
    - {'__init__', 'shape', 'size', 'view', 'dtype', 'device'}
  )
  return dict(
    uuid=int(row['uuid']),
    entry_point=row['entry_point'],
    repo_name=row['repo_name'],
    sha=row['sha'],
    licenses=list(row['licenses']) if row['licenses'] is not None else [],
    python_code=src,
    ops=ops,
    n_lines=src.count('\n') + 1,
  )


def cmd_static():
  import pyarrow.parquet as pq

  t = pq.read_table(
    DATA / 'kernelbook.parquet',
    columns=[
      'uuid',
      'entry_point',
      'python_code',
      'repo_name',
      'sha',
      'licenses',
      'synthetic',
    ],
  )
  rows = t.to_pylist()
  out, seen = [], set()
  for r in rows:
    k = static_filter(r)
    if not k:
      continue
    key = re.sub(r'\s+', ' ', k['python_code']).strip()
    if key in seen:
      continue
    seen.add(key)
    out.append(k)
  (DATA / 'kb_static.jsonl').write_text(
    '\n'.join(json.dumps(o) for o in out) + '\n'
  )
  print(f'static: {len(out)} of {len(rows)} rows survive')


class Alarm(Exception):
  pass


def _alarm(*a):
  raise Alarm()


def runtime_check(k):
  import torch

  torch.set_num_threads(1)
  ns = {'__name__': 'kb'}
  exec(compile(k['python_code'], 'kb.py', 'exec'), ns)
  Model = ns[k['entry_point']]
  init = ns['get_init_inputs']()
  if (
    isinstance(init, (list, tuple))
    and len(init) == 2
    and isinstance(init[0], (list, tuple))
    and isinstance(init[1], dict)
  ):
    args, kwargs = list(init[0]), dict(init[1])
  else:
    args, kwargs = list(init), {}
  torch.manual_seed(0)
  inputs = ns['get_inputs']()
  if not inputs or not all(torch.is_tensor(t) for t in inputs):
    return None, 'inputs not tensors'
  if any(t.numel() > 1_000_000 for t in inputs):
    return None, 'input too large'
  outs, times = [], []
  for rep in range(2):
    torch.manual_seed(1000)
    m = Model(*args, **kwargs).eval()
    ins = [t.clone() for t in inputs]
    t0 = time.time()
    with torch.no_grad():
      o = m(*ins)
    times.append(time.time() - t0)
    if not torch.is_tensor(o):
      return None, 'output not a tensor'
    if not o.is_floating_point():
      return None, f'output dtype {o.dtype}'
    if not torch.isfinite(o).all():
      return None, 'non-finite output'
    if o.numel() > 1_000_000 or o.numel() == 0:
      return None, 'output size'
    if any(o.data_ptr() == t.data_ptr() for t in ins):
      return None, 'output aliases input'
    if any(not torch.equal(a, b) for a, b in zip(ins, inputs)):
      return None, 'forward mutates inputs'
    outs.append(o)
  if not torch.equal(outs[0], outs[1]):
    return None, 'nondeterministic'
  if min(times) > 2.0:
    return None, f'slow {min(times):.2f}s'
  n_params = sum(p.numel() for p in m.parameters())
  return dict(
    ref_time_s=round(min(times), 4),
    out_shape=list(outs[0].shape),
    out_dtype=str(outs[0].dtype).replace('torch.', ''),
    in_shapes=[list(t.shape) for t in inputs],
    in_dtypes=[str(t.dtype).replace('torch.', '') for t in inputs],
    n_params=int(n_params),
    n_state=len(m.state_dict()),
  ), None


def cmd_shard(i, n):
  rows = [
    json.loads(l) for l in (DATA / 'kb_static.jsonl').read_text().splitlines()
  ]
  rows = rows[i::n]
  signal.signal(signal.SIGALRM, _alarm)
  out = (DATA / f'kb_runtime_{i}.jsonl').open('w')
  ok = 0
  for j, k in enumerate(rows):
    signal.alarm(8)
    try:
      info, why = runtime_check(k)
    except Alarm:
      info, why = None, 'alarm'
    except BaseException as e:
      info, why = None, f'{type(e).__name__}: {str(e)[:120]}'
    finally:
      signal.alarm(0)
    if info:
      k.update(info)
      out.write(json.dumps(k) + '\n')
      ok += 1
    else:
      out.write(json.dumps({'uuid': k['uuid'], 'rejected': why}) + '\n')
    if j % 200 == 0:
      print(f'shard {i}: {j}/{len(rows)} ok={ok}', flush=True)
  out.close()
  print(f'shard {i} done: {ok}/{len(rows)}')


def cmd_merge(n_select=300, seed=20260904):
  rows, rejected = [], {}
  for p in sorted(DATA.glob('kb_runtime_*.jsonl')):
    for l in p.read_text().splitlines():
      r = json.loads(l)
      if 'rejected' in r:
        why = r['rejected'].split(':')[0]
        rejected[why] = rejected.get(why, 0) + 1
      else:
        rows.append(r)
  rows.sort(key=lambda r: r['uuid'])
  rng = random.Random(seed)
  sel = rng.sample(rows, min(n_select, len(rows)))
  (DATA / 'kernelbook_selected.jsonl').write_text(
    '\n'.join(json.dumps(r) for r in sel) + '\n'
  )
  stats = dict(
    eligible=len(rows),
    selected=len(sel),
    rejected=rejected,
    with_params=sum(1 for r in sel if r['n_params'] > 0),
  )
  (DATA / 'kernelbook_selection_stats.json').write_text(
    json.dumps(stats, indent=1)
  )
  print(json.dumps(stats, indent=1))


if __name__ == '__main__':
  c = sys.argv[1]
  if c == 'static':
    cmd_static()
  elif c == 'shard':
    cmd_shard(int(sys.argv[2]), int(sys.argv[3]))
  elif c == 'merge':
    cmd_merge()
