# Research script kept as run (formatting only); lint exemptions below.
# ruff: noqa: ANN201,D103,E741,N812
"""Phase 1: build the corpus (Family P procedural + Family K KernelBook), mutants, and Phase 2 prompt batches.

Outputs: data/corpus.jsonl, data/mutants.jsonl, data/prompts/batch_XXX.md, data/batches.json, phase0/tasks.jsonl
"""

import json
import pathlib
import platform
import random
import sys
import time

import torch
import triton

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ops_library as L

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / 'data'


def _versions() -> tuple[str, str, str]:
  """Python, torch and triton versions as they appear in the task texts (recreate_corpus.py pins them)."""
  return platform.python_version(), torch.__version__, triton.__version__


PYVER, TVER, TRVER = _versions()
SEED = 20260904

P_COUNTS = {1: 150, 2: 250, 3: 200, 4: 100}
P_FORMATS = {'code_only': 250, 'code_plus_nl': 250, 'nl_plus_signature': 200}
K_FORMATS = {'code_only': 150, 'code_plus_nl': 150}
N_MUTANTS = 200
BATCH = 10

CONTRACT_P = (
  'Contract: define one or more `@triton.jit` kernels and `class ModelNew(torch.nn.Module)` with the same constructor '
  'signature as `Model` ({init_sig}). `ModelNew.{forward_sig}` must return a tensor with exactly the same shape, dtype '
  'and values as `Model.forward` for any valid input shape: the harness re-binds SHAPE (2-D or 3-D with leading batch '
  'dims; sizes are not necessarily powers of two) and SCALE (1.0 and 0.01), calls get_inputs()/get_init_inputs(), and '
  'compares against Model with atol={atol}, rtol={rtol}. Compute the result with your Triton kernel(s): in `forward` '
  'you may allocate outputs (torch.empty*), do shape/stride arithmetic, call .contiguous()/.view(), compute grids and '
  'launch kernels, but do not compute the result with torch ops (this is checked statically and a kernel launch is '
  'counted). Allowed imports: torch, triton, triton.language, math, numpy. Environment: Python {py}, torch {tv}, '
  'triton {trv}, CPU interpreter mode (TRITON_INTERPRET=1 -- no tl.dot restrictions beyond the documented ones); '
  'float32{fp16}.'
)
CONTRACT_K = (
  'Contract: define one or more `@triton.jit` kernels and `class ModelNew(torch.nn.Module)` with the same constructor '
  'signature as `Model` (`Model` is an alias of `{entry}`; `get_init_inputs()` returns `[args, kwargs]`). '
  '`ModelNew.forward` takes the same inputs as `Model.forward` and must return a tensor with exactly the same shape, '
  'dtype and values (atol={atol}, rtol={rtol}); it is verified with `get_inputs()` at several seeds, but write '
  'shape-generic code with masking anyway. If `Model` has parameters or buffers, `ModelNew` must register parameters/'
  'buffers with identical state_dict keys and shapes (e.g. keep the same nn submodules for their weights): the harness '
  "copies `Model`'s state_dict into `ModelNew` before comparing. You may read those tensors (`self.fc.weight`) and pass "
  'them to kernels, but you must not call the submodules or torch ops to compute the result; in `forward` you may '
  'allocate outputs (torch.empty*), do shape/stride arithmetic, call .contiguous()/.view(), compute grids and launch '
  'kernels (a kernel launch is counted and torch computation is checked statically). Allowed imports: torch, triton, '
  'triton.language, math, numpy. Environment: Python {py}, torch {tv}, triton {trv}, CPU interpreter mode '
  '(TRITON_INTERPRET=1 -- no tl.dot restrictions beyond the documented ones); float32.'
)


def fmt_float(x):
  return repr(float(x))


def test_shapes_for(shape, rng):
  others = [s for s in L.SHAPES if tuple(s) != tuple(shape)]
  chosen = []
  if tuple(shape) != L.NPOT_SHAPE:
    chosen.append(L.NPOT_SHAPE)
    others.remove(L.NPOT_SHAPE)
  else:
    chosen.append((1, 4096))
    others.remove((1, 4096))
  rng.shuffle(others)
  chosen += others[: 3 - len(chosen)]
  return [list(shape)] + [list(s) for s in chosen]


def tolerance_for(chain_names, k_src=None):
  if k_src is not None:
    heavy = any(
      w in k_src
      for w in (
        'matmul',
        'Linear',
        'bmm',
        'einsum',
        'conv',
        'Conv',
        '@',
        'mm(',
        'addmm',
        'baddbmm',
        'cumsum',
        'cumprod',
      )
    )
    return (1e-3, 1e-4) if heavy else (1e-4, 1e-4)
  return (
    (1e-3, 1e-4)
    if any(L.OPS[n]['cat'] == 'matmul' for n in chain_names)
    else (1e-4, 1e-4)
  )


def render_spec_p(task, info, spec_format):
  fp16 = (
    ' (a float16 variant of this task exists for GPU runs; it is verified in float32 here)'
    if task['also_fp16']
    else ''
  )
  contract = CONTRACT_P.format(
    init_sig=info['init_sig'],
    forward_sig=info['forward_sig'],
    atol=task['atol'],
    rtol=task['rtol'],
    py=PYVER,
    tv=TVER,
    trv=TRVER,
    fp16=fp16,
  )
  if spec_format == 'nl_plus_signature':
    src = task['reference_src']
    # keep header, SHAPE/SCALE, constructor signature, forward signature, get_inputs, get_init_inputs; drop the forward body
    lines = src.splitlines()
    out = []
    in_forward = False
    for ln in lines:
      if ln.startswith('    def forward('):
        in_forward = True
        out.append(
          ln
          + f'  # body withheld; returns a {info["out_dtype"]} tensor of shape {tuple(info["out_shape"])} for SHAPE={tuple(task["shapes"][0])}'
        )
        out.append('        ...')
        continue
      if in_forward:
        if ln.startswith('        ') or ln.strip() == '':
          if ln.strip() == '':
            in_forward = False
            out.append(ln)
          continue
        in_forward = False
      out.append(ln)
    body = (
      'Description: '
      + info['nl']
      + '\n\nReference interface (the PyTorch forward body is withheld; implement the '
      'description):\n```python\n' + '\n'.join(out).rstrip() + '\n```\n'
    )
  else:
    body = ''
    if spec_format == 'code_plus_nl':
      body += 'Description: ' + info['nl'] + '\n\n'
    body += (
      'Reference PyTorch module (this is the spec and the ground truth):\n```python\n'
      + task['reference_src'].rstrip()
      + '\n```\n'
    )
  return body + contract


def render_spec_k(task, spec_format):
  contract = CONTRACT_K.format(
    entry=task['entry_point'],
    atol=task['atol'],
    rtol=task['rtol'],
    py=PYVER,
    tv=TVER,
    trv=TRVER,
  )
  body = ''
  if spec_format == 'code_plus_nl':
    body += 'Description: ' + task['nl'] + '\n\n'
  body += (
    'Reference PyTorch module (this is the spec and the ground truth):\n```python\n'
    + task['reference_src'].rstrip()
    + '\n```\n'
  )
  return body + contract


def time_reference(src, test_shapes, family):
  """Run the reference on every test shape/scale; return max wall time (s)."""
  ns = {'__name__': 'ref'}
  exec(compile(src, 'ref.py', 'exec'), ns)
  worst = 0.0
  for sh in test_shapes or [None]:
    for scale in (1.0, 0.01) if family == 'procedural' else (1.0,):
      if sh is not None:
        ns['SHAPE'], ns['SCALE'] = tuple(sh), scale
      torch.manual_seed(0)
      inputs = ns['get_inputs']()
      init = ns['get_init_inputs']()
      if (
        isinstance(init, list)
        and len(init) == 2
        and isinstance(init[0], list)
        and isinstance(init[1], dict)
      ):
        args, kwargs = init
      else:
        args, kwargs = init, {}
      m = ns['Model'](*args, **kwargs).eval()
      t0 = time.time()
      with torch.no_grad():
        out = m(*inputs)
      worst = max(worst, time.time() - t0)
      assert torch.is_tensor(out)
      if out.is_floating_point():
        assert torch.isfinite(out).all(), (
          f'non-finite reference output for {sh} {scale}'
        )
  return worst


def build_p_task(chain, shape, tid, spec_format, also_fp16, rng):
  src, info = L.emit_reference(chain, shape)
  names = [c['name'] for c in chain]
  atol, rtol = tolerance_for(names)
  task = dict(
    id=tid,
    family='procedural',
    subfamily=f'{len(chain)}ops',
    ops=names,
    op_cats=[L.OPS[n]['cat'] for n in names],
    chain=chain,
    shapes=[list(shape)],
    dtype='float32',
    also_fp16=also_fp16,
    spec_format=spec_format,
    reference_src=src,
    test_shapes=test_shapes_for(shape, rng),
    atol=atol,
    rtol=rtol,
    nl=info['nl'],
    out_shape=list(info['out_shape']),
    out_dtype=info['out_dtype'],
    init_sig=info['init_sig'],
    forward_sig=info['forward_sig'],
    extra_inputs=[
      dict(e, shape=list(e['shape'])) for e in info['extra_inputs']
    ],
  )
  task['spec'] = render_spec_p(task, info, spec_format)
  return task


def gen_family_p(rng):
  tasks, seen, rejected = [], set(), []
  fmts = [f for f, n in P_FORMATS.items() for _ in range(n)]
  rng.shuffle(fmts)
  i = 0
  for n_ops, count in P_COUNTS.items():
    made = 0
    while made < count:
      chain = L.sample_chain(rng, n_ops)
      if chain is None:
        continue
      L.sample_init_values(chain, rng)
      shape = rng.choice(L.SHAPES)
      key = json.dumps([chain, list(shape)], sort_keys=True)
      if key in seen:
        continue
      seen.add(key)
      tid = f'P-{i + 1:04d}'
      t = build_p_task(chain, shape, tid, fmts[i], rng.random() < 0.2, rng)
      try:
        t['ref_time_s'] = round(
          time_reference(t['reference_src'], t['test_shapes'], 'procedural'), 4
        )
      except AssertionError as e:  # non-finite output (e.g. exp after cumsum overflows fp32): resample
        rejected.append((chain_names(chain), str(e)[:80]))
        continue
      i += 1
      tasks.append(t)
      made += 1
  print(
    f'P chains rejected for non-finite reference output: {len(rejected)} {rejected[:5]}'
  )
  return tasks


def chain_names(chain):
  return [c['name'] for c in chain]


def gen_family_k(rng):
  p = DATA / 'kernelbook_selected.jsonl'
  if not p.exists():
    return []
  rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  fmts = [f for f, n in K_FORMATS.items() for _ in range(n)]
  rng.shuffle(fmts)
  tasks = []
  for i, r in enumerate(rows, 1):
    src = (
      r['python_code'].rstrip()
      + f'\n\n\nModel = {r["entry_point"]}   # the harness uses `Model`\n'
    )
    atol, rtol = tolerance_for([], k_src=r['python_code'])
    nl = (
      f'Human-written PyTorch module `{r["entry_point"]}` from GitHub repo {r["repo_name"]}'
      f' (torch attributes used in the module body: {", ".join(r["ops"]) if r["ops"] else "none"}); '
      f'inputs of shape {r["in_shapes"]}, output of shape {r["out_shape"]} ({r["out_dtype"]}).'
    )
    t = dict(
      id=f'K-{i:04d}',
      family='kernelbook',
      subfamily='human',
      ops=r['ops'],
      op_cats=[],
      shapes=r['in_shapes'],
      dtype=r['out_dtype'],
      also_fp16=False,
      spec_format=fmts[i - 1],
      reference_src=src,
      test_shapes=None,
      atol=atol,
      rtol=rtol,
      nl=nl,
      out_shape=r['out_shape'],
      out_dtype=r['out_dtype'],
      entry_point=r['entry_point'],
      repo_name=r['repo_name'],
      sha=r['sha'],
      licenses=r['licenses'],
      kb_uuid=r['uuid'],
      n_params=r['n_params'],
      ref_time_s=r['ref_time_s'],
      inplace_ref=False,
    )
    t['spec'] = render_spec_k(t, t['spec_format'])
    tasks.append(t)
  return tasks


def gen_mutants(p_tasks, rng):
  """200 mutants over distinct P tasks, balanced over mutation kinds where applicable; a mutant whose output differs
  from the original by less than fp32 resolution on every test shape/scale is a generation defect and is resampled."""
  muts, used = [], set()
  kinds_cycle = list(L.MUTATION_KINDS)
  rejected_identical = 0
  attempts = 0
  while len(muts) < N_MUTANTS and attempts < 20000:
    attempts += 1
    kind = kinds_cycle[len(muts) % len(kinds_cycle)]
    cands = [
      t
      for t in p_tasks
      if t['id'] not in used and kind in L.applicable_mutations(t['chain'])
    ]
    if not cands:
      kinds_cycle = [k for k in kinds_cycle if k != kind] or list(
        L.MUTATION_KINDS
      )
      continue
    t = rng.choice(cands)
    mut = L.pick_mutation(t['chain'], rng, kind)
    msrc, _ = L.emit_reference(t['chain'], tuple(t['shapes'][0]), mut)
    if mutant_identical(t, msrc):
      rejected_identical += 1
      used.add(t['id'])  # try a different task for this kind
      continue
    used.add(t['id'])
    muts.append(
      dict(
        id=f'M-{len(muts) + 1:04d}',
        task_id=t['id'],
        family='procedural',
        ops=t['ops'],
        mutation=mut,
        mutation_kind=kind,
        reference_src=t['reference_src'],
        mutant_src=msrc,
        test_shapes=t['test_shapes'],
        atol=t['atol'],
        rtol=t['rtol'],
      )
    )
  return muts, rejected_identical


def mutant_identical(task, msrc):
  ref_ns, mut_ns = {'__name__': 'a'}, {'__name__': 'b'}
  exec(compile(task['reference_src'], 'ref.py', 'exec'), ref_ns)
  exec(compile(msrc, 'mut.py', 'exec'), mut_ns)
  for sh in task['test_shapes']:
    for scale in (1.0, 0.01):
      ref_ns['SHAPE'] = mut_ns['SHAPE'] = tuple(sh)
      ref_ns['SCALE'] = mut_ns['SCALE'] = scale
      torch.manual_seed(0)
      inputs = ref_ns['get_inputs']()
      init = ref_ns['get_init_inputs']()
      a = ref_ns['Model'](*init)(*[t.clone() for t in inputs])
      b = mut_ns['Model'](*init)(*[t.clone() for t in inputs])
      if a.shape != b.shape or a.dtype != b.dtype:
        return False
      a64, b64 = a.to(torch.float64), b.to(torch.float64)
      # "identical" = below fp32 resolution: no differential test in fp32 can observe it (e.g. gelu(x) == x for x > 5)
      if (a64 - b64).abs().max().item() > 1e-6 * (1.0 + a64.abs().max().item()):
        return False
  return True


PROMPT_HEAD = """You are a GPU kernel engineer. Below are {N} independent tasks. Environment: Python {pyver}, torch {tver},
triton {trver}, CPU interpreter mode (TRITON_INTERPRET=1). float32 unless stated.

For EACH task, write a single Python file that defines one or more @triton.jit kernels and a class `ModelNew`
(same constructor signature as `Model`) whose `forward` returns the same result as `Model.forward` for any
valid input shape, computing the result with your Triton kernel(s). In `forward` you may allocate outputs,
compute grids, and call `.contiguous()`, but you must NOT compute the result with torch operations.
Handle arbitrary sizes with masking; do not assume powers of two.

Rules: answer directly. Do NOT use tools, run commands, or create/edit files. Code only.
Format, strictly: for each task, a line `### <task_id>` followed by exactly one ```python fenced block.

"""


def write_prompts(tasks, rng):
  order = list(tasks)
  rng.shuffle(order)
  pdir = DATA / 'prompts'
  pdir.mkdir(parents=True, exist_ok=True)
  for old in pdir.glob('batch_*.md'):
    old.unlink()
  batches = {}
  for b in range(0, len(order), BATCH):
    name = f'batch_{b // BATCH:03d}'
    chunk = order[b : b + BATCH]
    txt = PROMPT_HEAD.format(N=len(chunk), pyver=PYVER, tver=TVER, trver=TRVER)
    txt += '\n'.join(f'### {t["id"]}\n{t["spec"]}\n' for t in chunk)
    (pdir / f'{name}.md').write_text(txt)
    batches[name] = [t['id'] for t in chunk]
    for t in chunk:
      t['gen_batch'] = name
  (DATA / 'batches.json').write_text(json.dumps(batches, indent=1))
  return batches


def phase0_tasks(rng):
  """Dedicated single-op tasks for the 10 hand-written kernels (Phase 0.2) and the anti-cheat check (Phase 0.3)."""
  spec = [
    ('relu', {}),
    ('add', {'bcast': 'last'}),
    ('softmax', {}),
    ('rmsnorm', {'eps': 1e-5}),
    ('matmul', {'N': 64}),
    ('sum', {}),
    ('cumsum', {}),
    ('layernorm', {'affine': True, 'eps': 1e-5}),
    ('transpose', {}),
    ('where', {}),
  ]
  out = []
  for i, (name, params) in enumerate(spec, 1):
    chain = [{'name': name, 'params': params}]
    t = build_p_task(
      chain, (16, 256), f'H-{i:02d}-{name}', 'code_only', False, rng
    )
    out.append(t)
  return out


def main():
  DATA.mkdir(exist_ok=True)
  (HERE / 'phase0').mkdir(exist_ok=True)
  rng = random.Random(SEED)
  p_tasks = gen_family_p(rng)
  k_tasks = gen_family_k(random.Random(SEED + 1))
  tasks = p_tasks + k_tasks
  print(f'P={len(p_tasks)} K={len(k_tasks)}')
  # all references must run on CPU < 2 s on every test shape
  worst_all = 0.0
  for t in tasks:
    if t['family'] == 'kernelbook':
      t['ref_time_s'] = round(
        time_reference(t['reference_src'], t['test_shapes'], t['family']), 4
      )
    worst_all = max(worst_all, t['ref_time_s'])
    assert t['ref_time_s'] < 2.0, (t['id'], t['ref_time_s'])
  print(f'reference max time {worst_all:.3f}s')
  write_prompts(tasks, random.Random(SEED + 2))
  (DATA / 'corpus.jsonl').write_text(
    '\n'.join(json.dumps(t) for t in tasks) + '\n'
  )
  muts, rej = gen_mutants(p_tasks, random.Random(SEED + 3))
  (DATA / 'mutants.jsonl').write_text(
    '\n'.join(json.dumps(m) for m in muts) + '\n'
  )
  kinds = {}
  for m in muts:
    kinds[m['mutation_kind']] = kinds.get(m['mutation_kind'], 0) + 1
  print(
    f'mutants={len(muts)} by kind {kinds}; rejected as output-identical: {rej}'
  )
  h = phase0_tasks(random.Random(SEED + 4))
  (HERE / 'phase0' / 'tasks.jsonl').write_text(
    '\n'.join(json.dumps(t) for t in h) + '\n'
  )
  stats = dict(
    P=len(p_tasks),
    K=len(k_tasks),
    mutants=len(muts),
    mutant_kinds=kinds,
    mutants_rejected_identical=rej,
    ref_max_time_s=round(worst_all, 4),
    versions=dict(python=PYVER, torch=TVER, triton=TRVER),
    p_formats={
      f: sum(1 for t in p_tasks if t['spec_format'] == f) for f in P_FORMATS
    },
    k_formats={
      f: sum(1 for t in k_tasks if t['spec_format'] == f) for f in K_FORMATS
    },
    p_by_ops={
      k: sum(1 for t in p_tasks if t['subfamily'] == f'{k}ops')
      for k in P_COUNTS
    },
  )
  (DATA / 'corpus_stats.json').write_text(json.dumps(stats, indent=1))
  print(json.dumps(stats, indent=1))


if __name__ == '__main__':
  main()
