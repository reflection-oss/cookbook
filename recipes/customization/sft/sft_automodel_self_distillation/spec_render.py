# Extracted verbatim from the corpus generator (formatting only); lint exemptions below.
# ruff: noqa: ANN201,D103
"""Task text (spec) rendering of the self-distillation corpus, extracted verbatim from the corpus generator; versions pinned to the
values embedded in the released specs."""

PYVER, TVER, TRVER = '3.12.14', '2.14.0+cpu', '3.8.0'

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
