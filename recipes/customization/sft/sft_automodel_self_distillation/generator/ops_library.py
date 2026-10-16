# Research script kept as run (formatting only); lint exemptions below.
# ruff: noqa: ANN201,ANN202,ARG001,ARG005,B007,D102,D103
"""Family P op library: ~30 primitive ops, chain sampler, PyTorch source emitter, NL phrase
renderer and mutation operator. No LLM involvement anywhere in this file (contamination rule).

Symbolic shapes. Every reference module exposes module-level `SHAPE` (the primary input shape)
and `SCALE` (magnitude of the primary input x only, default 1.0). The verification harness re-binds both before calling
`get_inputs()`, which is how "get_inputs() adapted to the shape" is implemented exactly.
Shapes are tracked symbolically as a batch prefix `*SHAPE[:-2]` plus a list of trailing-dim
expressions such as "SHAPE[-2]", "SHAPE[-1]" or a constant like "64" (matmul output width).
"""

import copy

SHAPES = [(4, 128), (16, 256), (64, 1024), (2, 8, 512), (1, 4096), (33, 257)]
NPOT_SHAPE = (33, 257)  # forces non-power-of-two masking
MATMUL_N = [32, 64, 128, 256, 512]


# ----------------------------------------------------------------------------------------------
# Symbolic shape state
# ----------------------------------------------------------------------------------------------
class State:
  def __init__(self):
    self.trailing = ['SHAPE[-2]', 'SHAPE[-1]']
    self.reduced = False  # a reduction has removed the last dim
    self.terminal = False  # argmax -> int64, nothing may follow

  def clone(self):
    return copy.deepcopy(self)

  def full_shape_expr(self):
    """Argument list for torch.randn(...) producing the current tensor shape."""
    if self.trailing == ['SHAPE[-2]', 'SHAPE[-1]']:
      return '*SHAPE'
    return '*SHAPE[:-2], ' + ', '.join(self.trailing)

  def last_expr(self):
    return self.trailing[-1]

  def concrete(self, shape):
    return tuple(shape[:-2]) + tuple(
      int(eval(e, {'SHAPE': shape})) for e in self.trailing
    )


def concrete_dim(expr, shape):
  return int(eval(expr, {'SHAPE': shape}))


# ----------------------------------------------------------------------------------------------
# Op definitions
# Each op: cat, init (constructor params: (name, choices)), extra (inputs: (name, kind)),
#          ok(state) -> bool, step(state) -> None (mutates), code(v) -> [lines],
#          phrase(v, shape) -> str.   `v` maps param/extra base names to their resolved names/values.
# ----------------------------------------------------------------------------------------------
OPS = {}


def op(
  name,
  cat,
  init=(),
  extra=(),
  ok=None,
  step=None,
  code=None,
  phrase=None,
  sample=None,
):
  OPS[name] = dict(
    name=name,
    cat=cat,
    init=list(init),
    extra=list(extra),
    ok=ok or (lambda s: not s.terminal),
    step=step or (lambda s, p: None),
    code=code,
    phrase=phrase,
    sample=sample or (lambda rng: {}),
  )


def not_reduced(s):
  return not s.terminal and not s.reduced


def two_trailing(s):
  return not s.terminal and len(s.trailing) == 2


def only_unary_after_reduction(s):
  return not s.terminal


# ---- Elementwise unary --------------------------------------------------------------------
op(
  'relu',
  'unary',
  code=lambda v: ['y = torch.relu(y)'],
  phrase=lambda v, sh: 'ReLU',
)
op(
  'leaky_relu',
  'unary',
  init=[('alpha', [0.01, 0.1, 0.2])],
  code=lambda v: [f'y = F.leaky_relu(y, negative_slope=self.{v["alpha"]})'],
  phrase=lambda v, sh: (
    'leaky ReLU (negative slope 0.1)'
    if v.get('_inline')
    else f'leaky ReLU (negative slope = constructor arg `{v["alpha"]}` = {v["alpha_val"]})'
  ),
)
op(
  'gelu',
  'unary',
  code=lambda v: ["y = F.gelu(y, approximate='tanh')"],
  phrase=lambda v, sh: 'GELU (tanh approximation)',
)
op(
  'silu',
  'unary',
  code=lambda v: ['y = F.silu(y)'],
  phrase=lambda v, sh: 'SiLU (x * sigmoid(x))',
)
op(
  'sigmoid',
  'unary',
  code=lambda v: ['y = torch.sigmoid(y)'],
  phrase=lambda v, sh: 'sigmoid',
)
op(
  'tanh',
  'unary',
  code=lambda v: ['y = torch.tanh(y)'],
  phrase=lambda v, sh: 'tanh',
)
op(
  'exp',
  'unary',
  code=lambda v: ['y = torch.exp(y)'],
  phrase=lambda v, sh: 'elementwise exp',
)
op(
  'abs',
  'unary',
  code=lambda v: ['y = torch.abs(y)'],
  phrase=lambda v, sh: 'elementwise absolute value',
)
op(
  'square',
  'unary',
  code=lambda v: ['y = y * y'],
  phrase=lambda v, sh: 'elementwise square',
)
op(
  'clamp',
  'unary',
  init=[('lo', None), ('hi', None)],
  sample=lambda rng: dict(
    bounds=rng.choice([(-1.0, 1.0), (0.0, 6.0), (-0.5, 0.5)])
  ),
  code=lambda v: [f'y = torch.clamp(y, self.{v["lo"]}, self.{v["hi"]})'],
  phrase=lambda v, sh: (
    f'clamp to [`{v["lo"]}`, `{v["hi"]}`] = [{v["lo_val"]}, {v["hi_val"]}] (constructor args)'
  ),
)
op(
  'scale_shift',
  'unary',
  init=[('a', None), ('b', None)],
  sample=lambda rng: dict(
    ab=rng.choice([(2.0, 1.0), (0.5, -1.0), (1.5, 0.25)])
  ),
  code=lambda v: [f'y = self.{v["a"]} * y + self.{v["b"]}'],
  phrase=lambda v, sh: (
    f'scale-and-shift `{v["a"]}` * y + `{v["b"]}` with {v["a"]} = {v["a_val"]}, {v["b"]} = {v["b_val"]} (constructor args)'
  ),
)


# ---- Elementwise binary (second input broadcast over the last dim or full shape) --------------
def _bin(name, sym, verb, kind_full='full', kind_last='last'):
  def code(v):
    return [f'y = {sym.format(y="y", b=v["b"])}']

  def phrase(v, sh):
    return f'{verb} the extra input `{v["b"]}` (shape {v["b_shape"]}, {v["b_bcast"]})'

  def extra(p):
    return [('b', kind_full if p['bcast'] == 'full' else kind_last)]

  op(
    name,
    'binary',
    extra=[('b', None)],
    ok=lambda s: not s.terminal,
    sample=lambda rng: dict(bcast=rng.choice(['full', 'last'])),
    code=code,
    phrase=phrase,
  )
  OPS[name]['extra_fn'] = extra


_bin('add', '{y} + {b}', 'add')
_bin('sub', '{y} - {b}', 'subtract')
_bin('mul', '{y} * {b}', 'multiply by')
_bin(
  'div', '{y} / {b}', 'divide by', kind_full='pos_full', kind_last='pos_last'
)
_bin('maximum', 'torch.maximum({y}, {b})', 'elementwise maximum with')
_bin('minimum', 'torch.minimum({y}, {b})', 'elementwise minimum with')
op(
  'where',
  'binary',
  extra=[('m', 'full')],
  ok=lambda s: not s.terminal,
  code=lambda v: [f'y = torch.where({v["m"]} > 0, y, torch.zeros_like(y))'],
  phrase=lambda v, sh: (
    f'masked select: keep y where the extra input `{v["m"]}` (same shape) is > 0, else 0'
  ),
)


# ---- Row-wise reductions over the last dim (keepdim=False) ------------------------------------
def _red_step(s, p):
  s.trailing = s.trailing[:-1]
  s.reduced = True


def _red(name, code_line, phrase_txt, terminal=False):
  def step(s, p):
    _red_step(s, p)
    if terminal:
      s.terminal = True

  op(
    name,
    'reduction',
    ok=two_trailing,
    step=step,
    code=lambda v: [code_line.format(dim=v.get('dim', '-1'))],
    phrase=lambda v, sh: phrase_txt,
  )


_red(
  'sum', 'y = y.sum(dim={dim})', 'sum over the last dimension (keepdim=False)'
)
_red(
  'mean',
  'y = y.mean(dim={dim})',
  'mean over the last dimension (keepdim=False)',
)
_red(
  'max',
  'y = torch.max(y, dim={dim}).values',
  'max over the last dimension (keepdim=False)',
)
_red(
  'min',
  'y = torch.min(y, dim={dim}).values',
  'min over the last dimension (keepdim=False)',
)
_red(
  'argmax',
  'y = torch.argmax(y, dim={dim})',
  'argmax over the last dimension (int64 indices, keepdim=False)',
  terminal=True,
)
_red(
  'l2norm',
  'y = torch.sqrt((y * y).sum(dim={dim}))',
  'L2 norm over the last dimension (keepdim=False)',
)
_red(
  'logsumexp',
  'y = torch.logsumexp(y, dim={dim})',
  'logsumexp over the last dimension (keepdim=False)',
)

# ---- Row-wise normalizations ------------------------------------------------------------------
op(
  'softmax',
  'normalization',
  ok=not_reduced,
  code=lambda v: [f'y = torch.softmax(y, dim={v.get("dim", "-1")})'],
  phrase=lambda v, sh: 'softmax over the last dimension',
)
op(
  'log_softmax',
  'normalization',
  ok=not_reduced,
  code=lambda v: [f'y = torch.log_softmax(y, dim={v.get("dim", "-1")})'],
  phrase=lambda v, sh: 'log-softmax over the last dimension',
)


def _ln_code(v):
  eps = f'self.{v["eps"]}' + (
    f' * {v["eps_mult"]}' if v.get('eps_mult') else ''
  )
  if v['affine']:
    return [
      f'y = F.layer_norm(y, (y.shape[-1],), weight={v["gamma"]}, bias={v["beta"]}, eps={eps})'
    ]
  return [f'y = F.layer_norm(y, (y.shape[-1],), eps={eps})']


def _ln_phrase(v, sh):
  s = f'LayerNorm over the last dimension (biased variance, eps = constructor arg `{v["eps"]}` = {v["eps_val"]}'
  if v['affine']:
    s += f'; affine weight `{v["gamma"]}` and bias `{v["beta"]}` of shape {v["gamma_shape"]} are extra inputs'
  return s + ')'


op(
  'layernorm',
  'normalization',
  init=[('eps', [1e-5, 1e-6, 1e-4])],
  extra=[('gamma', 'affine'), ('beta', 'affine')],
  ok=not_reduced,
  sample=lambda rng: dict(affine=rng.random() < 0.6),
  code=_ln_code,
  phrase=_ln_phrase,
)
OPS['layernorm']['extra_fn'] = lambda p: (
  [('gamma', 'affine'), ('beta', 'affine')] if p['affine'] else []
)


def _rms_code(v):
  eps = f'self.{v["eps"]}' + (
    f' * {v["eps_mult"]}' if v.get('eps_mult') else ''
  )
  return [
    f'y = y / torch.sqrt((y * y).mean(dim={v.get("dim", "-1")}, keepdim=True) + {eps})'
  ]


op(
  'rmsnorm',
  'normalization',
  init=[('eps', [1e-5, 1e-6, 1e-4])],
  ok=not_reduced,
  code=_rms_code,
  phrase=lambda v, sh: (
    f'RMSNorm over the last dimension: y / sqrt(mean(y^2) + eps), eps = constructor arg `{v["eps"]}` = {v["eps_val"]} (no affine)'
  ),
)


# ---- Structural --------------------------------------------------------------------------------
def _tr_step(s, p):
  s.trailing = [s.trailing[1], s.trailing[0]]


op(
  'transpose',
  'structural',
  ok=two_trailing,
  step=_tr_step,
  code=lambda v: ['y = y.transpose(-2, -1)'],
  phrase=lambda v, sh: (
    'transpose the last two dimensions (output is a fresh contiguous tensor of the transposed shape)'
  ),
)
op(
  'cumsum',
  'structural',
  ok=not_reduced,
  code=lambda v: [f'y = torch.cumsum(y, dim={v.get("dim", "-1")})'],
  phrase=lambda v, sh: 'inclusive cumulative sum along the last dimension',
)
op(
  'tril_fill',
  'structural',
  ok=two_trailing,
  code=lambda v: [
    f'mask = torch.tril(torch.ones(y.shape[-2], y.shape[-1], dtype=torch.bool, device=y.device), diagonal={v.get("diagonal", -1)})',
    'y = y.masked_fill(mask, 0.0)',
  ],
  phrase=lambda v, sh: (
    'masked fill: set the strictly-lower-triangular part of the last two dims (col < row) to 0'
  ),
)


# ---- Matmul ------------------------------------------------------------------------------------
def _mm_ok(s):
  return two_trailing(s) and (
    s.trailing[-1] == 'SHAPE[-1]' or s.trailing[-1].isdigit()
  )


def _mm_step(s, p):
  s.trailing = [s.trailing[0], str(p['N'])]


op(
  'matmul',
  'matmul',
  extra=[('w', 'weight')],
  ok=_mm_ok,
  step=_mm_step,
  sample=lambda rng: dict(N=rng.choice(MATMUL_N)),
  code=lambda v: [f'y = y @ {v["w"]}'],
  phrase=lambda v, sh: (
    f'matrix multiply y @ `{v["w"]}` (extra input, shape {v["w_shape"]}{", batched over the leading dims" if len(v["w_shape"]) > 2 else ""})'
  ),
)

_ACT = {
  'relu': 'F.relu({e})',
  'gelu': "F.gelu({e}, approximate='tanh')",
  'silu': 'F.silu({e})',
}
op(
  'matmul_bias_act',
  'matmul',
  extra=[('w', 'weight'), ('bias', 'bias')],
  ok=_mm_ok,
  step=_mm_step,
  sample=lambda rng: dict(N=rng.choice(MATMUL_N), act=rng.choice(list(_ACT))),
  code=lambda v: [
    f'y = {_ACT[v["act"]].format(e="y @ " + v["w"] + " + " + v["bias"])}'
  ],
  phrase=lambda v, sh: (
    f'fused matmul + bias + {v["act"] if v["act"] != "gelu" else "GELU (tanh approx)"}: act(y @ `{v["w"]}` + `{v["bias"]}`) with extra inputs `{v["w"]}` of shape {v["w_shape"]} and `{v["bias"]}` of shape {v["bias_shape"]}'
  ),
)

UNARY_OPS = [n for n, o in OPS.items() if o['cat'] == 'unary']
CATEGORIES = sorted({o['cat'] for o in OPS.values()})

# ----------------------------------------------------------------------------------------------
# Chain sampling
# ----------------------------------------------------------------------------------------------
NONNEG_PRODUCERS = {
  'softmax',
  'sigmoid',
  'exp',
  'square',
  'abs',
  'relu',
  'l2norm',
}


def sample_chain(rng, n_ops):
  """Return a list of {'name', 'params'} dicts forming a shape-compatible chain, or None."""
  for _ in range(200):
    st = State()
    chain = []
    good = True
    for i in range(n_ops):
      if st.reduced:
        cands = [n for n in UNARY_OPS if OPS[n]['ok'](st)]
      else:
        cands = [n for n in OPS if OPS[n]['ok'](st)]
      if chain:
        prev = chain[-1]['name']
        cands = [c for c in cands if c != prev]
        if prev in NONNEG_PRODUCERS:
          cands = [c for c in cands if c not in ('relu', 'abs')]
        # a terminal reduction must be last
      if i < n_ops - 1:
        cands = [c for c in cands if c != 'argmax']
      if not cands:
        good = False
        break
      # category-balanced draw: pick a category first so matmul/reductions aren't swamped by 11 unary ops
      cats = sorted({OPS[c]['cat'] for c in cands})
      cat = rng.choice(cats)
      name = rng.choice([c for c in cands if OPS[c]['cat'] == cat])
      params = OPS[name]['sample'](rng)
      OPS[name]['step'](st, params)
      chain.append({'name': name, 'params': params})
    if good:
      return chain
  return None


# ----------------------------------------------------------------------------------------------
# Source emission
# ----------------------------------------------------------------------------------------------
def _fmt(x):
  if isinstance(x, float):
    return repr(x)
  return str(x)


def resolve(chain):
  """Assign unique constructor-arg names and extra-input names; return per-op value dicts and state trace."""
  used_init, used_extra = {}, {}
  init_list = []  # (attr_name, value)
  extra_list = []  # (var_name, kind, state_before_op)
  values = []
  st = State()
  states = [st.clone()]
  for inst in chain:
    o = OPS[inst['name']]
    p = inst['params']
    v = dict(p)
    # constructor params
    for base, choices in o['init']:
      if p.get('_inline'):
        continue
      n = used_init.get(base, 0)
      used_init[base] = n + 1
      attr = base if n == 0 else f'{base}{n + 1}'
      if base in ('lo', 'hi'):
        val = p['bounds'][0 if base == 'lo' else 1]
      elif base in ('a', 'b') and inst['name'] == 'scale_shift':
        val = p['ab'][0 if base == 'a' else 1]
      else:
        val = p.get(base) if p.get(base) is not None else None
        if val is None:
          raise ValueError(f'missing init value {base} for {inst}')
      v[base] = attr
      v[base + '_val'] = val
      init_list.append((attr, val))
    # extra inputs
    extras = o['extra_fn'](p) if 'extra_fn' in o else o['extra']
    for base, kind in extras:
      n = used_extra.get(base, 0)
      used_extra[base] = n + 1
      var = base if n == 0 else f'{base}{n + 1}'
      v[base] = var
      extra_list.append((var, kind, st.clone(), p))
      v[base + '_kind'] = kind
    values.append(v)
    o['step'](st, p)
    states.append(st.clone())
  return init_list, extra_list, values, states


def _extra_shape_expr(kind, st, p):
  if kind in ('full', 'pos_full'):
    return st.full_shape_expr()
  if kind in ('last', 'pos_last', 'affine'):
    return st.last_expr()
  if kind == 'weight':
    return f'*SHAPE[:-2], {st.last_expr()}, {p["N"]}'
  if kind == 'bias':
    return str(p['N'])
  raise ValueError(kind)


def _extra_gen(kind, expr):
  # SCALE shrinks activation-like inputs (x and elementwise operands b / m) but not parameter-like inputs
  # (matmul weights, bias, layernorm affine): "small activations, normal weights", which makes eps terms observable.
  if kind.startswith('pos_'):
    return f'torch.rand({expr}) + 0.5'
  if kind in ('weight', 'bias', 'affine'):
    return f'torch.randn({expr})'
  return f'torch.randn({expr}) * SCALE'


def _extra_concrete_shape(kind, st, p, shape):
  if kind in ('full', 'pos_full'):
    return st.concrete(shape)
  if kind in ('last', 'pos_last', 'affine'):
    return (concrete_dim(st.last_expr(), shape),)
  if kind == 'weight':
    return tuple(shape[:-2]) + (concrete_dim(st.last_expr(), shape), p['N'])
  if kind == 'bias':
    return (p['N'],)
  raise ValueError(kind)


def sample_init_values(chain, rng):
  """Fill constructor-param values into params (eps, alpha) in place."""
  for inst in chain:
    o = OPS[inst['name']]
    for base, choices in o['init']:
      if choices is not None and base not in inst['params']:
        inst['params'][base] = rng.choice(choices)


def emit_reference(chain, shape, mutation=None):
  """Return (source, info). `mutation` is applied to a deep copy of the chain (see mutate())."""
  chain = copy.deepcopy(chain)
  base_init, base_extra, _, _ = resolve(chain)
  if mutation:
    chain = apply_mutation(chain, mutation)
  init_list, extra_list, values, states = resolve(chain)
  if mutation:
    # a mutant keeps the ORIGINAL constructor / forward signatures and get_inputs(); only the body differs
    init_list, extra_list = base_init, base_extra
  # concrete shapes for NL and for spec text
  ex_by_var = {var: (kind, st, p) for var, kind, st, p in extra_list}
  for v in values:
    for k in list(v):
      if k.endswith('_kind'):
        base = k[:-5]
        var = v[base]
        kind, st, p = ex_by_var[var]
        cshape = _extra_concrete_shape(kind, st, p, shape)
        v[base + '_shape'] = cshape
        v[base + '_bcast'] = (
          'broadcast over the last dimension'
          if (kind.endswith('last') or kind == 'affine')
          else 'same shape as y'
        )
  # ---- source
  args = ['x'] + [var for var, _, _, _ in extra_list]
  init_sig = ', '.join(
    ['self'] + [f'{a}: float = {_fmt(val)}' for a, val in init_list]
  )
  lines = [
    'import torch',
    'import torch.nn as nn',
    'import torch.nn.functional as F',
    '',
    f'SHAPE = {tuple(shape)}   # primary input shape; the harness re-binds SHAPE to test other shapes',
    'SCALE = 1.0            # magnitude of activation-like inputs (not weights/affine); the harness also runs SCALE = 0.01',
    '',
    '',
    'class Model(nn.Module):',
    f'    def __init__({init_sig}):',
    '        super().__init__()',
  ]
  for a, val in init_list:
    lines.append(f'        self.{a} = {a}')
  lines += ['', f'    def forward(self, {", ".join(args)}):', '        y = x']
  for inst, v in zip(chain, values):
    for cl in OPS[inst['name']]['code'](v):
      lines.append('        ' + cl)
  lines += [
    '        return y',
    '',
    '',
    'def get_inputs():',
    f'    x = torch.randn({State().full_shape_expr()}) * SCALE',
  ]
  for var, kind, st, p in extra_list:
    lines.append(
      f'    {var} = {_extra_gen(kind, _extra_shape_expr(kind, st, p))}'
    )
  lines.append(f'    return [{", ".join(args)}]')
  lines += [
    '',
    '',
    'def get_init_inputs():',
    f'    return [{", ".join(_fmt(val) for _, val in init_list)}]',
    '',
  ]
  src = '\n'.join(lines)
  # ---- NL sentence
  phrases = [
    OPS[inst['name']]['phrase'](v, shape) for inst, v in zip(chain, values)
  ]
  nl = (
    'Fused kernel: '
    + ', then '.join(phrases)
    + f', on a {tuple(shape)} float32 input tensor `x`.'
  )
  out_shape = states[-1].concrete(shape)
  info = dict(
    init_sig=f'Model({", ".join(f"{a}: float = {_fmt(val)}" for a, val in init_list)})',
    forward_sig=f'forward(self, {", ".join(args)})',
    nl=nl,
    out_shape=out_shape,
    out_dtype='int64' if states[-1].terminal else 'float32',
    extra_inputs=[
      dict(name=var, kind=kind, shape=_extra_concrete_shape(kind, st, p, shape))
      for var, kind, st, p in extra_list
    ],
    init_values=[val for _, val in init_list],
  )
  return src, info


# ----------------------------------------------------------------------------------------------
# Mutations (Phase 0 mutation check)
# ----------------------------------------------------------------------------------------------
MUTATION_KINDS = [
  'relu_to_leaky',
  'dim0',
  'eps_x10',
  'drop_last',
  'mask_off_by_one',
]


def applicable_mutations(chain):
  kinds = []
  names = [c['name'] for c in chain]
  if 'relu' in names:
    kinds.append('relu_to_leaky')
  if any(
    n
    in (
      'softmax',
      'log_softmax',
      'rmsnorm',
      'cumsum',
      'sum',
      'mean',
      'max',
      'min',
      'argmax',
      'l2norm',
      'logsumexp',
    )
    for n in names
  ):
    kinds.append('dim0')
  if any(n in ('layernorm', 'rmsnorm') for n in names):
    kinds.append('eps_x10')
  kinds.append('drop_last')
  if 'tril_fill' in names:
    kinds.append('mask_off_by_one')
  return kinds


def apply_mutation(chain, mutation):
  kind, idx = mutation['kind'], mutation.get('index')
  chain = copy.deepcopy(chain)
  if kind == 'relu_to_leaky':
    chain[idx] = {'name': 'leaky_relu', 'params': {'alpha': 0.1}}
    # leaky_relu has a constructor arg; since the mutant is wrapped as ModelNew with the ORIGINAL
    # constructor signature, we inline the slope instead.
    chain[idx]['params']['_inline'] = True
  elif kind == 'dim0':
    chain[idx]['params']['dim'] = '0'
  elif kind == 'eps_x10':
    chain[idx]['params']['eps_mult'] = 10
  elif kind == 'drop_last':
    chain = chain[:-1]
  elif kind == 'mask_off_by_one':
    chain[idx]['params']['diagonal'] = 0
  else:
    raise ValueError(kind)
  return chain


def pick_mutation(chain, rng, kind=None):
  kinds = applicable_mutations(chain)
  if kind is None:
    kind = rng.choice(kinds)
  names = [c['name'] for c in chain]
  if kind == 'relu_to_leaky':
    idx = rng.choice([i for i, n in enumerate(names) if n == 'relu'])
  elif kind == 'dim0':
    idx = rng.choice([
      i
      for i, n in enumerate(names)
      if n
      in (
        'softmax',
        'log_softmax',
        'rmsnorm',
        'cumsum',
        'sum',
        'mean',
        'max',
        'min',
        'argmax',
        'l2norm',
        'logsumexp',
      )
    ])
  elif kind == 'eps_x10':
    idx = rng.choice([
      i for i, n in enumerate(names) if n in ('layernorm', 'rmsnorm')
    ])
  elif kind == 'mask_off_by_one':
    idx = rng.choice([i for i, n in enumerate(names) if n == 'tril_fill'])
  else:
    idx = None
  return {'kind': kind, 'index': idx}


# leaky_relu inline variant used by the relu_to_leaky mutation (no constructor arg)
_orig_leaky_code = OPS['leaky_relu']['code']
OPS['leaky_relu']['code'] = lambda v: (
  ['y = F.leaky_relu(y, negative_slope=0.1)']
  if v.get('_inline')
  else _orig_leaky_code(v)
)
_orig_leaky_init = OPS['leaky_relu']['init']
