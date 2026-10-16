# Research script kept as run (formatting only); lint exemptions below.
# ruff: noqa: ANN201,ANN202,ARG001,ARG002,B023,BLE001,D102,D103,E402,E741,N806,N812,PGH003,PLC0415,PLR1702,PLW1510,PLW1514,SIM102
"""GPU grader: sandboxed differential verification of model-written Triton candidates.
The candidate is compiled on CUDA with TRITON_F32_DEFAULT=ieee, the reference runs on the CPU.

Subcommands
  worker     runs INSIDE the sandbox (nice, timeout, ulimit -v, TRITON_INTERPRET=1). Verifies one candidate
             against one task and writes a JSON result to --out.
  run        orchestrator: corpus + candidates dir -> results jsonl (resumable, N sandboxed workers).
  selfcheck  Model vs Model (trusted mode: Triton/anti-cheat checks disabled).
  mutation   mutants wrapped as ModelNew (trusted mode) -> every task must be non-PASS.
  single     verify one candidate file against one task id (full mode) and print the JSON.

Statuses: PASS | FAIL | RUNTIME_ERROR | SYNTAX_ERROR | FORBIDDEN_IMPORT | NO_TRITON_KERNEL | NO_KERNEL_LAUNCH |
          TORCH_FALLBACK | CONTRACT_VIOLATION | NO_OUTPUT | TIMEOUT

Sandbox (see `run`): fresh subprocess per candidate, `nice -n 10`, `timeout` (hard kill), `ulimit -v 8000000` (8 GB
virtual), cwd in a scratch dir, minimal environment, sockets disabled inside the worker (unprivileged `unshare -n`
is not permitted on this host), import allowlist enforced by AST scan before execution. The filesystem is NOT
isolated: the candidate runs with the invoking user's file access. Run the grader on a dedicated machine or in a
container that holds no credentials or data you care about.
"""

import argparse
import ast
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------
# 2.1 Static checks
# ------------------------------------------------------------------------------------------------
ALLOWED_IMPORT_ROOTS = {
  'torch',
  'triton',
  'math',
  'numpy',
  'typing',
}  # runbook list + typing (deviation, reported)
FORBIDDEN_BUILTINS = {
  '__import__',
  'exec',
  'eval',
  'open',
  'compile',
  'breakpoint',
  'globals',
}  # flagged only when CALLED

# 2.3 anti-cheat: explicit lists (reported in results/report.md)
ALLOWED_TORCH_CALLS = {
  'torch.empty',
  'torch.empty_like',
  'torch.zeros',
  'torch.zeros_like',
  'torch.ones',
  'torch.ones_like',
  'torch.full',
  'torch.full_like',
  'torch.empty_strided',
  'torch.no_grad',
  'torch.inference_mode',
  'torch.set_grad_enabled',
  'torch.is_tensor',
  'torch.finfo',
  'torch.iinfo',
  'torch.Size',
  'torch.device',
  'torch.get_default_dtype',
  'torch.promote_types',
  'torch.result_type',
  'torch.is_floating_point',
  'torch.cuda.is_available',
  'torch.cuda.current_device',
  'torch.cuda.synchronize',
  'torch.cuda.current_stream',
  'torch.cuda.device',
  'torch.numel',
  'torch.is_grad_enabled',
}
# Tensor methods that are allowed (allocation / metadata / views)
INT_METHODS = {
  'numel',
  'size',
  'stride',
  'dim',
  'data_ptr',
  'element_size',
  'is_contiguous',
  'storage_offset',
  'ndimension',
  'nelement',
  'is_floating_point',
  'get_device',
  'is_complex',
}
VIEW_METHODS = {
  'contiguous',
  'view',
  'reshape',
  'unsqueeze',
  'squeeze',
  'flatten',
  'to',
  'float',
  'half',
  'double',
  'clone',
  'detach',
  'type',
  'cpu',
  'cuda',
  'fill_',
  'zero_',
  'view_as',
  'new_empty',
  'new_zeros',
  'new_ones',
  'new_full',
  'requires_grad_',
  'as_strided',
  'unflatten',
  'type_as',
}
FORBIDDEN_ATTRS = {'T', 'mT', 'H', 'mH', 'real', 'imag'}
import re as _re

PARAM_ATTR_RE = _re.compile(
  r'^(weight|bias|running_mean|running_var|num_batches_tracked|.*_weight|.*_bias|weight_.*|bias_.*|'
  r'in_proj_.*|out_proj|data|param.*|alpha|beta|gamma|scale|shift|offset|mask|table|embedding.*|.*_emb.*|centers|'
  r'kernel|filters|w|b|u|v|g|h0|c0)$'
)
INT_ATTRS = {
  'shape',
  'dtype',
  'device',
  'ndim',
  'is_cuda',
  'layout',
  'requires_grad',
  'grad',
  'data',
}
PY_BUILTINS = {
  'range',
  'len',
  'int',
  'min',
  'max',
  'tuple',
  'list',
  'dict',
  'set',
  'isinstance',
  'enumerate',
  'zip',
  'print',
  'float',
  'bool',
  'sorted',
  'abs',
  'sum',
  'any',
  'all',
  'reversed',
  'round',
  'divmod',
  'pow',
  'str',
  'hasattr',
  'getattr',
  'setattr',
  'super',
  'type',
  'map',
  'filter',
  'iter',
  'next',
  'slice',
  'id',
  'repr',
  'callable',
  'issubclass',
  'ValueError',
  'RuntimeError',
  'TypeError',
  'AssertionError',
  'NotImplementedError',
  'Exception',
  'staticmethod',
  'classmethod',
  'property',
  'object',
}


def _dotted(node, aliases):
  """Return dotted name for Name/Attribute chain (aliases resolved) or None."""
  parts = []
  while isinstance(node, ast.Attribute):
    parts.append(node.attr)
    node = node.value
  if isinstance(node, ast.Name):
    root = aliases.get(node.id, node.id)
    return '.'.join([root] + parts[::-1])
  return None


def _is_jit_decorated(fn):
  for d in fn.decorator_list:
    target = d.func if isinstance(d, ast.Call) else d
    name = _dotted(target, {}) or ''
    if name in (
      'triton.jit',
      'jit',
      'triton.runtime.jit',
      'triton.autotune',
      'autotune',
      'triton.heuristics',
      'heuristics',
    ) or name.endswith('.jit'):
      return True
  return False


def static_checks(src, mode):
  """Return (status or None, message, info). mode: 'full' | 'trusted'."""
  info = {}
  try:
    tree = ast.parse(src)
  except SyntaxError as e:
    return 'SYNTAX_ERROR', f'SyntaxError: {e.msg} (line {e.lineno})', info
  # imports
  aliases = {}
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for a in node.names:
        root = a.name.split('.')[0]
        if root not in ALLOWED_IMPORT_ROOTS:
          return 'FORBIDDEN_IMPORT', f'import {a.name}', info
        aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else root
    elif isinstance(node, ast.ImportFrom):
      mod = node.module or ''
      root = mod.split('.')[0]
      if node.level or root not in ALLOWED_IMPORT_ROOTS:
        return (
          'FORBIDDEN_IMPORT',
          f'from {"." * node.level}{mod} import ...',
          info,
        )
      for a in node.names:
        aliases[a.asname or a.name] = f'{mod}.{a.name}'
    elif (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id in FORBIDDEN_BUILTINS
    ):
      return 'FORBIDDEN_IMPORT', f'call to builtin {node.func.id}()', info
  info['uses_typing'] = any(v.startswith('typing') for v in aliases.values())
  jit_fns = [
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and _is_jit_decorated(n)
  ]  # incl. class-level kernels
  info['n_jit_kernels'] = len(jit_fns)
  model_new = [
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelNew'
  ]
  if mode == 'full' and not jit_fns:
    return 'NO_TRITON_KERNEL', 'no function decorated with @triton.jit', info
  if mode == 'trusted' and not model_new:
    # self-check / mutation wrappers alias `ModelNew = Model`; no anti-cheat scan in trusted mode
    alias = [
      n
      for n in tree.body
      if isinstance(n, ast.Assign)
      and any(isinstance(t, ast.Name) and t.id == 'ModelNew' for t in n.targets)
    ]
    return (
      (None, '', info)
      if alias
      else ('CONTRACT_VIOLATION', 'no class ModelNew', info)
    )
  if not model_new:
    return 'CONTRACT_VIOLATION', 'no class ModelNew', info
  cls = model_new[0]
  fwd = [
    n
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == 'forward'
  ]
  if not fwd:
    return 'CONTRACT_VIOLATION', 'ModelNew has no forward', info
  if mode == 'full':
    v = torch_fallback_scan(tree, cls, fwd[0], aliases)
    if v:
      return 'TORCH_FALLBACK', v, info
  return None, '', info


def torch_fallback_scan(tree, cls, fwd, aliases):
  """AST scan of ModelNew.forward and every helper it calls. Returns a violation string or None."""
  mod_funcs = {
    n.name: n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and not _is_jit_decorated(n)
  }
  jit_names = {
    n.name
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and _is_jit_decorated(n)
  }
  # also jit kernels assigned via autotune wrappers: name = triton.autotune(...)(fn) -- treat module-level Assign of jit as kernel
  mod_classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)} - {
    'ModelNew'
  }
  methods = {
    n.name: n
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and not _is_jit_decorated(n)
  }  # jit kernels are not torch helpers
  jit_names |= {
    n.name
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and _is_jit_decorated(n)
  }
  # module-level names: `compute = torch.relu` or `F = torch.nn.functional` resolve to their dotted target so a
  # call through the alias is judged like the direct call; names bound to anything else (a call, a lambda, a
  # literal) stay unresolved and may not be called from forward.
  aliases = dict(aliases)
  mod_assigned = set()
  for n in tree.body:
    if isinstance(n, ast.Assign):
      dn = _dotted(n.value, aliases)
      for t in n.targets:
        if isinstance(t, ast.Name):
          if dn:
            aliases[t.id] = dn
          else:
            mod_assigned.add(t.id)

  # attributes assigned in __init__ from torch/nn calls (parameters, buffers, submodules) are tensor-like
  self_tensor_attrs = set()
  init_fn = methods.get('__init__')
  if init_fn is not None:
    for n in ast.walk(init_fn):
      if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
        dn = _dotted(n.value.func, aliases) or ''
        if (
          dn.split('.')[0] == 'torch'
          or dn.startswith('nn.')
          or 'Parameter' in dn
        ):
          for t in n.targets:
            if (
              isinstance(t, ast.Attribute)
              and isinstance(t.value, ast.Name)
              and t.value.id == 'self'
            ):
              self_tensor_attrs.add(t.attr)
      if (
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ('register_buffer', 'register_parameter')
      ):
        if n.args and isinstance(n.args[0], ast.Constant):
          self_tensor_attrs.add(str(n.args[0].value))

  def param_names(fn):
    ps = [a.arg for a in fn.args.posonlyargs + fn.args.args if a.arg != 'self']
    return (
      ps,
      [a.arg for a in fn.args.kwonlyargs],
      (fn.args.vararg.arg if fn.args.vararg else None),
    )

  def call_taint(fn, call, arg_is_tensor):
    """Which parameters of `fn` receive tensor-tainted arguments at this call site."""
    pos, kwonly, var = param_names(fn)
    t = set()
    for i, flag in enumerate(arg_is_tensor[0]):
      if flag:
        t.add(pos[i] if i < len(pos) else var or f'__extra{i}')
    for k, flag in arg_is_tensor[1]:
      if flag and k is not None:
        t.add(k)
      elif (
        flag and k is None
      ):  # **kwargs splat of a tensor-ish dict: be conservative
        t.update(pos + kwonly)
    return frozenset(t)

  fwd_pos, fwd_kw, fwd_var = param_names(fwd)
  worklist = [
    (
      'forward',
      fwd,
      frozenset(fwd_pos + fwd_kw + ([fwd_var] if fwd_var else [])),
    )
  ]
  seen = {('forward', worklist[0][2])}
  while worklist:
    fname, fn, tainted_params = worklist.pop()
    tainted = set(tainted_params)

    def is_tensor(e):
      """True if expression evaluates to a tensor (tainted). Raises _Viol on tensor computation."""
      if isinstance(e, ast.Name):
        return e.id in tainted
      if isinstance(e, ast.Attribute):
        if isinstance(e.value, ast.Name) and e.value.id == 'self':
          return e.attr in self_tensor_attrs
        if e.attr in FORBIDDEN_ATTRS and is_tensor(e.value):
          raise _Viol(f'tensor attribute .{e.attr}')
        if e.attr in INT_ATTRS:
          is_tensor(e.value)
          return e.attr == 'data' and is_tensor(e.value)
        # `self.fc.weight` is tensor-like; `self.conv.stride` / `self.fc.in_features` are hyper-parameters
        return is_tensor(e.value) and bool(PARAM_ATTR_RE.match(e.attr))
      if isinstance(e, ast.Subscript):
        if isinstance(e.value, ast.Attribute) and e.value.attr in INT_ATTRS:
          return False
        return is_tensor(e.value)
      if isinstance(e, ast.Call):
        f = e.func
        dn = _dotted(f, aliases)
        if dn and (dn.split('.')[0] == 'torch'):
          if dn not in ALLOWED_TORCH_CALLS:
            raise _Viol(f'call to {dn}')
          for a in e.args:
            is_tensor(a)
          for k in e.keywords:
            is_tensor(k.value)
          return dn not in (
            'torch.no_grad',
            'torch.inference_mode',
            'torch.set_grad_enabled',
            'torch.is_tensor',
            'torch.finfo',
            'torch.iinfo',
            'torch.Size',
            'torch.device',
            'torch.get_default_dtype',
            'torch.promote_types',
            'torch.result_type',
            'torch.is_floating_point',
            'torch.cuda.is_available',
            'torch.cuda.current_device',
            'torch.cuda.synchronize',
            'torch.cuda.current_stream',
            'torch.cuda.device',
            'torch.numel',
            'torch.is_grad_enabled',
          )
        if isinstance(f, ast.Attribute):
          recv = f.value
          if isinstance(recv, ast.Name) and recv.id == 'self':
            if f.attr in methods:
              flags = (
                [is_tensor(a) for a in e.args],
                [(k.arg, is_tensor(k.value)) for k in e.keywords],
              )
              key = (f.attr, call_taint(methods[f.attr], e, flags))
              if key not in seen:
                seen.add(key)
                worklist.append((f.attr, methods[f.attr], key[1]))
              return True  # conservatively treat the result as a tensor
            if (
              f.attr
              in (
                'register_buffer',
                'register_parameter',
                'parameters',
                'buffers',
                'to',
                'eval',
                'train',
                'named_parameters',
                'state_dict',
                'children',
                'modules',
              )
              or f.attr in jit_names
            ):
              for a in e.args:
                is_tensor(a)
              return False
            raise _Viol(f'call to self.{f.attr} (not a ModelNew method)')
          recv_is_tensor = is_tensor(recv)
          for a in e.args:
            is_tensor(a)
          for k in e.keywords:
            is_tensor(k.value)
          if recv_is_tensor:
            if f.attr in INT_METHODS:
              return False
            if f.attr in VIEW_METHODS:
              return True
            raise _Viol(f'tensor method .{f.attr}()')
          return False  # e.g. kernel.run(...), grid_fn(...), math.ceil(...)
        if isinstance(f, ast.Subscript):  # kernel[grid](...)
          is_tensor(f.value)
          for a in e.args:
            is_tensor(a)
          for k in e.keywords:
            is_tensor(k.value)
          return False
        if isinstance(f, ast.Name):
          flags = (
            [is_tensor(a) for a in e.args],
            [(k.arg, is_tensor(k.value)) for k in e.keywords],
          )
          if f.id in jit_names:
            return False
          if f.id in mod_funcs:
            key = (f.id, call_taint(mod_funcs[f.id], e, flags))
            if key not in seen:
              seen.add(key)
              worklist.append((f.id, mod_funcs[f.id], key[1]))
            return any(flags[0]) or any(fl for _, fl in flags[1]) or True
          if f.id in mod_classes:
            raise _Viol(f'instantiating class {f.id} in forward')
          if f.id in PY_BUILTINS or f.id in aliases:
            return False
          if f.id in mod_assigned:
            raise _Viol(
              f'call to module-level name {f.id} with an unresolved target'
            )
          return False
        if isinstance(f, ast.Lambda):
          return False
        for a in e.args:
          is_tensor(a)
        return False
      if isinstance(e, ast.BinOp):
        lt, rt = is_tensor(e.left), is_tensor(e.right)
        if lt or rt:
          raise _Viol(f'tensor arithmetic ({type(e.op).__name__})')
        return False
      if isinstance(e, ast.UnaryOp):
        if is_tensor(e.operand):
          raise _Viol(f'tensor unary op ({type(e.op).__name__})')
        return False
      if isinstance(e, ast.Compare):
        if all(isinstance(op, (ast.Is, ast.IsNot)) for op in e.ops) or any(
          isinstance(c, ast.Constant) and c.value is None
          for c in [e.left] + e.comparators
        ):
          return False  # `x is None` / `res is not None`: identity checks, not computation
        if is_tensor(e.left) or any(is_tensor(c) for c in e.comparators):
          raise _Viol('tensor comparison')
        return False
      if isinstance(e, ast.IfExp):
        is_tensor(e.test)
        return is_tensor(e.body) or is_tensor(e.orelse)
      if isinstance(e, (ast.Tuple, ast.List)):
        return any([is_tensor(x) for x in e.elts])
      if isinstance(e, ast.Starred):
        return is_tensor(e.value)
      if isinstance(e, ast.Lambda):
        return False
      if isinstance(e, ast.Dict):
        for v in e.values:
          if v is not None:
            is_tensor(v)
        return False
      if isinstance(e, ast.JoinedStr):
        return False
      if isinstance(e, (ast.Constant, ast.Slice)):
        return False
      if isinstance(e, ast.BoolOp):
        return any([is_tensor(v) for v in e.values])
      if isinstance(e, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
        for g in e.generators:
          if is_tensor(g.iter):
            raise _Viol('iterating over a tensor')
        # elements may reference loop vars; conservatively scan
        is_tensor(e.elt)
        return False
      if isinstance(e, ast.NamedExpr):
        t = is_tensor(e.value)
        if t:
          tainted.add(e.target.id)
        return t
      return False

    try:
      for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign):
          t = is_tensor(stmt.value)
          for tgt in stmt.targets:
            if isinstance(tgt, ast.Subscript) and is_tensor(tgt.value):
              raise _Viol('item assignment into a tensor')
            if (
              isinstance(tgt, ast.Attribute)
              and isinstance(tgt.value, ast.Name)
              and tgt.value.id == 'self'
              and t
            ):
              pass
            if t and isinstance(tgt, ast.Name):
              tainted.add(tgt.id)
            if isinstance(tgt, (ast.Tuple, ast.List)):
              if t:
                for el in tgt.elts:
                  if isinstance(el, ast.Name):
                    tainted.add(el.id)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
          if is_tensor(stmt.value) and isinstance(stmt.target, ast.Name):
            tainted.add(stmt.target.id)
        elif isinstance(stmt, ast.AugAssign):
          if is_tensor(stmt.target) or is_tensor(stmt.value):
            raise _Viol('augmented assignment on a tensor')
        elif isinstance(stmt, ast.Expr) or (
          isinstance(stmt, ast.Return) and stmt.value is not None
        ):
          is_tensor(stmt.value)
        elif isinstance(stmt, (ast.If, ast.While)):
          is_tensor(stmt.test)
        elif isinstance(stmt, ast.For):
          if is_tensor(stmt.iter):
            raise _Viol('iterating over a tensor')
        elif isinstance(stmt, ast.With):
          for it in stmt.items:
            is_tensor(it.context_expr)
        elif isinstance(stmt, ast.Assert):
          is_tensor(stmt.test)
    except _Viol as v:
      return f'{fname}: {v}'
  return None


class _Viol(Exception):
  pass


# ------------------------------------------------------------------------------------------------
# Worker (inside sandbox)
# ------------------------------------------------------------------------------------------------
class LaunchCounter:
  def __init__(self):
    self.n = 0
    self.patched = []

  def install(self):
    import triton.runtime.jit as J

    targets = [J.JITFunction]
    try:
      import triton.runtime.interpreter as I

      targets.append(I.InterpretedFunction)
    except Exception:
      pass
    for cls in targets:
      orig = cls.run
      me = self

      def make(orig):
        def run(self_, *a, **k):
          me.n += 1
          return orig(self_, *a, **k)

        return run

      cls.run = make(orig)
      self.patched.append((cls, orig))


def split_init(init):
  """KernelBench convention: flat list of positional args. KernelBook convention: [args_list, kwargs_dict]."""
  if (
    isinstance(init, (list, tuple))
    and len(init) == 2
    and isinstance(init[0], (list, tuple))
    and isinstance(init[1], dict)
  ):
    return list(init[0]), dict(init[1])
  return list(init), {}


def _first_line(e):
  s = str(e).strip().splitlines()
  return f'{type(e).__name__}: {s[0][:300] if s else ""}'


def _max_abs_err(a, b):
  import torch

  try:
    a = a.detach().to(torch.float64)
    b = b.detach().to(torch.float64)
    if a.shape != b.shape:
      return None
    d = (a - b).abs()
    d = torch.where(torch.isnan(a) & torch.isnan(b), torch.zeros_like(d), d)
    return float(d.max()) if d.numel() else 0.0
  except Exception:
    return None


def verify_in_process(task, candidate_src, mode='full', soft_timeout=120.0):
  """Run static checks then the differential test. Returns the result dict."""
  res = {
    'id': task['id'],
    'status': None,
    'msg': '',
    'per_shape': [],
    'launches_total': 0,
  }
  status, msg, info = static_checks(candidate_src, mode)
  res['static'] = info
  if status:
    res.update(status=status, msg=msg)
    return res
  import torch

  torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
  counter = LaunchCounter()
  if mode == 'full':
    counter.install()
  ref_ns = {'__name__': 'reference_module'}
  exec(compile(task['reference_src'], 'reference.py', 'exec'), ref_ns)
  cand_ns = {'__name__': 'candidate_module'}
  try:
    exec(
      compile(
        candidate_src,
        os.path.abspath('candidate.py')
        if os.path.exists('candidate.py')
        else 'candidate.py',
        'exec',
      ),
      cand_ns,
    )
    ModelNew = cand_ns['ModelNew']
  except BaseException as e:  # candidate code may raise anything
    if isinstance(e, (KeyboardInterrupt, SystemExit)):
      raise
    res.update(status='RUNTIME_ERROR', msg='import: ' + _first_line(e))
    return res

  shapes = task.get('test_shapes') or [None]
  runs = [(0, 1.0), (1, 1.0), (2, 1.0)] + (
    [(3, 0.01)] if task.get('family') == 'procedural' else []
  )
  overall = None
  for shape in shapes:
    rec = {
      'shape': shape,
      'status': 'PASS',
      'max_abs_err': 0.0,
      'launches': 0,
      'time_s': 0.0,
      'msg': '',
    }
    t_shape = 0.0
    for seed, scale in runs:
      if shape is not None:
        ref_ns['SHAPE'] = tuple(shape)
        ref_ns['SCALE'] = scale
      try:
        torch.manual_seed(seed)
        inputs = ref_ns['get_inputs']()
        init = (
          ref_ns['get_init_inputs']() if 'get_init_inputs' in ref_ns else []
        )
        init_args, init_kwargs = split_init(init)
        torch.manual_seed(1000 + seed)
        ref = ref_ns['Model'](*init_args, **init_kwargs).eval()
        with torch.no_grad():
          out_ref = ref(*[
            t.clone() if torch.is_tensor(t) else t for t in inputs
          ])
      except Exception as e:
        rec.update(status='RUNTIME_ERROR', msg='reference: ' + _first_line(e))
        break
      try:
        torch.manual_seed(1000 + seed)
        new = ModelNew(*init_args, **init_kwargs)
        new.eval()
        sd = ref.state_dict()
        if len(sd):
          try:
            new.load_state_dict(sd, strict=True)
          except Exception as e:
            rec.update(
              status='CONTRACT_VIOLATION',
              msg='state_dict mismatch: ' + _first_line(e),
            )
            break
        new = new.to('cuda')
        in_new = [
          t.clone().to('cuda') if torch.is_tensor(t) else t for t in inputs
        ]
        n0 = counter.n
        t0 = time.time()
        with torch.no_grad():
          out_new = new(*in_new)
        torch.cuda.synchronize()
        dt = time.time() - t0
        t_shape += dt
        launches = counter.n - n0
        rec['launches'] += launches
        res['launches_total'] += launches
      except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
          raise
        rec.update(status='RUNTIME_ERROR', msg=_first_line(e))
        break
      if mode == 'full' and launches == 0:
        rec.update(
          status='NO_KERNEL_LAUNCH',
          msg='ModelNew.forward launched no Triton kernel',
        )
        break
      if not torch.is_tensor(out_new):
        rec.update(
          status='FAIL', msg=f'output is {type(out_new).__name__}, not a tensor'
        )
        break
      if mode == 'full' and not task.get('inplace_ref'):
        ptrs = {t.data_ptr() for t in in_new if torch.is_tensor(t)}
        if out_new.data_ptr() in ptrs:
          rec.update(
            status='CONTRACT_VIOLATION', msg='output aliases an input tensor'
          )
          break
      if out_new.device.type != 'cuda':
        rec.update(
          status='FAIL',
          msg=f"output on {out_new.device}, expected the inputs' device (cuda)",
        )
        break
      out_new = out_new.cpu()
      if tuple(out_new.shape) != tuple(out_ref.shape):
        rec.update(
          status='FAIL',
          msg=f'shape mismatch: got {tuple(out_new.shape)}, expected {tuple(out_ref.shape)}',
          max_abs_err=None,
        )
        break
      if out_new.dtype != out_ref.dtype:
        rec.update(
          status='FAIL',
          msg=f'dtype mismatch: got {out_new.dtype}, expected {out_ref.dtype}',
          max_abs_err=_max_abs_err(out_new, out_ref),
        )
        break
      err = _max_abs_err(out_new, out_ref)
      rec['max_abs_err'] = max(
        rec['max_abs_err'] or 0.0, err if err is not None else float('inf')
      )
      try:
        torch.testing.assert_close(
          out_new, out_ref, atol=task['atol'], rtol=task['rtol'], equal_nan=True
        )
      except AssertionError as e:
        rec.update(
          status='FAIL',
          msg=f'mismatch seed={seed} scale={scale}: max_abs_err={err:.3e}; '
          + str(e).splitlines()[0][:200],
        )
        break
      if t_shape > soft_timeout:
        rec.update(
          status='TIMEOUT',
          msg=f'{t_shape:.1f}s of candidate forward time on this shape',
        )
        break
    rec['time_s'] = round(t_shape, 2)
    res['per_shape'].append(rec)
    if rec['status'] != 'PASS' and overall is None:
      overall = rec['status']
      res['msg'] = f'shape={shape}: {rec["msg"]}'
  res['status'] = overall or 'PASS'
  return res


def cmd_worker(a):
  # sockets off (no network for candidate code)
  import socket

  def _blocked(*args, **kw):
    raise OSError('network disabled in sandbox')

  class _BlockedSocket(
    socket.socket
  ):  # a class, so `class SSLSocket(socket)` in ssl.py still imports
    def __init__(self, *args, **kw):
      raise OSError('network disabled in sandbox')

  socket.socket = _BlockedSocket  # type: ignore
  socket.SocketType = _BlockedSocket  # type: ignore
  socket.create_connection = _blocked
  socket.getaddrinfo = _blocked
  os.environ.pop('TRITON_INTERPRET', None)
  os.environ['TRITON_F32_DEFAULT'] = 'ieee'
  task = json.loads(pathlib.Path(a.task).read_text())
  cand = pathlib.Path(a.candidate).read_text()
  t0 = time.time()
  import contextlib

  with contextlib.redirect_stdout(sys.stderr):
    try:
      res = verify_in_process(
        task, cand, mode=a.mode, soft_timeout=a.soft_timeout
      )
    except Exception as e:  # harness bug -> surface it, don't fake a status
      import traceback

      res = {
        'id': task['id'],
        'status': 'NO_OUTPUT',
        'msg': 'harness exception: ' + _first_line(e),
        'per_shape': [],
        'launches_total': 0,
        'traceback': traceback.format_exc()[-2000:],
      }
  res['wall_s'] = round(time.time() - t0, 2)
  res['mode'] = a.mode
  pathlib.Path(a.out).write_text(json.dumps(res))


# ------------------------------------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------------------------------------
def run_sandboxed(
  task,
  candidate_src,
  mode,
  scratch_root,
  ulimit_kb=8000000,
  soft_timeout=120.0,
  niceness=10,
):
  scratch = pathlib.Path(scratch_root) / task['id']
  if scratch.exists():
    shutil.rmtree(scratch, ignore_errors=True)
  scratch.mkdir(parents=True, exist_ok=True)
  (scratch / 'task.json').write_text(json.dumps(task))
  (scratch / 'candidate.py').write_text(candidate_src)
  out = scratch / 'result.json'
  n_shapes = max(1, len(task.get('test_shapes') or [None]))
  hard = int(soft_timeout * n_shapes + 90)  # import overhead + reference time
  inner = (
    f'exec {sys.executable} {pathlib.Path(__file__).resolve()} worker --task task.json --candidate candidate.py '
    f'--out result.json --mode {mode} --soft-timeout {soft_timeout}'
  )
  cmd = [
    'nice',
    '-n',
    str(niceness),
    'timeout',
    '-k',
    '5',
    str(hard),
    'bash',
    '-c',
    inner,
  ]
  env = {
    'PATH': '/usr/bin:/bin',
    'HOME': str(scratch),
    'TRITON_F32_DEFAULT': 'ieee',
    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
    'OMP_NUM_THREADS': '2',
    'MKL_NUM_THREADS': '2',
    'TRITON_CACHE_DIR': str(scratch / '.triton'),
    'PYTHONDONTWRITEBYTECODE': '1',
    'no_proxy': '*',
    'PYTHONHASHSEED': '0',
    'TMPDIR': str(scratch),
  }
  t0 = time.time()
  proc = subprocess.run(
    cmd, cwd=str(scratch), env=env, capture_output=True, text=True
  )
  wall = time.time() - t0
  if out.exists():
    res = json.loads(out.read_text())
  elif proc.returncode in (124, 137, -9) or wall >= hard - 1:
    res = {
      'id': task['id'],
      'status': 'TIMEOUT',
      'msg': f'hard timeout after {hard}s',
      'per_shape': [],
      'launches_total': 0,
    }
  elif proc.returncode in (-11, 139, -7, 134, -6):
    res = {
      'id': task['id'],
      'status': 'RUNTIME_ERROR',
      'per_shape': [],
      'launches_total': 0,
      'msg': f'worker crashed (signal {proc.returncode}): compiled-kernel crash (e.g. illegal memory access from an unmasked load/store)',
    }
  else:
    tail = (proc.stderr or '').strip().splitlines()[-3:]
    res = {
      'id': task['id'],
      'status': 'NO_OUTPUT',
      'msg': f'worker exit={proc.returncode}: {" | ".join(tail)[:300]}',
      'per_shape': [],
      'launches_total': 0,
    }
  res['wall_s_total'] = round(wall, 2)
  res['stderr_tail'] = (proc.stderr or '')[-600:]
  shutil.rmtree(scratch, ignore_errors=True)
  return res


def load_jsonl(p):
  return [
    json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()
  ]


def _orchestrate(
  jobs, out_path, workers, mode, scratch_root, soft_timeout, label
):
  out_path = pathlib.Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  done = {r['id'] for r in load_jsonl(out_path)} if out_path.exists() else set()
  todo = [(t, c) for t, c in jobs if t['id'] not in done]
  print(
    f'[{label}] {len(todo)} to run ({len(done)} already done), workers={workers}, mode={mode}',
    flush=True,
  )
  t0 = time.time()
  counts = {}
  with out_path.open('a') as fh, cf.ThreadPoolExecutor(workers) as ex:
    futs = {
      ex.submit(
        run_sandboxed, t, c, mode, scratch_root, soft_timeout=soft_timeout
      ): t['id']
      for t, c in todo
    }
    for i, f in enumerate(cf.as_completed(futs), 1):
      r = f.result()
      counts[r['status']] = counts.get(r['status'], 0) + 1
      fh.write(json.dumps(r) + '\n')
      fh.flush()
      if i % 10 == 0 or i == len(todo):
        el = time.time() - t0
        print(
          f'[{label}] {i}/{len(todo)} {el / 60:.1f} min  {counts}', flush=True
        )
  return counts


def cmd_run(a):
  tasks = load_jsonl(a.corpus)
  cdir = pathlib.Path(a.candidates)
  jobs = []
  for t in tasks:
    p = cdir / f'{t["id"]}.py'
    if p.exists():
      jobs.append((t, p.read_text()))
    else:
      jobs.append((t, None))
  # tasks without a candidate file -> NO_OUTPUT immediately
  out_path = pathlib.Path(a.out)
  done = {r['id'] for r in load_jsonl(out_path)} if out_path.exists() else set()
  with_cand = [(t, c) for t, c in jobs if c is not None]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  if not a.skip_missing:
    with out_path.open('a') as fh:
      for t, c in jobs:
        if c is None and t['id'] not in done:
          fh.write(
            json.dumps({
              'id': t['id'],
              'status': 'NO_OUTPUT',
              'msg': 'no candidate parsed from the model response',
              'per_shape': [],
              'launches_total': 0,
            })
            + '\n'
          )
  _orchestrate(
    with_cand, a.out, a.workers, 'full', a.scratch, a.soft_timeout, 'run'
  )


def cmd_selfcheck(a):
  tasks = load_jsonl(a.corpus)
  jobs = [(t, t['reference_src'] + '\n\nModelNew = Model\n') for t in tasks]
  _orchestrate(
    jobs, a.out, a.workers, 'trusted', a.scratch, a.soft_timeout, 'selfcheck'
  )


def cmd_mutation(a):
  muts = load_jsonl(a.mutants)
  jobs = [(m, m['mutant_src'] + '\n\nModelNew = Model\n') for m in muts]
  _orchestrate(
    jobs, a.out, a.workers, 'trusted', a.scratch, a.soft_timeout, 'mutation'
  )


def cmd_single(a):
  tasks = {t['id']: t for t in load_jsonl(a.corpus)}
  task = tasks[a.task_id]
  res = run_sandboxed(
    task,
    pathlib.Path(a.candidate).read_text(),
    a.mode,
    a.scratch,
    soft_timeout=a.soft_timeout,
  )
  print(json.dumps(res, indent=1))


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd', required=True)
  w = sub.add_parser('worker')
  w.add_argument('--task', required=True)
  w.add_argument('--candidate', required=True)
  w.add_argument('--out', required=True)
  w.add_argument('--mode', default='full')
  w.add_argument('--soft-timeout', type=float, default=120.0)
  for name in ('run', 'selfcheck', 'mutation', 'single'):
    p = sub.add_parser(name)
    p.add_argument('--corpus', default=str(HERE / 'data' / 'corpus.jsonl'))
    p.add_argument('--mutants', default=str(HERE / 'data' / 'mutants.jsonl'))
    p.add_argument('--candidates', default=str(HERE / 'candidates'))
    p.add_argument('--out', default=str(HERE / 'results' / f'{name}_raw.jsonl'))
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--scratch', default=str(HERE / 'tmp' / 'sandbox'))
    p.add_argument('--soft-timeout', type=float, default=120.0)
    p.add_argument('--mode', default='full')
    p.add_argument('--task-id')
    p.add_argument(
      '--skip-missing',
      action='store_true',
      help='run: do not record NO_OUTPUT for tasks without a candidate yet',
    )
    p.add_argument('--candidate')
  a = ap.parse_args()
  {
    'worker': cmd_worker,
    'run': cmd_run,
    'selfcheck': cmd_selfcheck,
    'mutation': cmd_mutation,
    'single': cmd_single,
  }[a.cmd](a)


if __name__ == '__main__':
  main()
