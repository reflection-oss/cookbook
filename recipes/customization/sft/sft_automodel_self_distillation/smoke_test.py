"""Offline smoke test of the self-distillation data kit.

Checks the manifest, the edit-script applier, the split lists, the failure exit code of recreate_corpus.py, the prompt
lookup and length policy of build_sft_rows.py, the answer parser of sample_beam.py, and the verifier's static
anti-cheat scan, including a regression test for calls through module-level aliases. Needs no network, GPU, torch or
model files. With --online it also rebuilds three tasks from their pointers (GitHub + Hugging Face hub) and checks
the hashes.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_sft_rows  # noqa: E402
import recreate_corpus  # noqa: E402
import sample_beam  # noqa: E402
import verify_gpu  # noqa: E402
from manifest_io import load_manifest, sha256  # noqa: E402

MANIFEST = HERE / 'dataset_manifest.json.gz'

CLEAN_CANDIDATE = """
import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
  pid = tl.program_id(0)
  offs = pid * BLOCK + tl.arange(0, BLOCK)
  mask = offs < n
  x = tl.load(x_ptr + offs, mask=mask)
  tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


class ModelNew(torch.nn.Module):
  def forward(self, x):
    y = torch.empty_like(x)
    n = x.numel()
    relu_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
    return y
"""

ALIASED_TORCH_CALL = CLEAN_CANDIDATE.replace(
  'class ModelNew', 'compute = torch.relu\n\n\nclass ModelNew'
).replace('    return y\n', '    return compute(y)\n')

ALIASED_MODULE = CLEAN_CANDIDATE.replace(
  'class ModelNew', 'F = torch.nn.functional\n\n\nclass ModelNew'
).replace('    return y\n', '    return F.relu(y)\n')

UNRESOLVED_CALLABLE = CLEAN_CANDIDATE.replace(
  'class ModelNew', 'act = lambda t: t * 2\n\n\nclass ModelNew'
).replace('    return y\n', '    return act(y)\n')


def check_manifest() -> dict:
  """The manifest is complete, stores no task code and compresses under the repo's file-size limit."""
  manifest = load_manifest(MANIFEST)
  tasks = manifest['tasks']
  assert len(tasks) == 1000, f'expected 1000 tasks, found {len(tasks)}'
  kinds: dict[str, int] = {}
  for t in tasks.values():
    kinds[t['rebuild']['kind']] = kinds.get(t['rebuild']['kind'], 0) + 1
    assert 'reference_src' not in t['meta'] and 'spec' not in t['meta'], (
      'task code must not be stored'
    )
    if t['rebuild']['kind'] == 'edit_script':
      assert all(op[0] in ('k', 'i') for op in t['rebuild']['ops'])
      assert all(
        f['sha256'] and f['commit'] and f['path'] for f in t['rebuild']['files']
      )
  assert kinds == {
    'edit_script': 652,
    'kernelbook_row': 128,
    'generator': 220,
  }, kinds
  assert MANIFEST.stat().st_size < 2**20, (
    'manifest must stay under the 1 MiB checked-in file limit'
  )
  for key in ('spec_renderer',):
    f = manifest['rebuild'][key]
    assert (
      sha256((HERE / f['file']).read_text(encoding='utf-8')) == f['sha256']
    ), f'{f["file"]} differs'
  head = manifest['prompt_header']
  assert (
    sha256((HERE / head['file']).read_text(encoding='utf-8')) == head['sha256']
  ), 'prompt header differs'
  assert (
    sha256((HERE / 'verify_gpu.py').read_bytes())
    == manifest['verifier']['sha256']
  ), 'verify_gpu.py differs'
  print(f'ok: manifest complete ({kinds}), pinned files match')
  return manifest


def check_applier() -> None:
  """The edit-script applier reproduces a module from synthetic upstream lines."""
  files = [
    [
      'import torch',
      '',
      'class Up(torch.nn.Module):',
      '    def forward(self, x):',
      '        return x * 2',
    ]
  ]
  ops = [
    ['k', 0, 1, '', ''],
    ['i', ''],
    ['k', 0, 3, '', ''],
    ['k', 0, 4, '  ', ''],
    ['k', 0, 5, '    ', ''],
    ['i', ''],
    ['i', 'Model = Up'],
  ]
  expected = 'import torch\n\nclass Up(torch.nn.Module):\n  def forward(self, x):\n    return x * 2\n\nModel = Up'
  assert recreate_corpus.apply_ops(ops, files) == expected
  print('ok: applier round trip')


def check_recreate_corpus(manifest: dict) -> None:
  """Split lists hold only rebuilt tasks, and a failed task makes the CLI exit nonzero."""
  with tempfile.TemporaryDirectory() as tmp:
    out = pathlib.Path(tmp)
    recreate_corpus.write_splits(
      out,
      {
        'tasks': {
          'A': {'split': 'train'},
          'B': {'split': 'val'},
          'C': {'split': 'train'},
        }
      },
      ['A', 'B'],
    )
    assert (out / 'train_ids.txt').read_text() == 'A\n' and (
      out / 'val_ids.txt'
    ).read_text() == 'B\n'
    # one library task whose upstream file cannot be fetched from an empty local clone directory
    tid = next(
      t
      for t, v in manifest['tasks'].items()
      if v['rebuild']['kind'] == 'edit_script'
    )
    tiny = {
      k: manifest[k]
      for k in ('rebuild', 'prompt_header', 'environment_rewrite')
    }
    tiny['tasks'] = {tid: manifest['tasks'][tid]}
    (out / 'manifest.json').write_text(json.dumps(tiny), encoding='utf-8')
    (out / 'libsrc').mkdir()
    res = subprocess.run(
      [
        sys.executable,
        str(HERE / 'recreate_corpus.py'),
        '--manifest',
        str(out / 'manifest.json'),
        '--out',
        str(out / 'corpus'),
        '--libsrc',
        str(out / 'libsrc'),
      ],
      capture_output=True,
      text=True,
      check=False,
    )
    assert res.returncode == 1 and 'RECREATE_CORPUS_FAILED' in res.stderr, (
      res.stdout + res.stderr
    )
    assert 'FAILED ' + tid in res.stdout, res.stdout
    assert (out / 'corpus' / 'train_ids.txt').read_text() == '' and (
      out / 'corpus' / 'val_ids.txt'
    ).read_text() == ''
  print('ok: split lists exclude failed tasks, failed rebuild exits 1')


def check_build_sft_rows() -> None:
  """First PASS sample, prompt lookup with hash check, and the sequence-length boundary."""
  with tempfile.TemporaryDirectory() as tmp:
    run = pathlib.Path(tmp) / 'run'
    prompts = pathlib.Path(tmp) / 'prompts'
    (run / 'raw' / 'T-1').mkdir(parents=True)
    prompts.mkdir()
    (prompts / 'T-1.md').write_text('prompt text', encoding='utf-8')
    for k, status in ((0, 'FAIL'), (1, 'PASS'), (2, 'PASS')):
      (run / 'raw' / 'T-1' / f's{k}.json').write_text(
        json.dumps({'k': k, 'status': status}), encoding='utf-8'
      )
    (run / 'raw' / 'T-1' / 's10.json').write_text(
      json.dumps({'k': 10, 'status': 'PASS'}), encoding='utf-8'
    )
    assert build_sft_rows.first_pass(run, 'T-1')['k'] == 1
    assert build_sft_rows.first_pass(run, 'T-2') is None
    assert (
      build_sft_rows.find_prompt(prompts, 'T-1', sha256('prompt text'))
      == 'prompt text'
    )
    for bad in ('T-2', 'T-1'):
      try:
        build_sft_rows.find_prompt(prompts, bad, sha256('other text'))
      except (FileNotFoundError, ValueError):
        continue
      raise AssertionError(f'{bad}: prompt lookup should have failed')
  budget = build_sft_rows.DEFAULT_MAX_SEQ_TOKENS
  assert build_sft_rows.fits_budget(budget + 1, budget), (
    'a row that shifts to exactly the budget fits'
  )
  assert not build_sft_rows.fits_budget(budget + 2, budget), (
    'one token more does not'
  )
  print('ok: first PASS, prompt lookup, length boundary')


def check_parse_answer() -> None:
  """The answer parser takes the block under the task header, else the last ModelNew block."""
  text = 'intro\n### T-1\n```python\nclass ModelNew: pass\n```\n### T-2\n```python\nother\n```\n'
  assert sample_beam.parse_answer('T-1', text) == (
    'exact_header',
    'class ModelNew: pass',
  )
  assert sample_beam.parse_answer(
    'T-9', '```python\nx = 1\n```\n```python\nclass ModelNew: ...\n```'
  ) == ('lone_block', 'class ModelNew: ...')
  assert sample_beam.parse_answer('T-9', 'no code here') == ('none', None)
  print('ok: answer parser')


def check_static_scan() -> None:
  """A clean kernel passes the static checks; torch calls through module-level aliases are rejected."""
  status, msg, _ = verify_gpu.static_checks(CLEAN_CANDIDATE, 'full')
  assert status is None, (status, msg)
  for name, src in (
    ('aliased torch call', ALIASED_TORCH_CALL),
    ('aliased module', ALIASED_MODULE),
    ('unresolved callable', UNRESOLVED_CALLABLE),
  ):
    status, msg, _ = verify_gpu.static_checks(src, 'full')
    assert status == 'TORCH_FALLBACK', (
      f'{name}: expected TORCH_FALLBACK, got {status} {msg}'
    )
  print('ok: static scan rejects aliased and unresolved calls')


def check_online() -> None:
  """Rebuild three tasks from their pointers and check the hashes (needs network)."""
  with tempfile.TemporaryDirectory() as out:
    cmd = [
      sys.executable,
      str(HERE / 'recreate_corpus.py'),
      '--manifest',
      str(MANIFEST),
      '--out',
      out,
      '--ids',
      'FLA-0001,K-0183,P-0113',
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0 and 'RECREATE_CORPUS_OK' in res.stdout, (
      res.stdout + res.stderr
    )
    print('ok: three tasks rebuilt online and hash-verified')


def main() -> None:
  """Run every offline check, and the online rebuild with --online."""
  manifest = check_manifest()
  check_applier()
  check_recreate_corpus(manifest)
  check_build_sft_rows()
  check_parse_answer()
  check_static_scan()
  if '--online' in sys.argv:
    check_online()


if __name__ == '__main__':
  main()
