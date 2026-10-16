"""Rebuild the task corpus of the self-distillation SFT dataset from pointers and transforms.

The manifest stores no task code and no third-party code. Each task is rebuilt from its pointer:
  library tasks    : download each referenced upstream file at its pinned commit (GitHub raw), verify its sha256,
                     apply the recorded edit script (upstream lines by number + our own inserted lines)
  KernelBook tasks : fetch GPUMODE/KernelBook dataset_permissive.parquet (huggingface_hub), take the row by uuid,
                     apply the rule
  procedural tasks : regenerate the family with the shipped generator and its seed (needs torch), take the task by id
Every rebuilt module, the rendered task text (spec_render.py) and the prompt (shared header) are checked against
the recorded sha256. Writes corpus.jsonl, prompts/<id>.md and the split id lists of the tasks that were rebuilt.

Exit code 0 when every selected task was rebuilt or explicitly skipped, 1 when any task failed.
Options: --libsrc DIR (local clones instead of GitHub), --kernelbook PARQUET (local file),
--skip-procedural / --skip-kernelbook, --ids a,b.

usage: recreate_corpus.py --manifest dataset_manifest.json.gz --out DIR
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import random
import re
import subprocess
import sys
import types
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest_io import load_manifest, sha256  # noqa: E402

ALIAS = {'timm': 'pytorch-image-models', 'torchvision': 'vision'}


class RebuildError(Exception):
  """A task could not be rebuilt or did not match its recorded hash."""


def check(ok: bool, msg: str) -> None:
  """Raise RebuildError with msg unless ok. These checks decide whether a task enters the corpus."""
  if not ok:
    raise RebuildError(msg)


@dataclasses.dataclass
class Sources:
  """Where upstream material comes from, with a per-file cache."""

  libsrc: str | None = None
  kernelbook: str | None = None
  cache: dict = dataclasses.field(default_factory=dict)

  def upstream_file(self, f: dict, lib: str) -> list[str]:
    """Lines of one referenced upstream file at its pinned commit (GitHub raw or a local clone), sha256-checked."""
    key = (f['repo'], f['commit'], f['path'])
    if key not in self.cache:
      if self.libsrc:
        d = pathlib.Path(self.libsrc) / ALIAS.get(lib, lib)
        p = subprocess.run(
          ['git', '-C', str(d), 'show', f'{f["commit"]}:{f["path"]}'],
          capture_output=True,
          check=False,
        )
        data = p.stdout if p.returncode == 0 else None
      else:
        org_repo = f['repo'].split('github.com/')[1]
        try:
          data = urllib.request.urlopen(
            f'https://raw.githubusercontent.com/{org_repo}/{f["commit"]}/{f["path"]}',
            timeout=60,
          ).read()
        except OSError:
          data = None
      check(
        data is not None and sha256(data) == f['sha256'],
        f'{f["repo"]}@{f["commit"][:10]}:{f["path"]}: fetch failed or sha256 differs',
      )
      self.cache[key] = data.decode('utf-8').split('\n')
    return self.cache[key]

  def kernelbook_rows(self, uuids: list) -> dict:
    """KernelBook rows (entry point, python code) by uuid from the dataset parquet (local file or Hugging Face hub)."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    path = self.kernelbook
    if not path:
      from huggingface_hub import hf_hub_download  # noqa: PLC0415

      path = hf_hub_download(
        'GPUMODE/KernelBook', 'dataset_permissive.parquet', repo_type='dataset'
      )
    t = pq.read_table(path, columns=['uuid', 'entry_point', 'python_code'])
    want = {int(u) for u in uuids}
    return {
      int(u): (e, c)
      for u, e, c in zip(
        t.column('uuid').to_pylist(),
        t.column('entry_point').to_pylist(),
        t.column('python_code').to_pylist(),
      )
      if int(u) in want
    }


def apply_ops(ops: list, files: list[list[str]]) -> str:
  """Apply an edit script: copy upstream lines by (file, line) with our whitespace, or insert our own lines."""
  out = []
  for op in ops:
    if op[0] == 'k':
      _, fi, ln, pre, suf = op
      out.append(pre + files[fi][ln - 1].strip() + suf)
    else:
      out.append(op[1])
  return '\n'.join(out)


def procedural_tasks(manifest: dict) -> dict:
  """Regenerate the procedural family with the shipped generator and its seed (needs torch); tasks by id."""
  g = manifest['rebuild']['procedural_generator']
  sys.modules.setdefault(
    'triton', types.SimpleNamespace(__version__=g['pinned_versions']['triton'])
  )
  sys.path.insert(0, str(HERE / 'generator'))
  import gen_corpus as gen  # noqa: PLC0415

  gen.PYVER, gen.TVER, gen.TRVER = (
    g['pinned_versions']['python'],
    g['pinned_versions']['torch'],
    g['pinned_versions']['triton'],
  )
  return {t['id']: t for t in gen.gen_family_p(random.Random(g['seed']))}


def load_renderer(manifest: dict) -> dict:
  """Namespace of spec_render.py, hash-checked against the manifest."""
  sr_path = HERE / manifest['rebuild']['spec_renderer']['file']
  sr_src = sr_path.read_text(encoding='utf-8')
  check(
    sha256(sr_src) == manifest['rebuild']['spec_renderer']['sha256'],
    f'{sr_path.name} differs from the manifest',
  )
  ns: dict = {}
  exec(compile(sr_src, str(sr_path), 'exec'), ns)
  return ns


def load_header(manifest: dict) -> str:
  """The shared prompt header, hash-checked against the manifest."""
  head = (HERE / manifest['prompt_header']['file']).read_text(encoding='utf-8')
  check(
    sha256(head) == manifest['prompt_header']['sha256'], 'prompt header differs'
  )
  return head


def rebuild_task(
  tid: str,
  manifest: dict,
  src: Sources,
  kb: dict,
  proc: dict,
  renderer: dict,
  head: str,
) -> tuple[dict, str]:
  """Rebuild one task; returns (corpus row, prompt). Raises RebuildError on any hash mismatch."""
  t = manifest['tasks'][tid]
  rb = t['rebuild']
  if rb['kind'] == 'edit_script':
    files = [src.upstream_file(f, t['origin']['lib']) for f in rb['files']]
    ref = apply_ops(rb['ops'], files)
  elif rb['kind'] == 'kernelbook_row':
    e, c = kb[int(rb['uuid'])]
    ref = c.rstrip() + f'\n\n\nModel = {e}   # the harness uses `Model`\n'
  else:
    ref = proc[rb['task_id']]['reference_src']
  check(sha256(ref) == t['reference_sha256'], 'reference module sha256 differs')
  row = dict(id=tid, **t['meta'], reference_src=ref)
  spec = (
    proc[rb['task_id']]['spec']
    if rb['kind'] == 'generator'
    else renderer['render_spec_k'](row, row['spec_format'])
  )
  check(sha256(spec) == t['spec_sha256'], 'task text sha256 differs')
  row['spec'] = spec
  er = manifest['environment_rewrite']
  gspec, n = re.subn(er['pattern'], er['replacement'], spec, count=1)
  check(n == 1, 'environment sentence not found')
  prompt = head + f'### {tid}\n{gspec}\n'
  check(sha256(prompt) == t['prompt_sha256'], 'prompt sha256 differs')
  return row, prompt


def write_splits(out: pathlib.Path, manifest: dict, ok_ids: list[str]) -> None:
  """Write train_ids.txt and val_ids.txt listing only the tasks that were rebuilt."""
  for split in ('train', 'val'):
    (out / f'{split}_ids.txt').write_text(
      ''.join(
        f'{tid}\n' for tid in ok_ids if manifest['tasks'][tid]['split'] == split
      ),
      encoding='utf-8',
    )


def main(argv: list[str] | None = None) -> int:
  """Rebuild the selected tasks; returns the process exit code."""
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument('--manifest', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--ids')
  ap.add_argument('--libsrc')
  ap.add_argument('--kernelbook')
  ap.add_argument('--skip-procedural', action='store_true')
  ap.add_argument('--skip-kernelbook', action='store_true')
  a = ap.parse_args(argv)
  manifest = load_manifest(a.manifest)
  out = pathlib.Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  (out / 'prompts').mkdir(exist_ok=True)
  sel = sorted(a.ids.split(',')) if a.ids else sorted(manifest['tasks'])
  renderer = load_renderer(manifest)
  head = load_header(manifest)
  src = Sources(libsrc=a.libsrc, kernelbook=a.kernelbook)
  kinds = {tid: manifest['tasks'][tid]['rebuild']['kind'] for tid in sel}
  skipped_kinds = set()
  if a.skip_kernelbook:
    skipped_kinds.add('kernelbook_row')
  if a.skip_procedural:
    skipped_kinds.add('generator')
  kb = (
    src.kernelbook_rows([
      manifest['tasks'][t]['rebuild']['uuid']
      for t in sel
      if kinds[t] == 'kernelbook_row'
    ])
    if 'kernelbook_row' in kinds.values()
    and 'kernelbook_row' not in skipped_kinds
    else {}
  )
  proc = (
    procedural_tasks(manifest)
    if 'generator' in kinds.values() and 'generator' not in skipped_kinds
    else {}
  )
  ok_ids, skipped, failed = [], [], []
  with open(out / 'corpus.jsonl', 'w', encoding='utf-8') as fc:
    for tid in sel:
      if kinds[tid] in skipped_kinds:
        skipped.append(tid)
        continue
      try:
        row, prompt = rebuild_task(tid, manifest, src, kb, proc, renderer, head)
      except Exception as e:  # noqa: BLE001  any failure is recorded per task and fails the run
        failed.append((tid, f'{type(e).__name__}: {str(e)[:120]}'))
        continue
      (out / 'prompts' / f'{tid}.md').write_text(prompt, encoding='utf-8')
      fc.write(json.dumps(row) + '\n')
      ok_ids.append(tid)
  write_splits(out, manifest, ok_ids)
  print(
    f'rebuilt and hash-verified: {len(ok_ids)}/{len(sel)} tasks '
    f'(skipped {len(skipped)}, failed {len(failed)}) -> {out}/corpus.jsonl, prompts/'
  )
  for tid, msg in failed[:10]:
    print('  FAILED', tid, msg)
  if failed:
    print('RECREATE_CORPUS_FAILED', file=sys.stderr)
    return 1
  print('RECREATE_CORPUS_OK')
  return 0


if __name__ == '__main__':
  sys.exit(main())
