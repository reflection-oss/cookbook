"""Check that a finished run exported a servable checkpoint."""

from __future__ import annotations

import json
import math
import os
import pathlib


def main() -> None:
  """Assert the export is complete, passed its generation test and trained."""
  run_dir = pathlib.Path(
    os.environ.get('RUN_DIR', '/data/gemma4-tool-reasoning')
  )
  export = run_dir / 'export'
  for name in ('config.json', 'tokenizer_config.json', 'chat_template.jinja'):
    assert (export / name).exists(), f'missing {export / name}'
  assert (export / 'tokenizer.json').exists(), 'no tokenizer'
  index = export / 'model.safetensors.index.json'
  if index.exists():
    with index.open(encoding='utf-8') as f:
      shards = sorted(set(json.load(f)['weight_map'].values()))
  else:
    shards = ['model.safetensors']
  missing = [s for s in shards if not (export / s).exists()]
  assert not missing, f'missing weight files: {missing}'
  assert (export / 'SMOKE_OK').exists(), 'export did not pass the smoke test'
  state = run_dir / 'checkpoints' / 'trainer_state.json'
  with state.open(encoding='utf-8') as f:
    history = json.load(f)['log_history']
  losses = [row['loss'] for row in history if 'loss' in row]
  evals = [row['eval_loss'] for row in history if 'eval_loss' in row]
  assert losses and all(math.isfinite(x) for x in losses), 'bad training loss'
  assert evals and math.isfinite(evals[-1]), 'no finite held-out loss'
  print(
    f'ok: {len(shards)} weight file(s), {len(losses)} logged steps, '
    f'training loss {losses[0]:.3f} -> {losses[-1]:.3f}, '
    f'held-out loss {evals[-1]:.3f}'
  )


if __name__ == '__main__':
  main()
