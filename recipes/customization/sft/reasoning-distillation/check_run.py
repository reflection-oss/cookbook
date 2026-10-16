"""Print the losses of a finished run and plot the loss curve."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics

from matplotlib import figure


def main() -> None:
  """Print the logged losses and the held-out loss, and save a plot."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--run-dir',
    type=pathlib.Path,
    default=pathlib.Path(
      os.environ.get('RUN_DIR', '/data/gemma4-tool-reasoning')
    ),
    help='run folder written by train.sh',
  )
  parser.add_argument(
    '--plot', type=pathlib.Path, default=pathlib.Path('loss_curve.png')
  )
  args = parser.parse_args()
  state = args.run_dir / 'checkpoints' / 'trainer_state.json'
  with state.open(encoding='utf-8') as f:
    history = json.load(f)['log_history']
  train = [row for row in history if 'loss' in row]
  evals = [row for row in history if 'eval_loss' in row]
  summary = [row for row in history if 'train_runtime' in row]

  print('step  training loss  learning rate')
  for row in train:
    print(f'{row["step"]:>4}  {row["loss"]:>13.4f}  {row["learning_rate"]:.2e}')
  for row in evals:
    print(f'held-out loss after step {row["step"]}: {row["eval_loss"]:.4f}')
  if summary:
    print(f'training time: {summary[-1]["train_runtime"] / 60:.1f} minutes')
  window = min(5, len(train))
  first = statistics.mean(row['loss'] for row in train[:window])
  last = statistics.mean(row['loss'] for row in train[-window:])
  print(
    f'training loss, mean of the first and last {window} logged steps: '
    f'{first:.3f} -> {last:.3f}'
  )

  fig = figure.Figure(figsize=(8, 4.5), dpi=160)
  ax = fig.subplots()
  steps, losses = [r['step'] for r in train], [r['loss'] for r in train]
  ax.plot(steps, losses, label='training loss')
  eval_steps = [r['step'] for r in evals]
  ax.plot(eval_steps, [r['eval_loss'] for r in evals], 'o', label='held-out')
  ax.set_xlabel('optimizer step')
  ax.set_ylabel('loss')
  ax.grid(True, alpha=0.3)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(args.plot)
  print(f'saved {args.plot}')


if __name__ == '__main__':
  main()
