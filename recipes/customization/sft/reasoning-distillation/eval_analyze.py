"""Section scores, deltas against a baseline, loop diagnostics and tokens.

    python eval_analyze.py <root> <model_dir> [--baseline <root> <model_dir>]

`model_dir` is the registry name with `/` replaced by `_`, the directory
name under `result/` and `score/`. Sections are unweighted means of their
categories; the overall number is BFCL's own, read from data_overall.csv.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import statistics

from bfcl_eval.constants import model_config

SECTIONS = {
  'single-turn non-live': 'simple_python simple_java simple_javascript '
  'multiple parallel parallel_multiple',
  'single-turn live': 'live_simple live_multiple live_parallel '
  'live_parallel_multiple',
  'irrelevance': 'irrelevance live_irrelevance',
  'relevance': 'live_relevance',
  'multi-turn': 'multi_turn_base multi_turn_miss_func '
  'multi_turn_miss_param multi_turn_long_context',
  'memory': 'memory_kv memory_vector memory_rec_sum',
}
_OVERALL_LINE = re.compile(r'^\d+,([\d.]+%),(.*?),https://')


def load(root: str, model: str, kind: str) -> dict[str, list[dict]]:
  """Rows of every BFCL_v4_<category>_<kind>.json file, by category."""
  out = {}
  for name in glob.glob(f'{root}/{kind}/{model}/**/*.json', recursive=True):
    match = re.search(rf'BFCL_v4_(.*)_{kind}\.json$', name)
    if match:
      with pathlib.Path(name).open(encoding='utf-8') as f:
        out[match.group(1)] = [json.loads(x) for x in f if x.strip()]
  return out


def overall(root: str, model: str) -> str:
  """BFCL's overall accuracy from data_overall.csv, or '?'."""
  mapping = model_config.MODEL_CONFIG_MAPPING
  by_display = {cfg.display_name: key for key, cfg in mapping.items()}
  csv_path = pathlib.Path(root) / 'score' / 'data_overall.csv'
  if csv_path.exists():
    with csv_path.open(encoding='utf-8') as f:
      for line in f:
        match = _OVERALL_LINE.match(line)
        key = by_display.get(match.group(2), '') if match else ''
        if key.replace('/', '_') == model:
          return match.group(1)
  return '?'


def flat_sum(value: object) -> float:
  """Sum a number or an arbitrarily nested list of numbers."""
  if isinstance(value, list):
    return sum(flat_sum(v) for v in value)
  return float(value) if isinstance(value, int | float) else 0.0


def loop_diagnostics(res: dict[str, list[dict]]) -> dict[str, int]:
  """Count multi-turn entries that hit the step limit, loop, or errored."""
  entries: list[dict] = []
  for category in SECTIONS['multi-turn'].split():
    entries.extend(res.get(category, []))
  errors = [e for e in entries if isinstance(e['result'], str)]
  repeats = 0
  for entry in entries:
    if entry in errors:
      continue
    for turn in entry['result']:
      calls = [json.dumps(s) for s in turn if isinstance(s, list)]
      if any(
        calls[i] == calls[i + 1] == calls[i + 2] for i in range(len(calls) - 2)
      ):
        repeats += 1
        break
  return {
    'entries': len(entries),
    'step_limit_hits': sum(
      'forced to quit' in json.dumps(e.get('inference_log', ''))
      for e in entries
      if e not in errors
    ),
    'repeated_call_entries': repeats,
    'inference_errors': len(errors),
  }


def main() -> None:
  """Print the section table, the diagnostics and the token total."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('root')
  parser.add_argument('model')
  parser.add_argument('--baseline', nargs=2, metavar=('ROOT', 'MODEL'))
  args = parser.parse_args()
  score = load(args.root, args.model, 'score')
  own = {c: rows[0]['accuracy'] * 100 for c, rows in score.items()}
  base_score = load(*args.baseline, 'score') if args.baseline else {}
  base = {c: rows[0]['accuracy'] * 100 for c, rows in base_score.items()}
  print(
    f'{"section":<24}{"score":>8}'
    + f'{"baseline":>10}{"delta":>8}' * bool(base)
  )
  for section, categories in SECTIONS.items():
    have = [c for c in categories.split() if c in own]
    if not have:
      continue
    mean = statistics.mean(own[c] for c in have)
    line = f'{section:<24}{mean:>8.1f}'
    have_base = [c for c in categories.split() if c in base]
    if have_base:
      base_mean = statistics.mean(base[c] for c in have_base)
      line += f'{base_mean:>10.1f}{mean - base_mean:>+8.1f}'
    print(line)
  print(f'{"overall (BFCL formula)":<24}{overall(args.root, args.model):>8}')
  res = load(args.root, args.model, 'result')
  print('multi-turn diagnostics:', json.dumps(loop_diagnostics(res)))
  # Memory prerequisite steps are not scored entries.
  rows: list[dict] = []
  for category, results in res.items():
    if 'prereq' not in category:
      rows.extend(results)
  tokens = sum(flat_sum(r.get('output_token_count', 0)) for r in rows)
  latency = [flat_sum(r.get('latency', 0)) for r in rows]
  if latency:
    print(
      f'output tokens (scored entries): {tokens / 1e6:.2f}M over '
      f'{len(latency)} entries; mean latency {statistics.mean(latency):.1f} s'
    )


if __name__ == '__main__':
  main()
