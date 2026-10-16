"""Turn a sample_beam.py run into SFT rows.

For every task in the manifest, the first PASS sample (lowest k) becomes one
example: the user prompt, the model's own reasoning and its final answer,
rendered with Beam's chat template at reasoning effort "max" and tokenized with
the loss mask on the assistant tokens. Rows go to OUT/{train,val}.jsonl and
OUT/{train,val}.tok.jsonl according to the manifest split.

Tasks without a PASS sample are skipped and listed. Rows whose shifted length
exceeds --max-seq-tokens are excluded and listed, so the training job never
sees a row its packer would drop; set it to the recipe's packed_sequence_size.

usage: build_sft_rows.py --run RUN --prompts DIR --manifest dataset_manifest.json.gz --out DATA
       [--tokenizer DIR] [--max-seq-tokens 32768] [--ids a,b]
  --prompts    the prompts/ directory written by recreate_corpus.py, the one sample_beam.py read
  --tokenizer  local dir with tokenizer.json + chat_template.jinja (default: download the manifest's hf_id)
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from manifest_io import load_manifest, sha256

DEFAULT_MAX_SEQ_TOKENS = 32768


def first_pass(run_dir: pathlib.Path, tid: str) -> dict | None:
  """The lowest-k sample of a task with status PASS, or None."""
  d = run_dir / 'raw' / tid
  if not d.exists():
    return None
  for p in sorted(d.glob('s*.json'), key=lambda p: int(p.stem[1:])):
    rec = json.loads(p.read_text(encoding='utf-8'))
    if rec.get('status') == 'PASS':
      return rec
  return None


def find_prompt(
  prompts_dir: pathlib.Path, tid: str, expected_sha256: str
) -> str:
  """The prompt text a sample was drawn from, checked against the sample's recorded hash."""
  p = prompts_dir / f'{tid}.md'
  if not p.exists():
    raise FileNotFoundError(f'{tid}: prompt file {p} not found; pass --prompts')
  prompt = p.read_text(encoding='utf-8')
  if sha256(prompt) != expected_sha256:
    raise ValueError(
      f'{tid}: {p} differs from the prompt the sample was drawn from (sha256 mismatch)'
    )
  return prompt


def fits_budget(n_ids: int, max_seq_tokens: int) -> bool:
  """Whether a row of n_ids tokens fits the training budget after the next-token shift."""
  return n_ids - 1 <= max_seq_tokens


def load_tokenizer(tok_dir: pathlib.Path | None, spec: dict):  # noqa: ANN201
  """Beam's fast tokenizer with its chat template, both hash-checked against the manifest."""
  if tok_dir is None:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    tok_dir = pathlib.Path(
      snapshot_download(
        spec['hf_id'],
        allow_patterns=[
          'tokenizer.json',
          'chat_template.jinja',
          'tokenizer_config.json',
        ],
      )
    )
  for f, key in (
    ('tokenizer.json', 'tokenizer_json_sha256'),
    ('chat_template.jinja', 'chat_template_sha256'),
  ):
    got = sha256((tok_dir / f).read_bytes())
    if got != spec[key]:
      raise ValueError(
        f'{f}: sha256 {got} != manifest (different tokenizer or chat template)'
      )
  from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

  tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_dir / 'tokenizer.json'))
  tok.chat_template = (tok_dir / 'chat_template.jinja').read_text(
    encoding='utf-8'
  )
  return tok


def render(
  tok, prompt: str, reasoning: str, content: str, effort: str, eos: int
) -> tuple[list[int], list[int]]:
  """Token ids of the prompt alone and of prompt + assistant turn; the second must extend the first."""
  user = {'role': 'user', 'content': prompt}
  asst = {'role': 'assistant', 'thinking': reasoning, 'content': content}
  ids_of = lambda x: list(x['input_ids'] if hasattr(x, 'keys') else x)
  p = ids_of(
    tok.apply_chat_template(
      [user], tokenize=True, add_generation_prompt=True, reasoning_effort=effort
    )
  )
  q = ids_of(
    tok.apply_chat_template(
      [user, asst], tokenize=True, reasoning_effort=effort
    )
  )
  if q[: len(p)] != p or q[-1] != eos:
    raise ValueError(
      'render check failed: assistant turn does not extend the prompt'
    )
  return p, q


def main(argv: list[str] | None = None) -> int:
  """Build the SFT rows; returns the process exit code."""
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument('--run', required=True)
  ap.add_argument('--prompts', required=True)
  ap.add_argument('--manifest', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--tokenizer')
  ap.add_argument('--max-seq-tokens', type=int, default=DEFAULT_MAX_SEQ_TOKENS)
  ap.add_argument('--ids')
  a = ap.parse_args(argv)
  manifest = load_manifest(a.manifest)
  run_dir = pathlib.Path(a.run)
  prompts_dir = pathlib.Path(a.prompts)
  out = pathlib.Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  spec = manifest['sft']['tokenizer']
  tok = load_tokenizer(pathlib.Path(a.tokenizer) if a.tokenizer else None, spec)
  sel = sorted(a.ids.split(',')) if a.ids else sorted(manifest['tasks'])
  rows = collections.defaultdict(list)
  missing, overlong = [], []
  for tid in sel:
    rec = first_pass(run_dir, tid)
    if rec is None:
      missing.append(tid)
      continue
    prompt = find_prompt(prompts_dir, tid, rec['prompt_sha256'])
    resp = rec['response']
    try:
      p, q = render(
        tok,
        prompt,
        resp['reasoning'],
        resp['content'],
        spec['reasoning_effort'],
        spec['eos_id'],
      )
    except ValueError as e:
      raise ValueError(f'{tid}: {e}') from e
    if not fits_budget(len(q), a.max_seq_tokens):
      overlong.append((tid, len(q) - 1))
      continue
    task = manifest['tasks'][tid]
    row = dict(
      id=tid,
      cls=task['cls'],
      k=rec['k'],
      prompt=prompt,
      reasoning=resp['reasoning'],
      content=resp['content'],
    )
    trow = dict(
      id=tid, input_ids=q, loss_mask=[0] * len(p) + [1] * (len(q) - len(p))
    )
    rows[task['split']].append((row, trow))
  for split in ('train', 'val'):
    with (
      open(out / f'{split}.jsonl', 'w', encoding='utf-8') as fm,
      open(out / f'{split}.tok.jsonl', 'w', encoding='utf-8') as ft,
    ):
      for row, trow in rows[split]:
        fm.write(json.dumps(row) + '\n')
        ft.write(json.dumps(trow) + '\n')
    supervised = sum(sum(t['loss_mask']) for _, t in rows[split])
    print(f'{split}: {len(rows[split])} rows, {supervised} supervised tokens')
  print(f'tasks without a PASS sample: {len(missing)} {missing[:10]}')
  print(
    f'rows over {a.max_seq_tokens} tokens, excluded: {len(overlong)} '
    f'{[f"{tid} ({n} tokens)" for tid, n in overlong[:10]]}'
  )
  return 0


if __name__ == '__main__':
  sys.exit(main())
