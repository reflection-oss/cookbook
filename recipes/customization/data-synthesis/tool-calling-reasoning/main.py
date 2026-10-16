"""Build tool-calling training data with Beam-written reasoning.

Stages, each a subcommand that reads and writes plain JSONL:

  select-toolace   pick the ToolACE conversations to enrich
  select-nemotron  pick state-dependent Nemotron customer-service conversations
  enrich           have Beam add a thought to every assistant turn (resumable)
  audit            check the thoughts and list the turns to drop
  assemble         build OpenAI-style messages rows with reasoning_content
  mix              shuffle the two halves into one training file

`run.sh` chains them; the settings live in config.py; see README.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import assemble
import audit
import config
import dotenv
import formats
import mix
import nemotron
import teacher
import toolace


def _spread[T](items: list[T], limit: int | None) -> list[T]:
  """The first `limit` items at even spacing, or all of them."""
  if not limit:
    return items
  return items[:: max(1, len(items) // limit)][:limit]


def _select_toolace(a: argparse.Namespace) -> None:
  rows = toolace.load_toolace()
  keep, breakdown = toolace.select_rows(rows)
  print(f'toolace: {len(keep)} rows selected; kinds={breakdown}')
  n = formats.write_jsonl(a.out, (rows[i] for i in _spread(keep, a.limit)))
  print(f'toolace: wrote {n} rows -> {a.out}')


def _select_nemotron(a: argparse.Namespace) -> None:
  pool_path = pathlib.Path(a.pool)
  if pool_path.exists():
    pool = formats.read_jsonl(str(pool_path))
    print(f'nemotron: pool of {len(pool)} rows from {pool_path}')
  else:
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool = list(nemotron.stream_interactive_agent(config.NEMOTRON_POOL_ROWS))
    formats.write_jsonl(str(pool_path), pool)
    print(f'nemotron: streamed a pool of {len(pool)} rows -> {pool_path}')
  selected, stats = nemotron.select_state_dependent(pool)
  print('nemotron:', json.dumps(stats))
  converted: list[formats.Row] = []
  for raw in _spread(selected, a.limit):
    # The dataset's uuid keeps the id stable across selections and limits.
    row = nemotron.to_toolace_row(raw, f'nemotron_cs_{nemotron.uuid_of(raw)}')
    if row is not None:
      converted.append(row)
  formats.write_jsonl(a.out, converted)
  print(f'nemotron: wrote {len(converted)} rows -> {a.out}')


def _enrich(a: argparse.Namespace) -> None:
  cfg = config.TeacherConfig(thought_style=a.thought_style, workers=a.workers)
  teacher.enrich(formats.read_jsonl(a.rows), a.out, cfg)


def _audit(a: argparse.Namespace) -> None:
  rows = formats.read_jsonl(a.rows)
  records, failed, unknown = audit.load_records(a.records, rows)
  summary, bad_turns = audit.audit(rows, records)
  summary['failed_conversations'] = failed
  summary['records_not_in_rows'] = unknown
  print(json.dumps(summary, indent=2))
  prefix = pathlib.Path(a.out_prefix)
  prefix.parent.mkdir(parents=True, exist_ok=True)
  prefix.with_name(prefix.name + '_summary.json').write_text(
    json.dumps(summary, indent=2), encoding='utf-8'
  )
  prefix.with_name(prefix.name + '_bad_turns.json').write_text(
    json.dumps([{'row': r, 'msg': m} for r, m in bad_turns]), encoding='utf-8'
  )


def _assemble(a: argparse.Namespace) -> None:
  rows = formats.read_jsonl(a.rows)
  records, _, unknown = audit.load_records(a.records, rows)
  out, stats = assemble.assemble(
    rows,
    records,
    assemble.load_excluded(a.exclude),
    system_text=a.system_text,
    no_ask=a.no_ask,
  )
  if unknown:
    stats['records_not_in_rows'] = unknown
  formats.write_jsonl(a.out, out)
  print(json.dumps(stats, indent=2))


def _mix(a: argparse.Namespace) -> None:
  print(json.dumps(mix.mix(a.toolace, a.nemotron, a.out), indent=2))


def build_parser() -> argparse.ArgumentParser:
  """The command line interface."""
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  sub = p.add_subparsers(dest='cmd', required=True)
  limit_help = 'keep N conversations spread evenly over the selection'

  s = sub.add_parser('select-toolace', help='pick the ToolACE conversations')
  s.add_argument('--out', required=True)
  s.add_argument('--limit', type=int, help=limit_help)
  s.set_defaults(fn=_select_toolace)

  s = sub.add_parser('select-nemotron', help='pick Nemotron conversations')
  s.add_argument('--out', required=True)
  s.add_argument(
    '--pool',
    default='work/nemotron_pool.jsonl',
    help='cached pool; streamed from the Hub when absent',
  )
  s.add_argument('--limit', type=int, help=limit_help)
  s.set_defaults(fn=_select_nemotron)

  s = sub.add_parser(
    'enrich', help='add a thought to every assistant turn (resumable)'
  )
  s.add_argument('--rows', required=True)
  s.add_argument('--out', required=True)
  s.add_argument(
    '--thought-style', choices=list(config.THOUGHT_STYLES), default='purpose'
  )
  s.add_argument('--workers', type=int, default=config.TeacherConfig.workers)
  s.set_defaults(fn=_enrich)

  s = sub.add_parser('audit', help='check the thoughts; list the turns to drop')
  s.add_argument('--rows', required=True)
  s.add_argument('--records', required=True)
  s.add_argument('--out-prefix', required=True)
  s.set_defaults(fn=_audit)

  s = sub.add_parser(
    'assemble', help='build messages rows with reasoning_content'
  )
  s.add_argument('--rows', required=True)
  s.add_argument('--records', required=True)
  s.add_argument('--out', required=True)
  s.add_argument('--exclude', help='<audit prefix>_bad_turns.json')
  s.add_argument(
    '--system-text',
    choices=['none', 'toolace'],
    default='none',
    help='toolace: keep the policy text as a system message (Nemotron)',
  )
  s.add_argument(
    '--no-ask',
    action='store_true',
    help='drop conversations with a clarification or refusal turn (ToolACE)',
  )
  s.set_defaults(fn=_assemble)

  s = sub.add_parser('mix', help='shuffle the two halves into one file')
  s.add_argument('--toolace', required=True)
  s.add_argument('--nemotron', required=True)
  s.add_argument('--out', required=True)
  s.set_defaults(fn=_mix)
  return p


def main() -> None:
  """Parse the command line and run one stage."""
  dotenv.load_dotenv()
  a = build_parser().parse_args()
  a.fn(a)


if __name__ == '__main__':
  main()
