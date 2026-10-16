"""Smoke test: the generator scripts parse (they need torch/triton/pyarrow to run, which this check does not require)."""

import ast
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
  """Parse every generator script and check the expected entry points exist."""
  wanted = {
    'gen_corpus.py': {'gen_family_p', 'gen_family_k', 'main'},
    'ops_library.py': set(),
    'select_kernelbook.py': set(),
  }
  for name, funcs in wanted.items():
    tree = ast.parse((HERE / name).read_text(encoding='utf-8'), filename=name)
    found = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    missing = funcs - found
    assert not missing, f'{name}: missing {sorted(missing)}'
  print(f'ok: {len(wanted)} generator scripts parse')


if __name__ == '__main__':
  main()
