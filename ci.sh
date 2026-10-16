#!/usr/bin/env bash
# Check that every recipe is complete and every script parses. Needs no
# network, credentials, or GPU: smoke tests that call a live endpoint run
# from each recipe's folder, not here.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/cookbook-pycache"

fail=0

# Every folder that holds code is a recipe and needs a README and a smoke test.
while IFS= read -r -d '' dir; do
  if compgen -G "$dir/*.py" >/dev/null || compgen -G "$dir/*.sh" >/dev/null || compgen -G "$dir/*.ipynb" >/dev/null; then
    [[ -f "$dir/README.md" ]] || {
      echo "missing README.md: $dir" >&2
      fail=1
    }
    if [[ ! -f "$dir/smoke_test.py" && ! -f "$dir/smoke_test.sh" ]]; then
      echo "missing smoke test: $dir" >&2
      fail=1
    fi
  fi
done < <(find recipes -type d -print0)

# Python parses.
python3 -m compileall -q recipes skills || fail=1

# Shell scripts parse.
while IFS= read -r -d '' script; do
  bash -n "$script" || {
    echo "syntax error: $script" >&2
    fail=1
  }
done < <(find recipes -name '*.sh' -print0)

if ((fail)); then
  echo 'ci.sh: checks failed' >&2
  exit 1
fi
echo 'ci.sh: ok'
