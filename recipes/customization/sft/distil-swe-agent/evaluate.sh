#!/usr/bin/env bash
# Run mini-swe-agent against the served model on every Harbor task in
# TASKS_DIR, K attempts each, and print pass@1..4. The agent config is the
# plain one: temperature 1.0, 16k output tokens, thinking on and kept.
set -euo pipefail
cd "$(dirname "$0")"

TASKS_DIR="${TASKS_DIR:?folder of Harbor tasks}"
SERVED_NAME="${SERVED_NAME:-gemma-4-e4b-distil}"
JOB="${JOB:-$SERVED_NAME}"
K="${K:-4}"
PARALLEL="${PARALLEL:-24}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:30000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
export HARBOR_TELEMETRY=0
CONFIG='{"agent":{"step_limit":250},"environment":{"timeout":300},"model":{"model_kwargs":{"timeout":3600,"temperature":1.0,"max_tokens":16384,"extra_body":{"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}}}}}'

harbor run -e docker -p "$TASKS_DIR" -a mini-swe-agent -m "openai/$SERVED_NAME" \
  -n "$PARALLEL" -k "$K" -o jobs --job-name "$JOB" --quiet \
  --agent-setup-timeout-multiplier 3 --ak version=2.4.6 --ak "config=$CONFIG"

# pass@k: the unbiased estimator over the n attempts of each environment.
python - "jobs/$JOB" <<'PY'
import collections, json, math, pathlib, sys
rewards = collections.defaultdict(list)
for result in sorted(pathlib.Path(sys.argv[1]).glob('task_*__*/result.json')):
  verifier = json.loads(result.read_text()).get('verifier_result') or {}
  reward = (verifier.get('rewards') or {}).get('reward') or 0
  rewards[result.parent.name.split('__')[0]].append(reward >= 1)
def pass_at_k(n, c, k):
  return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)
attempts = min(map(len, rewards.values()), default=0)
print(f'environments={len(rewards)} attempts={sum(map(len, rewards.values()))}')
for k in range(1, attempts + 1):  # only the k every environment has attempts for
  score = [pass_at_k(len(r), sum(r), k) for r in rewards.values()]
  print(f'pass@{k}={100 * sum(score) / len(score):.1f}%', end='  ')
print()
PY
