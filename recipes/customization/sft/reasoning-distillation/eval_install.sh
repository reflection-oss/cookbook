#!/usr/bin/env bash
# Install the Berkeley Function Calling Leaderboard (BFCL v4) at a pinned
# commit, add the Gemma 4 handler and register this recipe's six models.
#
#   BFCL_DIR=/data/gorilla bash eval_install.sh
set -euo pipefail

BFCL_DIR="${BFCL_DIR:-$HOME/gorilla}"
COMMIT=6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$BFCL_DIR/.git" ]]; then
  git clone https://github.com/ShishirPatil/gorilla.git "$BFCL_DIR"
fi
git -C "$BFCL_DIR" fetch --quiet origin
git -C "$BFCL_DIR" checkout --quiet "$COMMIT"
LEADERBOARD="$BFCL_DIR/berkeley-function-call-leaderboard"
pip install -e "$LEADERBOARD"
cp "$HERE/bfcl_gemma4.py" \
  "$LEADERBOARD/bfcl_eval/model_handler/api_inference/gemma4.py"

# Append a marked block to the registry; running again replaces it.
python - "$LEADERBOARD/bfcl_eval/constants/model_config.py" <<'PY'
import pathlib
import sys

BEGIN = '# --- gemma4 tool-reasoning recipe: begin ---'
END = '# --- gemma4 tool-reasoning recipe: end ---'
TEMPLATE = """    {key!r}: ModelConfig(
        model_name={served!r},
        display_name={display!r},
        url='https://huggingface.co/google/gemma-4-{up}-it',
        org='Google',
        license='Gemma Terms of Use',
        model_handler=Gemma4ThinkingKeepHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    ),
"""
entries = []
for size in ('12b', 'e4b', 'e2b'):
  up = size.upper()
  entries.append(TEMPLATE.format(
    key=f'google/gemma-4-{size}-it-thinking-keephist-FC',
    served=f'google/gemma-4-{size}-it',
    display=f'Gemma-4-{size}-it (FC) (Thinking + thoughts kept in history)',
    up=up,
  ))
  entries.append(TEMPLATE.format(
    key=f'google/gemma-4-{size}-it-tool-reasoning-FC',
    served=f'gemma-4-{size}-it-tool-reasoning',
    display=f'Gemma-4-{size}-it + tool-reasoning recipe (FC) '
    '(Thinking + thoughts kept in history)',
    up=up,
  ))
block = (
  f'{BEGIN}\nfrom bfcl_eval.model_handler.api_inference.gemma4 import (\n'
  '    Gemma4ThinkingKeepHandler,\n)\n\nMODEL_CONFIG_MAPPING.update({\n'
  + ''.join(entries) + f'}})\n{END}\n'
)
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
if BEGIN in text:
  head, rest = text.split(BEGIN, 1)
  text = head + rest.split(END, 1)[1].lstrip('\n')
path.write_text(text.rstrip('\n') + '\n\n' + block, encoding='utf-8')
print(f'registered {len(entries)} models in {path}')
PY
echo "BFCL ready at $LEADERBOARD (commit $COMMIT)"
