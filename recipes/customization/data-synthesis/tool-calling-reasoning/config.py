"""Constants and defaults of the recipe: the values of our run.

Edit this file to change a run. The command line only chooses inputs,
outputs, the thought style and the concurrency.
"""

from __future__ import annotations

import dataclasses

TOOLACE_REPO = 'Team-ACE/ToolACE'  # Apache-2.0
TOOLACE_REVISION = '6bda777c88d21e5a204703c1ee45597a8fa4f734'
NEMOTRON_REPO = 'nvidia/Nemotron-SFT-Agentic-v2'  # CC-BY-4.0
NEMOTRON_REVISION = '7c804833427f633ccd53b582dbf02525fd680f78'
NEMOTRON_FILE = 'data/interactive_agent.jsonl'

API_BASE = 'https://api.reflection.ai/openai/v1'
MODEL = 'Beam-1.0-501B-A23B'

# The thought-length instruction given to the teacher. "purpose" is the
# recipe; "verbose" and "concise" are the ablation arms of the write-up.
THOUGHT_STYLES = {
  'purpose': (
    'Write the thought the way a careful engineer thinks before acting: '
    'enough to justify the decision and each parameter value, and nothing '
    'that does not change the decision.'
  ),
  'verbose': 'Think out loud and maximize your tokens when adding CoT.',
  'concise': (
    'Keep every thought SHORT: 2 to 4 sentences, at most about 70 words. '
    'Say what the user needs, the decision (which function and why, or why '
    'none / what to ask), and the key parameter choices. Do not restate the '
    'conversation, do not recite the policy, do not list alternatives you '
    'are not using.'
  ),
}

# A thought shorter than MIN_THOUGHT_CHARS is rejected at the teacher; one
# longer than MAX_THOUGHT_CHARS drops its conversation at assembly.
MIN_THOUGHT_CHARS = 60
MAX_THOUGHT_CHARS = 6000


@dataclasses.dataclass(frozen=True)
class TeacherConfig:
  """One streamed request per conversation."""

  thought_style: str = 'purpose'
  temperature: float = 0.2
  top_p: float = 0.95
  max_tokens: int = 16_384  # doubled when a reply is cut off, up to the cap
  max_tokens_cap: int = 65_536
  reasoning_effort: str = 'max'
  timeout_s: float = 1800.0
  retries: int = 8
  workers: int = 16


# ToolACE: every conversation with a tool result, plus a seeded sample of
# single-call conversations; conversations without a call are skipped.
TOOLACE_SINGLE_CALL_SAMPLE = 840
TOOLACE_SEED = 11

# Nemotron: the first POOL_ROWS of the file, the look-up-then-act rule, then
# a seeded sample of N conversations of at most MAX_CHARS characters each.
NEMOTRON_POOL_ROWS = 52_828
NEMOTRON_N = 3000
NEMOTRON_SEED = 27
NEMOTRON_MAX_CHARS = 40_000

# Audit: a thought needs at least MIN_WORDS words and at most MAX_WORDS.
AUDIT_MIN_WORDS = 12
AUDIT_MAX_WORDS = 600

# Assemble: thought flags that drop the whole conversation.
DROP_FLAGS = frozenset({'empty', 'meta_reference', 'too_long', 'tool_tokens'})

# Mix: the shuffle seed and the prefix that keeps the two id spaces apart.
MIX_SEED = 83
NEMOTRON_ID_PREFIX = 'sd_'
