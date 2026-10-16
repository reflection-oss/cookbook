"""BFCL handler for Gemma 4 served behind an OpenAI-compatible endpoint.

Installed by eval_install.sh as `bfcl_eval/model_handler/api_inference/
gemma4.py`. It starts no server and loads no tokenizer: the server applies
the chat template and its tool-call and reasoning parsers. The endpoint
comes from the `.env` of the BFCL project root: `REMOTE_OPENAI_BASE_URL`
(default `http://localhost:<LOCAL_SERVER_PORT>/v1`), `REMOTE_OPENAI_API_KEY`.
"""

from __future__ import annotations

import os
import re
import time

import openai
import tenacity
from bfcl_eval.constants import enums, eval_config
from bfcl_eval.model_handler import utils as handler_utils
from bfcl_eval.model_handler.api_inference import openai_completion

_CONTEXT_OVERFLOW_RE = re.compile(
  r'maximum context length of (\d+) tokens.*?'
  r'(\d+) tokens from the input messages',
  re.DOTALL,
)


class OpenAICompatibleHandler(openai_completion.OpenAICompletionsHandler):
  """Chat Completions handler for a pre-existing endpoint."""

  # Hard cap on generated tokens per request, against runaway generations.
  max_tokens: int | None = 8192
  # Extra `chat_template_kwargs` forwarded to the server.
  chat_template_kwargs: dict[str, object] | None = None
  # Keep `reasoning_content` on assistant messages fed back as history.
  keep_reasoning_in_history: bool = False
  # (base_url, model_name) -> served model name, cached across categories.
  _served_model_name_cache: dict[tuple[str, str], str] = {}

  def __init__(
    self,
    model_name: str,
    temperature: float,
    registry_name: str,
    is_fc_model: bool,
    **kwargs: object,
  ) -> None:
    local_endpoint = os.getenv('LOCAL_SERVER_ENDPOINT', 'localhost')
    local_port = os.getenv('LOCAL_SERVER_PORT', eval_config.LOCAL_SERVER_PORT)
    self.base_url = os.getenv(
      'REMOTE_OPENAI_BASE_URL', f'http://{local_endpoint}:{local_port}/v1'
    )
    self.api_key = os.getenv('REMOTE_OPENAI_API_KEY', 'EMPTY')
    super().__init__(
      model_name, temperature, registry_name, is_fc_model, **kwargs
    )
    self.model_style = enums.ModelStyle.OPENAI_COMPLETIONS
    cache_key = (self.base_url, self.model_name)
    if cache_key not in self._served_model_name_cache:
      self._served_model_name_cache[cache_key] = self._served_name()
    self.served_model_name = self._served_model_name_cache[cache_key]

  def _build_client_kwargs(self) -> dict[str, object]:
    # Long entries take a while; retries are handled by generate_with_backoff.
    return {
      'base_url': self.base_url,
      'api_key': self.api_key,
      'timeout': 3600,
      'max_retries': 0,
    }

  def _served_name(self) -> str:
    """The server's name for this model, matched case-insensitively.

    The registry entry names the model the benchmark intends to score, so a
    server that does not serve it stops the run instead of scoring whatever
    it does serve.
    """
    tag = f'[{type(self).__name__}]'
    try:
      served = [model.id for model in self.client.models.list().data]
    except Exception as exc:  # noqa: BLE001 - any failure means "send as-is"
      print(f'{tag} Could not list models at {self.base_url} ({exc}).')
      return self.model_name
    if self.model_name in served or not served:
      return self.model_name
    by_lower = {name.lower(): name for name in served}
    resolved = by_lower.get(self.model_name.lower())
    if resolved is None:
      raise RuntimeError(
        f"{tag} '{self.model_name}' is not served at {self.base_url}; "
        f'the server lists {served}. Serve the model under that name.'
      )
    print(f"{tag} Using served model name '{resolved}'.")
    return resolved

  @handler_utils.retry_with_backoff(
    error_type=[
      openai.RateLimitError,
      openai.APIConnectionError,
      openai.APITimeoutError,
      openai.InternalServerError,
    ],
    stop=tenacity.stop_after_attempt(8),
    reraise=True,
  )
  def generate_with_backoff(self, **kwargs: object) -> tuple[object, float]:
    """Send one request; shrink `max_tokens` when the context is nearly full."""
    start_time = time.time()
    try:
      api_response = self.client.chat.completions.create(**kwargs)
    except openai.BadRequestError as exc:
      match = _CONTEXT_OVERFLOW_RE.search(str(exc))
      if not match or 'max_tokens' not in kwargs:
        raise
      room = int(match.group(1)) - int(match.group(2)) - 16
      if room < 64:
        raise
      kwargs = {**kwargs, 'max_tokens': min(kwargs['max_tokens'], room)}
      api_response = self.client.chat.completions.create(**kwargs)
    return api_response, time.time() - start_time

  def _query_FC(self, inference_data: dict) -> tuple[object, float]:  # noqa: N802
    messages: list[dict] = inference_data['message']
    tools = inference_data['tools']
    inference_data['inference_input_log'] = {
      'message': repr(messages),
      'tools': tools,
    }
    kwargs: dict[str, object] = {
      'messages': messages,
      'model': self.served_model_name,
      'temperature': self.temperature,
    }
    if self.max_tokens is not None:
      kwargs['max_tokens'] = self.max_tokens
    if self.chat_template_kwargs:
      kwargs['extra_body'] = {'chat_template_kwargs': self.chat_template_kwargs}
    if tools:
      kwargs['tools'] = tools
    return self.generate_with_backoff(**kwargs)

  def _parse_query_response_FC(self, api_response: object) -> dict:  # noqa: N802
    response_data = super()._parse_query_response_FC(api_response)
    # Servers with a reasoning parser return the thinking in
    # `reasoning_content`; it is logged, and kept in the chat history only
    # when the model's template is designed for that.
    self._add_reasoning_content_if_available_FC(api_response, response_data)
    if self.keep_reasoning_in_history:
      reasoning = response_data.get('reasoning_content')
      history = response_data.get('model_responses_message_for_chat_history')
      if reasoning and isinstance(history, dict):
        history['reasoning_content'] = reasoning
    return response_data


class Gemma4ThinkingKeepHandler(OpenAICompatibleHandler):
  """Thinking on, and the thoughts of the current round kept in the history.

  Gemma 4's template renders those and drops earlier rounds. This is the
  convention every number in the recipe is measured under.
  """

  chat_template_kwargs = {'enable_thinking': True}
  keep_reasoning_in_history = True
  max_tokens = 8192
