"""
OpenRouter client for every generative LLM call in SimCheck.

OpenRouter exposes an OpenAI-compatible API, so the openai SDK is used with a
different base URL. One key covers Claude (rating escalation/explanations)
and the LLM-visibility probes.

Jev classification does NOT go through here: OpenRouter only offers
typesafe/jev-router (a model router), not the typed classifier.
"""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI, OpenAIError


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_TIMEOUT_SECONDS = 60

# Model IDs in one place so they can be swapped without code changes
MODELS = {
    "rater": "anthropic/claude-sonnet-5-5",
    "probe_perplexity": "perplexity/sonar",
    "probe_gpt": "openai/gpt-5.6-luna",
}

# Shown to OpenRouter for attribution/rate-limit grouping
_APP_HEADERS = {"X-Title": "SimCheck"}


class LLMError(Exception):
    """Raised when an OpenRouter call fails or returns unusable output."""


class OpenRouterClient:
    """
    Thin wrapper over the openai SDK pointed at OpenRouter.

    Args:
        api_key: OpenRouter API key (from simcheck.config.load_api_keys)
        client: Optional pre-built OpenAI-compatible client (tests inject a fake)
    """

    def __init__(self, api_key: Optional[str] = None, client=None):
        if client is None:
            if not api_key:
                raise LLMError("OpenRouter API key is not configured.")
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                timeout=LLM_TIMEOUT_SECONDS,
                default_headers=_APP_HEADERS,
            )
        self._client = client

    def __repr__(self) -> str:
        return "OpenRouterClient()"  # never expose the key

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: dict,
        schema_name: str = "result",
        max_tokens: int = 2000,
    ) -> tuple:
        """
        Run a chat completion constrained to a JSON schema.

        Args:
            model: OpenRouter model ID
            system: System prompt
            user: User message
            schema: JSON schema (strict mode: all properties required)
            schema_name: Name for the schema
            max_tokens: Completion token cap

        Returns:
            (parsed JSON dict, cost in USD or None)

        Raises:
            LLMError: On API failure, truncation, or invalid JSON
        """
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                max_tokens=max_tokens,
                # Only route to providers that honor response_format
                extra_body={"provider": {"require_parameters": True}},
            )
        except OpenAIError as e:
            raise LLMError(f"OpenRouter call to {model} failed: {e}") from e

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(f"{model} response truncated at {max_tokens} tokens.")
        try:
            parsed = json.loads(choice.message.content or "")
        except json.JSONDecodeError as e:
            raise LLMError(f"{model} returned invalid JSON: {e}") from e

        usage = getattr(response, "usage", None)
        cost = getattr(usage, "cost", None) if usage is not None else None
        return parsed, cost
