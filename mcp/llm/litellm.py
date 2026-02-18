"""LiteLLM backend via Anthropic Messages API endpoint."""

from __future__ import annotations

import os

from .base import LLMBackend


class LiteLLMLLM(LLMBackend):
    """LiteLLM backend using the Anthropic Messages API endpoint.

    Reads credentials from environment variables:
        LITELLM_API_KEY    — required
        LITELLM_BASE_URL   — required (e.g. https://your-litellm-host.example.com)

    Args:
        model_id: Model string passed to LiteLLM (e.g. "GCP/claude-4-sonnet").
    """

    def __init__(self, model_id: str = "GCP/claude-4-sonnet") -> None:
        self._api_key = os.environ["LITELLM_API_KEY"]
        base_url = os.environ["LITELLM_BASE_URL"]
        self._messages_url = base_url.rstrip("/") + "/v1/messages"
        self._model_id = model_id

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import requests

        resp = requests.post(
            self._messages_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model_id,
                "max_tokens": 2048,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
