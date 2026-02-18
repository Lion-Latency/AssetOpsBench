"""WatsonX LLM backend."""

from __future__ import annotations

import os
import time

from .base import LLMBackend


class WatsonXLLM(LLMBackend):
    """WatsonX LLM backend using the WatsonX REST API directly.

    Uses `requests` (already a core dependency) instead of the
    `ibm_watsonx_ai` SDK, which is incompatible with Python 3.14.

    Reads credentials from environment variables:
        WATSONX_APIKEY       — required
        WATSONX_PROJECT_ID   — required
        WATSONX_URL          — optional (defaults to us-south)

    Args:
        model_id: WatsonX model ID string, e.g.
                  "meta-llama/llama-4-maverick-17b-128e-instruct-fp8".
    """

    _IAM_URL = "https://iam.cloud.ibm.com/identity/token"
    _GENERATION_PATH = "/ml/v1/text/generation?version=2023-05-29"

    def __init__(self, model_id: str = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8") -> None:
        self._api_key = os.environ["WATSONX_APIKEY"]
        self._project_id = os.environ["WATSONX_PROJECT_ID"]
        base_url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self._generation_url = base_url.rstrip("/") + self._GENERATION_PATH
        self._model_name = model_id
        self._token: str | None = None
        self._token_expiry: float = 0.0

    def _get_token(self) -> str:
        """Return a valid IAM bearer token, refreshing if within 60 s of expiry."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        import requests

        resp = requests.post(
            self._IAM_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=(
                "grant_type=urn:ibm:params:oauth:grant-type:apikey"
                f"&apikey={self._api_key}"
            ),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        return self._token

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import requests

        resp = requests.post(
            self._generation_url,
            headers={
                "Authorization": f"Bearer {self._get_token()}",
                "Content-Type": "application/json",
            },
            json={
                "model_id": self._model_name,
                "input": prompt,
                "parameters": {"max_new_tokens": 2048, "temperature": temperature},
                "project_id": self._project_id,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["results"][0]["generated_text"]
