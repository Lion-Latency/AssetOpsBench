"""LLM backend abstractions for the plan-execute orchestrator."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate text given a prompt."""
        ...


class WatsonXLLM(LLMBackend):
    """WatsonX LLM backend using ibm-watsonx-ai.

    Reads credentials from environment variables:
        WATSONX_APIKEY       — required
        WATSONX_PROJECT_ID   — required
        WATSONX_URL          — optional (defaults to us-south)

    Args:
        model_id: Integer model ID following the project convention.
                  16 → llama-4-maverick, 19 → granite-3-3-8b.
    """

    MODEL_MAP: dict[int, str] = {
        16: "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        19: "ibm/granite-3-3-8b-instruct",
    }

    def __init__(self, model_id: int = 16) -> None:
        try:
            from ibm_watsonx_ai import APIClient, Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
        except ImportError as exc:
            raise ImportError(
                "ibm-watsonx-ai is required for WatsonXLLM. "
                "Install it with: pip install ibm-watsonx-ai"
            ) from exc

        api_key = os.environ["WATSONX_APIKEY"]
        project_id = os.environ["WATSONX_PROJECT_ID"]
        url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        model_name = self.MODEL_MAP.get(model_id, self.MODEL_MAP[16])

        credentials = Credentials(api_key=api_key, url=url)
        self._model = ModelInference(
            model_id=model_name,
            api_client=APIClient(credentials, project_id=project_id),
        )

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._model.generate_text(
            prompt=prompt,
            params={"temperature": temperature, "max_new_tokens": 2048},
        )
