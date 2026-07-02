"""Gemini provider (alternative demo path). Implements LLMProvider.

Uses the `google-genai` client. Same contract as the OpenAI path — the RAG
chain never sees the difference; only `LLM_PROVIDER=gemini` selects it.

Temperature 0: a document-grounded QA assistant should be faithful to the
retrieved context, not creative.
"""

from __future__ import annotations

import logging

from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        # Imported here so the google-genai SDK is only required on this path.
        from google import genai

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return (response.text or "").strip()
