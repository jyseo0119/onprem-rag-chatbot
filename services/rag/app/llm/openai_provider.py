"""OpenAI provider (default demo path). Implements LLMProvider.

Chat Completions with temperature 0 — a document-grounded QA assistant should be
deterministic and faithful to the retrieved context, not creative.
"""

from __future__ import annotations

import logging

from app.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        # Imported here so the openai SDK is only required when this path is used.
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
