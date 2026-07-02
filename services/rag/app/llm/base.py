"""LLM provider abstraction — the pivot of the hybrid demo/on-prem story.

The RAG chain depends only on this interface. Switching from an API LLM to
self-hosted vLLM+Qwen2.5 is a matter of selecting a different implementation
via `LLM_PROVIDER`, not rewriting the chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import settings


class LLMProvider(ABC):
    """Minimal contract every LLM backend must satisfy."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model's completion for a single prompt."""
        raise NotImplementedError


def get_provider(name: str | None = None) -> LLMProvider:
    """Factory: resolve `LLM_PROVIDER` to a concrete provider.

    This is the single pivot of the hybrid demo/on-prem story — the RAG chain
    never names a concrete backend. `openai` is the demo default, `gemini` an
    alternative API path, and `vllm` the self-hosted on-prem path (Qwen2.5).

    Imports are local so pulling in one provider's SDK never forces the others.
    """
    name = (name or settings.llm_provider).lower()

    if name == "openai":
        from app.llm.openai_provider import OpenAIProvider

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)

    if name == "gemini":
        from app.llm.gemini_provider import GeminiProvider

        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return GeminiProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)

    if name == "vllm":
        from app.llm.vllm_provider import VLLMProvider

        # No key check: vLLM is self-hosted and does not authenticate by default.
        return VLLMProvider(
            base_url=settings.vllm_base_url,
            model=settings.vllm_model,
            api_key=settings.vllm_api_key,
        )

    raise ValueError(f"unknown LLM_PROVIDER '{name}' (expected: openai | gemini | vllm)")
