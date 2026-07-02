"""vLLM provider (on-prem path, Qwen2.5). Implements LLMProvider.

vLLM exposes an OpenAI-compatible API, so this is a thin subclass of the OpenAI
provider pointed at `VLLM_BASE_URL`. That is what makes the API -> on-prem
switch a config change, not a rewrite: identical Chat Completions call, identical
prompt assembly, identical RAG chain — only the endpoint (and the model weights
behind it) differ.

vLLM does not check the API key, but the OpenAI client requires a non-empty one,
so we default to "EMPTY" (the vLLM convention).
"""

from __future__ import annotations

from app.llm.openai_provider import OpenAIProvider


class VLLMProvider(OpenAIProvider):
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        super().__init__(api_key=api_key or "EMPTY", model=model, base_url=base_url)
