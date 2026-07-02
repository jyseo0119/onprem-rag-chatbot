"""Central settings, loaded from environment / .env.

Every tunable in the pipeline is surfaced here so behaviour can be changed
without touching code — see `.env.example` for the documented knobs.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider: "openai" | "gemini" | "vllm"
    llm_provider: str = "openai"

    # API providers
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # On-prem (vLLM, OpenAI-compatible)
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: str = "EMPTY"

    # Embeddings (BGE-m3)
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "onprem_docs"

    # Retrieval tuning
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.4

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # OCR (scanned PDFs, optional path — needs `uv sync --extra ocr`)
    enable_ocr: bool = False
    ocr_lang: str = "en"


settings = Settings()
