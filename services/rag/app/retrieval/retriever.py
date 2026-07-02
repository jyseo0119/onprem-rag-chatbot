"""Retrieval: embed the query, search Qdrant, apply top-k + score threshold.

The score threshold is the guardrail against confident hallucination: when no
chunk clears RETRIEVAL_SCORE_THRESHOLD, the chain answers "not found in the
documents" rather than inventing one. Tuning notes in docs/decisions.md.

The embedder and store are injected so the API server (stage 3) can share a
single loaded BGE-m3 model (the ~2.3 GB load is the expensive part) across
requests; `from_settings()` is the convenience path for the CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import settings
from app.embedding.bge_m3 import BGEM3Embedder
from app.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned for a query, carrying its citation metadata and score."""

    text: str
    source: str
    page: int
    chunk_index: int
    score: float


class Retriever:
    def __init__(
        self,
        embedder: BGEM3Embedder,
        store: QdrantStore,
        top_k: int,
        score_threshold: float,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

    @classmethod
    def from_settings(cls) -> Retriever:
        """Build a retriever from env-driven settings (CLI / single-process use)."""
        embedder = BGEM3Embedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
        store = QdrantStore(url=settings.qdrant_url, collection=settings.qdrant_collection)
        return cls(
            embedder=embedder,
            store=store,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Embed the query and return the chunks that clear the score threshold."""
        query = query.strip()
        if not query:
            return []

        query_vector = self.embedder.embed_query(query)
        hits = self.store.search(
            query_vector,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )
        logger.debug("query=%r -> %d hit(s) above %.2f", query, len(hits), self.score_threshold)

        return [
            RetrievedChunk(
                text=hit.payload.get("text", ""),
                source=hit.payload.get("source", "unknown"),
                page=hit.payload.get("page", 0),
                chunk_index=hit.payload.get("chunk_index", 0),
                score=hit.score,
            )
            for hit in hits
        ]
