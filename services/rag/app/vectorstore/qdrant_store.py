"""Qdrant vector store wrapper.

Collection lifecycle (create/recreate), chunk upsert with payload metadata
(source filename, page, text), and similarity search. Payload carries what the
citation UI needs to attribute answers back to documents.

Point ids are deterministic (uuid5 of source|page|chunk_index) so re-ingesting
the same document overwrites its points instead of duplicating them — ingestion
is idempotent, which matters the moment someone re-runs it on an updated PDF.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.ingest.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One similarity-search result: the chunk's citation payload + its score."""

    score: float
    payload: dict

# Stable namespace for deriving point ids; must not change once data exists.
_ID_NAMESPACE = uuid.UUID("6f1c8b2e-3d4a-5b6c-7d8e-9f0a1b2c3d4e")


def _point_id(chunk: Chunk) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{chunk.source}|{chunk.page}|{chunk.chunk_index}"))


class QdrantStore:
    def __init__(self, url: str, collection: str) -> None:
        self.collection = collection
        self.client = QdrantClient(url=url)

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        """Create the collection if missing (cosine). `recreate` wipes it first."""
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists and recreate:
            logger.info("recreating collection %s (dropping existing points)", self.collection)
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        batch_size: int = 64,
    ) -> int:
        """Upsert chunks with their vectors and citation payload."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        points = [
            PointStruct(
                id=_point_id(chunk),
                vector=list(vector),
                payload={
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        for batch in _batched(points, batch_size):
            self.client.upsert(collection_name=self.collection, points=batch)
        return len(points)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """Return the top_k most similar chunks, filtered by score_threshold.

        Qdrant applies the threshold server-side (cosine floor), so a query with
        no chunk above the floor comes back empty — that empty result is what the
        RAG chain turns into an honest "not found in the documents" instead of a
        hallucinated answer.
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [SearchHit(score=p.score, payload=p.payload or {}) for p in response.points]


def _batched(items: Sequence, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])
