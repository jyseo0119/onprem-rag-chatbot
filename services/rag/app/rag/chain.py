"""RAG chain: retrieve -> assemble grounded prompt -> generate -> answer + sources.

Returns citations alongside the answer so the UI can show *where* each claim came
from — the core trust signal of a document RAG system.

Two guardrails against confident hallucination live here:
  1. If retrieval returns nothing above the score threshold, we short-circuit and
     answer "not found" without ever calling the LLM (also saves a token spend).
  2. The system prompt pins the model to the provided context and tells it to say
     so when the answer isn't there, rather than filling the gap from priors.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from app.llm.base import LLMProvider, get_provider
from app.retrieval.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = (
    "I couldn't find anything relevant to that question in the indexed documents."
)

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about a company's internal "
    "documents. Use ONLY the numbered context passages provided. If the answer is "
    "not contained in them, say you could not find it in the documents — never "
    "guess or use outside knowledge. Cite the passages you rely on with their "
    "bracketed numbers, e.g. [1], [2]."
)


@dataclass
class Source:
    """A cited passage, mirrored back to the UI alongside the answer."""

    n: int  # citation marker used in the prompt/answer ([n])
    source: str
    page: int
    chunk_index: int
    score: float
    text: str


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, source-stamped context blocks."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {chunk.source}, page {chunk.page})\n{chunk.text}")
    return "\n\n".join(blocks)


class RAGChain:
    def __init__(self, retriever: Retriever, provider: LLMProvider) -> None:
        self.retriever = retriever
        self.provider = provider

    @classmethod
    def from_settings(cls) -> RAGChain:
        """Build the chain from env-driven settings (CLI / single-process use)."""
        return cls(retriever=Retriever.from_settings(), provider=get_provider())

    def answer(self, query: str) -> dict:
        """Answer `query` from the documents; return {answer, sources}."""
        chunks = self.retriever.retrieve(query)
        if not chunks:
            return {"answer": NOT_FOUND_MESSAGE, "sources": []}

        context = _build_context(chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        answer = self.provider.generate(prompt, system=SYSTEM_PROMPT)

        sources = [
            asdict(
                Source(
                    n=i,
                    source=chunk.source,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    score=round(chunk.score, 4),
                    text=chunk.text,
                )
            )
            for i, chunk in enumerate(chunks, start=1)
        ]
        return {"answer": answer, "sources": sources}
