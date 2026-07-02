"""FastAPI entrypoint.

Exposes the RAG backend:
  GET  /health   liveness probe
  POST /ingest   index PDFs from a path into Qdrant
  POST /query    retrieve + generate a grounded answer with sources

The heavy resource — the ~2.3 GB BGE-m3 embedder — is created once at startup
and shared (via `app.state`) across both /ingest and /query, so the model is
loaded a single time and stays warm across requests. Loading itself is lazy
(first embed call triggers the download/load), so startup stays cheap.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.embedding.bge_m3 import BGEM3Embedder
from app.ingest.pipeline import ingest_path
from app.llm.base import get_provider
from app.rag.chain import RAGChain
from app.retrieval.retriever import Retriever
from app.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the shared, request-spanning resources once per process."""
    app.state.embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    app.state.store = QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
    )
    app.state.retriever = Retriever(
        embedder=app.state.embedder,
        store=app.state.store,
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )
    logger.info(
        "RAG backend ready (provider=%s, collection=%s, model load is lazy)",
        settings.llm_provider,
        settings.qdrant_collection,
    )
    yield


app = FastAPI(title="On-Prem RAG Chatbot", version="0.1.0", lifespan=lifespan)


# --- request/response models ------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="the question to ask")


class SourceModel(BaseModel):
    n: int
    source: str
    page: int
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]


class IngestRequest(BaseModel):
    # Server-side path (relative to the backend's working dir). Mirrors the CLI
    # default; the demo indexes public dummy PDFs dropped into data/raw.
    path: str = "../../data/raw"
    recreate: bool = False
    # None -> use ENABLE_OCR from settings. Set true/false to override per request.
    enable_ocr: bool | None = None


class FileResultModel(BaseModel):
    source: str
    status: str
    pages: int
    chunks: int
    detail: str


class IngestResponse(BaseModel):
    collection: str
    indexed_files: int
    total_files: int
    total_chunks: int
    files: list[FileResultModel]


# --- endpoints --------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — deliberately dependency-free so it stays fast."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    """Index every PDF under `path` into Qdrant, reusing the shared embedder."""
    report = ingest_path(
        req.path,
        recreate=req.recreate,
        enable_ocr=req.enable_ocr,
        embedder=request.app.state.embedder,
        store=request.app.state.store,
    )
    if not report.files:
        raise HTTPException(status_code=404, detail=f"no PDFs found at {req.path}")

    return IngestResponse(
        collection=settings.qdrant_collection,
        indexed_files=report.indexed_files,
        total_files=len(report.files),
        total_chunks=report.total_chunks,
        files=[FileResultModel(**vars(f)) for f in report.files],
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve grounded context and generate an answer with citations."""
    try:
        provider = get_provider()
    except RuntimeError as exc:  # e.g. OPENAI_API_KEY not set
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NotImplementedError as exc:  # gemini / vllm land in stage 5
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ValueError as exc:  # unknown LLM_PROVIDER
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chain = RAGChain(retriever=request.app.state.retriever, provider=provider)
    try:
        result = chain.answer(req.query)
    except Exception as exc:  # upstream (Qdrant / LLM) failure -> bad gateway
        logger.exception("query failed")
        raise HTTPException(status_code=502, detail=f"query failed: {exc}") from exc

    return QueryResponse(**result)
