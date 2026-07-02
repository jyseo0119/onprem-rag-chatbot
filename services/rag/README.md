# services/rag — RAG backend

Python (FastAPI) backend: document ingestion, retrieval, and grounded generation.

## Setup

```bash
uv sync                 # core deps
uv sync --extra ocr     # + PaddleOCR for scanned PDFs (optional, heavy)
```

## Run

```bash
uv run uvicorn app.main:app --reload      # API server
uv run python -m scripts.ingest_cli ../../data/raw   # batch ingest
```

## Layout

```
app/
├─ main.py            FastAPI app (/health, /ingest, /query)
├─ config.py          env-backed settings (single source of tunables)
├─ ingest/            loader, ocr, chunker, pipeline
├─ embedding/         bge_m3 embedder
├─ vectorstore/       qdrant_store
├─ retrieval/         retriever (top-k + score threshold)
├─ llm/               base interface + openai / gemini / vllm providers
└─ rag/               chain (retrieve -> prompt -> generate -> sources)
scripts/
└─ ingest_cli.py      batch indexing entrypoint
```

Configuration lives in the root `.env` (see `.env.example`).
