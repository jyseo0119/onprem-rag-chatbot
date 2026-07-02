# Architecture

How the pieces fit together, what data flows between them, and the one seam
where the demo and an on-prem deployment diverge. For *why* each choice was made
(chunking, embedding, thresholds), see [`decisions.md`](decisions.md).

## Two paths, one codebase

```
                          ┌──────────────────────────────────────────┐
   PDFs (data/raw)        │            services/rag (FastAPI)         │
        │                 │                                          │
        ▼                 │   POST /ingest                           │
  ┌───────────┐           │        │                                 │
  │  ingest   │◀──────────┼────────┘                                 │
  │ pipeline  │           │                                          │
  └─────┬─────┘           │   POST /query ──▶ RAG chain              │
        │ upsert          │        ▲              │                  │
        ▼                 │        │              ▼                  │
   ┌─────────┐            │   ┌──────────┐   ┌──────────┐            │
   │ Qdrant  │◀───────────┼───│retriever │   │   LLM    │            │
   │ (vector)│  search    │   └──────────┘   │ provider │            │
   └─────────┘            │                  └────┬─────┘            │
                          └───────────────────────┼──────────────────┘
                                                   │
                          ┌────────────────────────┴──────────────┐
                          │  openai / gemini  (demo, API)          │
                          │  vllm + Qwen2.5   (on-prem, self-host) │
                          └────────────────────────────────────────┘

   apps/web (Next.js)  ──▶  /api/chat (BFF proxy)  ──▶  POST /query
```

Everything left of the LLM provider box — ingestion, embeddings, the vector DB,
retrieval, prompt assembly, citations, the API, and the UI — is identical in
both deployments. Only the provider on the right swaps. That is the entire point
of the design: production RAG quality lives in retrieval, not in which LLM
answers, so the LLM is the only thing abstracted behind an interface.

## Ingest path

`app/ingest/pipeline.py :: ingest_path()` — driven by `scripts/ingest_cli.py`
(batch, single process) or `POST /ingest` (shared warm embedder).

```
discover PDFs ─▶ load_pdf ─▶ scanned? ─┬─no──▶ chunk ─▶ embed ─▶ upsert ─▶ Qdrant
                                       └─yes─▶ [OCR path] ─▶ chunk ─▶ …
```

1. **load** (`loader.py`) — extract per-page text into a `LoadedDoc`
   (`source`, `pages[]`, `is_probably_scanned`). A PDF yielding `< ~40 chars/page`
   is flagged as scanned.
2. **route on scanned** (`pipeline._ingest_file`) — a scanned PDF is either sent
   to the optional OCR path (when `ENABLE_OCR=true` **and** `uv sync --extra ocr`)
   or skipped with an actionable status; it is never indexed as blank pages. OCR
   (`ocr.py`) rasterises each page at 200 dpi with PyMuPDF and transcribes it with
   PaddleOCR, returning the *same* `LoadedDoc` shape so everything downstream is
   unchanged.
3. **chunk** (`chunker.py`) — structure-aware recursive split
   (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=120` chars), chunked per page so each chunk
   keeps an exact source + page.
4. **embed** (`embedding/bge_m3.py`) — BGE-m3 dense vector, 1024-dim, cosine,
   L2-normalized. The ~2.3 GB model is the expensive part, so it is loaded once
   and reused across all files (and, in the server, across requests).
5. **upsert** (`vectorstore/qdrant_store.py`) — into the `onprem_docs` collection.
   Point IDs are a deterministic `uuid5(source, page, chunk_index)`, so
   re-ingesting the same document is idempotent (overwrite, not duplicate).

Each file is reported independently (`FileResult`: `indexed` / `indexed_ocr` /
`skipped_scanned` / `empty` / `error`), so one corrupt or scanned PDF never sinks
the batch.

### Qdrant payload schema

Every point carries the metadata the citation UI needs:

```json
{ "source": "safety-manual.pdf", "page": 3, "chunk_index": 7, "text": "…" }
```

## Query path

`app/rag/chain.py :: RAGChain.answer()` — driven by `scripts/query_cli.py` or
`POST /query`.

1. **retrieve** (`retrieval/retriever.py`) — embed the query with the same
   BGE-m3, then `qdrant_store.search()` with `top_k=5` and a server-side cosine
   floor `score_threshold=0.4`. An out-of-domain question returns **zero** chunks.
2. **short-circuit** — if retrieval is empty, return the fixed "not found"
   message and **never call the LLM** (cheaper, and impossible to hallucinate
   from).
3. **assemble** — render the chunks as numbered, source-stamped context blocks
   and build a grounded prompt; the system prompt pins the model to the context
   and to bracketed `[n]` citations.
4. **generate** — the configured `LLMProvider` answers at temperature 0.
5. **return** `{answer, sources[]}` where each source is
   `{n, source, page, chunk_index, score, text}` — the `[n]` matching the inline
   citations in the answer.

## Components

| Component | Path | Responsibility |
|-----------|------|----------------|
| Config | `app/config.py` | Every tunable, env-backed (single source of knobs) |
| Loader / OCR | `app/ingest/{loader,ocr}.py` | PDF → per-page text (`LoadedDoc`) |
| Chunker | `app/ingest/chunker.py` | Structure-aware split, per-page |
| Embedder | `app/embedding/bge_m3.py` | BGE-m3 dense vectors (demo + on-prem) |
| Vector store | `app/vectorstore/qdrant_store.py` | Qdrant collection, upsert, search |
| Retriever | `app/retrieval/retriever.py` | top-k + score threshold → `RetrievedChunk` |
| LLM provider | `app/llm/*.py` | `LLMProvider` interface + openai / gemini / vllm |
| RAG chain | `app/rag/chain.py` | retrieve → prompt → generate → answer + sources |
| API | `app/main.py` | FastAPI `/health` `/ingest` `/query` |
| Web UI | `apps/web/` | Next.js chat + citations + BFF proxy |

## The demo ↔ on-prem seam

Only `app/llm/*` and one env var move:

| | Demo | On-prem |
|---|---|---|
| `LLM_PROVIDER` | `openai` / `gemini` | `vllm` |
| Provider | API SDK call | `VLLMProvider` (OpenAI-compatible, points at `VLLM_BASE_URL`) |
| Embeddings, Qdrant, retrieval, prompt, UI | — | **unchanged** |

Because vLLM exposes an OpenAI-compatible API, `VLLMProvider` is a thin subclass
of `OpenAIProvider` with a different base URL. Full switch procedure and the GPU
compose profile are in [`on-prem-switch.md`](on-prem-switch.md).

## Serving model & error mapping

- **Warm singleton embedder.** The FastAPI `lifespan` builds the embedder, store,
  and retriever once and shares them via `app.state`; `/ingest` and `/query` reuse
  them, so BGE-m3 loads a single time and stays warm. `/health` touches none of
  them, staying a fast liveness signal.
- **Honest status codes.** `/query` maps provider errors so an operator can tell
  config problems from backend problems: missing key → **503**, unimplemented
  provider → **501**, unknown `LLM_PROVIDER` → **400**, upstream Qdrant/LLM failure
  → **502**. The web BFF (`apps/web/app/api/chat/route.ts`) translates those into
  one actionable message each and keeps the backend URL and any auth off the
  browser.

## What a real on-prem deployment adds

This repo is a prototype of the *data path*. A production install on a
manufacturer's network additionally needs, at minimum: access control and
per-document authorization, request/answer logging and monitoring, GPU capacity
planning for vLLM, scheduled/large-scale ingestion (not a manual folder drop),
and OCR quality gating for real scanned archives. Those are named where relevant
in [`decisions.md`](decisions.md) rather than half-built here.
