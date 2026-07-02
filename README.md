# On-Prem RAG Chatbot

A minimal, runnable **Retrieval-Augmented Generation** chatbot over internal
documents, built to demonstrate a production-shaped RAG pipeline.

It runs two ways from the same codebase:

- **Demo path** — API LLM (OpenAI / Gemini). Clone, set a key, run.
- **On-prem path** — self-hosted LLM serving (vLLM + Qwen2.5). Swap one env
  var; embeddings and the vector DB already run locally. See
  [`docs/on-prem-switch.md`](docs/on-prem-switch.md).

> This is a **prototype / proof asset**, not a product. Where it matters, the
> docs separate "what this demo does" from "what production actually needs".

## Architecture

```
PDF ──▶ load ──▶ [OCR if scanned] ──▶ chunk ──▶ embed (BGE-m3)
                                                     │
                                                     ▼
   answer ◀── LLM ◀── prompt+context ◀── retrieve ◀── Qdrant (vector DB)
```

| Layer      | Choice                         | Notes                                  |
|------------|--------------------------------|----------------------------------------|
| Embedding  | BGE-m3                         | Runs locally in both demo and on-prem  |
| Vector DB  | Qdrant                         | Local via Docker                       |
| LLM        | OpenAI / Gemini → vLLM+Qwen2.5 | Swappable behind one interface         |
| OCR        | PaddleOCR                      | Optional path for scanned PDFs         |
| Frontend   | Next.js                        | Minimal chat UI                        |

## Repo layout

```
.
├─ services/rag/   # Python RAG backend (FastAPI): ingest + retrieve + generate
├─ apps/web/       # Next.js chat UI
├─ data/           # Demo documents (public dummy PDFs only)
├─ docs/           # Architecture & on-prem switch guide & design decisions
└─ docker-compose.yml
```

## Quickstart

> Status: backend (ingest + retrieve + generate) and the Next.js chat UI are
> implemented and wired end to end. A grounded answer needs an LLM API key on the
> backend; without one the UI surfaces a clear "set the key" message.

### 1. Prerequisites

- Docker (for Qdrant)
- [uv](https://docs.astral.sh/uv/) (Python dependency manager)
- Node.js 20+ (for the web UI)
- An `OPENAI_API_KEY` **or** `GEMINI_API_KEY` for the demo path

### 2. Configure

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER and the matching API key
```

### 3. Start infrastructure

```bash
docker compose up -d qdrant
```

### 4. Ingest documents

```bash
# put some public PDFs in data/raw/ first
cd services/rag
uv sync
uv run python -m scripts.ingest_cli ../../data/raw
```

### 5. Run the API + UI

```bash
# backend
cd services/rag && uv run uvicorn app.main:app --reload

# frontend (separate terminal)
cd apps/web && npm install && npm run dev
```

Open http://localhost:3000 and ask a question about your documents.

The backend can also be driven directly:

```bash
curl -s localhost:8000/health
# {"status":"ok"}

# index PDFs (loads BGE-m3 once, then reuses it for queries)
curl -s -X POST localhost:8000/ingest \
  -H 'Content-Type: application/json' -d '{"path":"../../data/raw"}'

# ask a grounded question — answer comes back with cited sources
curl -s -X POST localhost:8000/query \
  -H 'Content-Type: application/json' -d '{"query":"What PPE must operators wear?"}'
```

Interactive API docs are served at http://localhost:8000/docs.

## On-prem path

The default demo talks to an API LLM. To serve the model yourself with
vLLM + Qwen2.5 instead, see [`docs/on-prem-switch.md`](docs/on-prem-switch.md).
Because vLLM exposes an OpenAI-compatible API, the switch is essentially:

```bash
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8001/v1   # vLLM container mapped to host 8001
```

## Design notes

See [`docs/decisions.md`](docs/decisions.md) for the reasoning behind chunking
strategy, the BGE-m3 choice, and retrieval threshold tuning — the parts that
tend to break in real deployments.
