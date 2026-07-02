# apps/web — Chat UI

Minimal Next.js (App Router) chat interface for the RAG backend.

## What it does

- **Chat window** — type a question, get a grounded answer back.
- **Source citations** — every answer expands to show the passages it was built
  from: document, page, chunk index, and the retrieval similarity score. Those
  `[n]` markers match the citations the model puts in its answer.
- **Friendly errors** — backend states (no API key, provider not wired up,
  Qdrant/LLM unreachable, backend process down) are translated into a single
  actionable message instead of a raw HTTP body.

## Why a BFF proxy

The browser never calls FastAPI directly. It POSTs to `app/api/chat/route.ts`
(same origin), which forwards server-side to `BACKEND_URL`. That keeps the
backend URL (and any future auth) off the client, removes the CORS surface, and
is the natural place to map backend status codes to UI messages.

```
browser ──▶ /api/chat (Next server) ──▶ FastAPI /query ──▶ {answer, sources}
```

## Run

```bash
cp .env.example .env.local   # optional: override BACKEND_URL (default :8000)
npm install
npm run dev                  # http://localhost:3000
```

The FastAPI backend must be running (`cd services/rag && uv run uvicorn
app.main:app --reload`), Qdrant must be up, and documents must be ingested — see
the root README. A grounded answer additionally needs a configured LLM key on
the backend; without one, `/query` returns 503 and the UI shows a message
telling you to set it.

## Config

| Var           | Default                 | Notes                                    |
|---------------|-------------------------|------------------------------------------|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI base URL, server-side only.      |

Not prefixed with `NEXT_PUBLIC_`, so it is never shipped to the browser.
