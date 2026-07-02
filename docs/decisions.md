# Design decisions

The parts of RAG that break in production, and how this repo handles them.
Numbers are measured on the demo corpus where stated; see
[`architecture.md`](architecture.md) for how the components fit together.

## Chunking
- **Structure-aware recursive split, not fixed-size.** Fixed-offset slicing cuts
  mid-sentence; the embedding of half a sentence rarely matches a query, and the
  fact you needed is stranded across the cut. We split on the largest natural
  boundary that fits — paragraph → line → sentence → word → (last resort)
  character. See `app/ingest/chunker.py`.
- **Size / overlap.** `CHUNK_SIZE=800`, `CHUNK_OVERLAP=120` characters. Counted in
  characters, not tokens, so the chunker stays tokenizer-independent; 800 chars is
  well inside BGE-m3's 8192-token window. Overlap lets a fact near a boundary be
  retrieved from either neighbouring chunk.
- **Per-page chunking for exact citations.** Each chunk is stamped with its source
  filename + page, so the UI can cite an exact page. Trade-off: a paragraph
  straddling a page break is cut at the break. Accepted for clean page-level
  attribution; revisit if it measurably hurts recall.
- **Scanned PDFs are skipped, not silently indexed.** If a PDF yields almost no
  extractable text (`< ~40 chars/page`) it is flagged as scanned (`is_probably_scanned`)
  and either sent to the optional OCR path or skipped with an actionable status —
  never indexed as blank pages. See "OCR" below.

## Embeddings — why BGE-m3
- Multilingual (Korean) strength, runs locally, demo/on-prem parity.
- We index BGE-m3's **dense** vector only (1024-dim, cosine, L2-normalized so
  cosine == dot product). Its sparse/ColBERT outputs are left as a later hybrid
  option, not wired into demo retrieval. See `app/embedding/bge_m3.py`.
- Vectors are the *same* in demo and on-prem — only the answering LLM swaps — so
  retrieval quality is identical across both deployments.

## Retrieval threshold
- **A cosine floor turns "no relevant document" into an honest non-answer.**
  Without it, the nearest chunk is *always* returned no matter how weak the
  match, and the LLM will dress that weak context up as a confident answer. The
  floor (`RETRIEVAL_SCORE_THRESHOLD`) is applied server-side by Qdrant, so an
  out-of-domain query comes back with zero chunks.
- **How the chain uses an empty result.** `app/rag/chain.py` short-circuits on an
  empty retrieval: it returns a fixed "couldn't find it in the documents" message
  and never calls the LLM — cheaper, and impossible to hallucinate from.
- **Where 0.4 came from (BGE-m3, this demo corpus).** Measured on the sample
  conveyor-safety manual: an on-topic query ("what PPE must operators wear?")
  scores **0.70** on the right chunk and 0.48–0.54 on the rest of the same
  document; a clearly off-topic query ("capital of France?") tops out **below
  0.4** on every chunk. 0.4 sits in that gap — it keeps same-document context
  while rejecting unrelated questions. It is corpus-dependent: a larger, more
  diverse corpus should re-tune it (use `scripts/query_cli.py --retrieve-only`,
  which prints per-chunk scores for exactly this).
- **Only the top result is strongly on-topic here.** With a 4-page toy manual the
  whole document is loosely related, so scores 2–4 are naturally lower. `top_k=5`
  is generous for this size; on a real corpus top-k matters more.

## API layer
- **The embedder is a process singleton, shared by both endpoints.** BGE-m3 is
  ~2.3 GB; loading it per request (or once for ingest and again for query) would
  be wasteful and slow. It is created once in the FastAPI lifespan and shared via
  `app.state` across `/ingest` and `/query`, so it loads a single time and stays
  warm. Loading is lazy (first embed triggers it), so startup itself stays cheap.
  See `app/main.py`; `ingest_path()` grew an `embedder`/`store` injection seam for
  exactly this.
- **A missing/unconfigured LLM is a 503, not a 500.** `/query` maps provider
  errors to honest HTTP codes: no API key → **503** (service unavailable, with the
  fix in the message), an unimplemented on-prem provider → **501**, an unknown
  `LLM_PROVIDER` → **400**. Upstream Qdrant/LLM failures during answering surface
  as **502**. The point is that an operator reading the status code knows whether
  the problem is their config or the backend.
- **`/health` is dependency-free on purpose.** It never touches Qdrant or the
  model, so it stays a fast liveness signal that a load balancer can poll without
  waking the model or coupling liveness to Qdrant being reachable.

## Frontend (chat UI)
- **A BFF proxy sits between the browser and FastAPI.** The browser POSTs to a
  same-origin Next route (`app/api/chat/route.ts`), which forwards server-side to
  `BACKEND_URL/query`. This keeps the backend URL (and any future auth header) off
  the client, removes the CORS surface entirely, and gives one server-side place
  to translate backend status codes into UI messages. The browser only ever sees
  `{answer, sources}` or one human-readable `{error}` string.
- **Backend status codes become actionable messages, not raw 5xx.** The proxy maps
  the /query contract to plain English: 503 → "set the LLM API key", 501 → "that
  provider isn't wired up yet", 502 → "can't reach Qdrant/LLM — is the DB up and
  are docs ingested?", plus a connection-refused case → "is the backend running?".
  An out-of-domain question isn't an error here — the backend returns 200 with the
  fixed not-found answer and zero sources, so the UI just shows that answer.
- **Citations are first-class, not a footnote.** Each answer expands to the exact
  passages behind it — source file, page, chunk index, and the retrieval score —
  with `[n]` markers matching the model's inline citations. Showing the score lets
  a reviewer eyeball grounding strength, which is the whole trust proposition of a
  document RAG UI. See `app/sources.tsx`.
- **No streaming, deliberately.** The backend returns the full `{answer, sources}`
  in one shot (it also short-circuits without calling the LLM on empty retrieval),
  so the UI is a plain request/response with a pending state rather than SSE — less
  moving surface for a prototype, and citations arrive atomically with the answer.
- **Minimal dependency footprint.** Next.js + React only, hand-written CSS, no UI
  or styling framework — the point is to demonstrate the RAG wiring, not a design
  system. Pinned to a patched Next 15.x (the initial scaffold version carried
  CVE-2025-66478).

## On-prem switch (LLM provider)
- **The switch is a config change because vLLM speaks the OpenAI API.**
  `VLLMProvider` is a thin subclass of `OpenAIProvider` pointed at `VLLM_BASE_URL`
  (`app/llm/vllm_provider.py`) — identical Chat Completions call, identical
  temperature-0 grounded-QA behaviour. Going from API to self-hosted Qwen2.5 flips
  `LLM_PROVIDER=vllm` and one URL; retrieval, prompt assembly, citations, the API
  server, and the web UI are all untouched. This is the payoff of abstracting only
  the LLM behind `LLMProvider` — see `docs/on-prem-switch.md`.
- **Gemini is the second API path, not a third design.** Implemented via
  `google-genai` with the same interface and temperature 0; it exists so switching
  API vendors is also a one-line env change, and as an intermediate step before
  committing GPU hardware.
- **vLLM is off by default and gated behind a GPU.** It lives under the compose
  `onprem` profile so the default `docker compose up` stays light and GPU-free; the
  demo's happy path is still the API LLM.
- **vLLM host port moved 8000 → 8001.** The FastAPI backend already owns host 8000,
  so mapping the vLLM container to 8001 avoids a collision when both run on one box.
  In-container vLLM still serves on 8000, so a backend inside the same compose
  network would use `http://vllm:8000/v1` instead.
- **No key check on the vLLM path.** vLLM does not authenticate by default, so
  `get_provider('vllm')` constructs without demanding a key (defaults to the vLLM
  `EMPTY` convention); the API paths still raise a 503-mapped error when their key
  is missing.

## OCR (scanned PDFs)
- **OCR is an optional, off-by-default branch — not a second pipeline.** The scanned
  path (`app/ingest/ocr.py`) returns the same `LoadedDoc` (per-page text) as the text
  loader, so chunking, embedding, and page-level citations downstream are byte-for-byte
  identical. OCR only changes *how* a page's text is obtained, never what happens to it
  after. Real manufacturing archives are full of scanned SOPs and inspection sheets, so
  the seam matters even though the demo corpus is born-digital.
- **Gated twice, on purpose.** It runs only when the heavy extra is installed
  (`uv sync --extra ocr` → PaddleOCR + PyMuPDF) *and* `ENABLE_OCR=true`. Both the import
  and the ~hundreds-of-MB model load are lazy, so the default install stays lean and the
  core pipeline imports with none of the OCR deps present. Missing deps surface as a
  friendly "install the extra" status on the file, not a crash that sinks the batch.
- **Rasterise, then recognise.** PDFs carry no pixels the OCR engine can read directly,
  so PyMuPDF renders each page to a 200-dpi image (the usual legibility/latency sweet
  spot) which PaddleOCR then transcribes line-by-line. Language is configurable
  (`OCR_LANG`, e.g. `korean`) since the target documents are Korean.
- **Prototype, not production-verified.** The path is wired end-to-end and unit-checked
  for its guards and routing, but it is *not* exercised against a real scan in this repo
  (no GPU / no scanned sample on hand), and the PaddleOCR result-shape parsing targets
  the 2.x `.ocr()` API. In production this is where the real work lives — OCR quality
  gating (drop low-confidence lines), layout/table handling, and per-language model
  choice — deliberately left as documented surface rather than overclaimed.

## Demo vs on-prem — what is real vs what production still needs
- **What is genuinely shared.** Ingestion, BGE-m3 embeddings, the Qdrant vector DB,
  retrieval + thresholding, prompt assembly, citation plumbing, the FastAPI server,
  and the web UI are byte-for-byte the same in both deployments. The *only* thing
  that changes going on-prem is the LLM provider and one env var (see "On-prem
  switch"). So the retrieval quality a reviewer sees in the demo is the retrieval
  quality on-prem — the LLM swap doesn't touch it.
- **What this repo deliberately does not claim.** It is a prototype of the data
  path, run against a 4-page public dummy manual, single-node, single-user. The
  LLM real-call path, the vLLM/GPU path, and OCR against a real scan are wired and
  guard-checked but not exercised here (no key / no GPU / no scanned sample on
  hand) — each is flagged as such in its section above rather than overclaimed.
- **What a real on-prem deployment additionally needs.** Access control and
  per-document authorization (who may see which document); request/answer logging,
  evaluation, and monitoring; GPU capacity planning and batching for vLLM;
  scheduled, incremental, large-scale ingestion instead of a manual folder drop;
  OCR quality gating (confidence thresholds, layout/table handling, per-language
  models) for messy scanned archives; and a re-tuned retrieval threshold on the
  real, larger corpus. These are the parts that separate "runs on my laptop" from
  "serves a plant" — named here so the boundary is explicit, not blurred.
