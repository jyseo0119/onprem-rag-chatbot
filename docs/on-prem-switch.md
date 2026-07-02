# Switching to on-prem LLM serving (vLLM + Qwen2.5)

The default demo answers with an API LLM (OpenAI or Gemini). This document shows
how to serve the model yourself instead, with **no change to the RAG logic** —
only the LLM provider and a couple of env vars.

## What actually changes

The whole codebase depends on one interface — `LLMProvider.generate()` in
`app/llm/base.py`. `get_provider()` reads `LLM_PROVIDER` and returns a concrete
backend. Everything downstream (retrieval, prompt assembly, citation handling,
the API server, the web UI) is identical across providers.

| Layer            | Demo (API)         | On-prem (vLLM)              | Changes? |
| ---------------- | ------------------ | --------------------------- | -------- |
| Embeddings       | BGE-m3 (local)     | BGE-m3 (local)              | no       |
| Vector DB        | Qdrant (docker)    | Qdrant (docker)             | no       |
| Chunking / retrieval | same           | same                        | no       |
| RAG chain        | same               | same                        | no       |
| **LLM backend**  | OpenAI / Gemini    | vLLM serving Qwen2.5        | **yes**  |
| Config           | `OPENAI_API_KEY`   | `VLLM_BASE_URL` + `LLM_PROVIDER=vllm` | **yes** |

Switching is a config change because vLLM exposes an **OpenAI-compatible API**.
`VLLMProvider` is a thin subclass of `OpenAIProvider` pointed at `VLLM_BASE_URL`
(`app/llm/vllm_provider.py`) — same Chat Completions call, same temperature-0
grounded-QA behaviour.

## Why vLLM

- OpenAI-compatible server → drop-in for the existing client, zero rewrite.
- Continuous batching / paged attention → real throughput under concurrent load,
  which matters once more than one employee is asking questions.
- Runs the weights on your own GPU, so **no document text leaves the network** —
  the whole point of an on-prem deployment for a manufacturer's internal docs.

## Why Qwen2.5 for this workload

- Strong Korean + English handling, which fits mixed Korean/English technical
  manuals better than English-first open models.
- 7B instruct is a sensible prototype size (fits a single 24 GB GPU at
  `--max-model-len 8192`); scale to 14B/32B or add quantization for production.
- Llama 3.3 is a fine alternative — swap `--model` and `VLLM_MODEL`, nothing
  else changes.

## Run it

1. Start vLLM (requires an NVIDIA GPU + `nvidia-container-toolkit`):

   ```bash
   docker compose --profile onprem up -d vllm
   ```

   First start downloads the weights into the `hf_cache` volume, so it is slow
   once and fast afterwards. The container serves on port 8000 internally, mapped
   to host **8001** so it never collides with the FastAPI backend on 8000.

2. Point the backend at it — in `.env`:

   ```bash
   LLM_PROVIDER=vllm
   VLLM_BASE_URL=http://localhost:8001/v1
   VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
   # VLLM_API_KEY is unused by default (vLLM does not authenticate); leave EMPTY.
   ```

3. Restart the backend. `/query` now answers from your self-hosted model. The
   web UI and API contract are unchanged.

> If the backend itself later runs inside the same compose network, use the
> service DNS name instead of localhost: `VLLM_BASE_URL=http://vllm:8000/v1`.

## Gemini path (still an API, not on-prem)

If you only need a different *API* provider (e.g. cost or availability), set
`LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`. Same one-line switch, no on-prem
infrastructure. Useful as an intermediate step before committing GPU hardware.

## Prototype vs. production

What this demo shows is the *switch*, not a tuned deployment. For real on-prem
serving you would additionally consider:

- **GPU sizing / quantization** — 7B fp16 needs ~16 GB; use AWQ/GPTQ or FP8 to
  fit larger models or raise concurrency on the same card.
- **Concurrency & context** — tune `--max-model-len` and `--gpu-memory-utilization`
  against your real question/context lengths; long retrieved contexts dominate
  the KV cache budget.
- **Availability** — a single container is a single point of failure; production
  wants a replica + health checks in front of the backend.
- **Prompt/format parity** — verify Qwen's chat template renders the system
  prompt the way the API models did; grounded-QA faithfulness can shift between
  models even with identical prompts, so re-check the not-found behaviour.
