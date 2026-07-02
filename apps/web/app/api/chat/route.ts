// BFF proxy: browser -> this route (same origin) -> FastAPI /query.
//
// Why a proxy instead of calling FastAPI from the browser:
//   - the backend URL and any future auth stay server-side (no CORS, no leaks),
//   - we translate the backend's HTTP status codes into friendly UI messages
//     right here, so the client only ever deals with {answer, sources} or a
//     single human-readable error string.

import { NextResponse } from "next/server";
import type { QueryResponse } from "@/app/types";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

// Map backend failures (see the /query error mapping in main.py) to messages a
// non-technical user can act on, instead of surfacing raw 5xx bodies.
function friendlyError(status: number, detail: string): string {
  switch (status) {
    case 422:
      return "Please enter a question.";
    case 503:
      return "The LLM API key isn't configured on the server. Set OPENAI_API_KEY (or the matching provider key) in the backend .env and restart it.";
    case 501:
      return "The selected LLM provider isn't wired up yet (on-prem providers land in a later stage).";
    case 400:
      return `Backend rejected the request: ${detail}`;
    case 502:
      return "The backend couldn't reach Qdrant or the LLM. Is the vector DB up and are documents ingested?";
    default:
      return detail || `Backend error (HTTP ${status}).`;
  }
}

export async function POST(request: Request) {
  let query: string;
  try {
    const body = await request.json();
    query = typeof body?.query === "string" ? body.query.trim() : "";
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  if (!query) {
    return NextResponse.json({ error: "Please enter a question." }, { status: 400 });
  }

  let backendRes: Response;
  try {
    backendRes = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      cache: "no-store",
    });
  } catch {
    // Connection refused / DNS — backend process is down.
    return NextResponse.json(
      { error: `Can't reach the RAG backend at ${BACKEND_URL}. Is it running (uvicorn app.main:app)?` },
      { status: 502 },
    );
  }

  if (!backendRes.ok) {
    // FastAPI surfaces errors as {"detail": "..."}.
    let detail = "";
    try {
      const errBody = await backendRes.json();
      detail = typeof errBody?.detail === "string" ? errBody.detail : JSON.stringify(errBody?.detail ?? "");
    } catch {
      /* non-JSON error body — fall back to the status-based message */
    }
    return NextResponse.json(
      { error: friendlyError(backendRes.status, detail) },
      { status: backendRes.status },
    );
  }

  const data = (await backendRes.json()) as QueryResponse;
  return NextResponse.json(data);
}
