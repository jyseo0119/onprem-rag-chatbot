// Shared types mirroring the FastAPI backend contract (services/rag/app/main.py).
// Keep these in sync with QueryResponse / SourceModel on the backend.

export interface Source {
  n: number;
  source: string;
  page: number;
  chunk_index: number;
  score: number;
  text: string;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
}

export interface ApiError {
  error: string;
}
