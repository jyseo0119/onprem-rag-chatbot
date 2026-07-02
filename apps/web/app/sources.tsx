"use client";

import { useState } from "react";
import type { Source } from "@/app/types";

// Collapsible citation panel shown under an assistant answer. Each entry maps to
// a [n] marker in the answer text; the score is the retrieval similarity so a
// reviewer can eyeball how strong the grounding was.
export function Sources({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen((v) => !v)}>
        {open ? "▾" : "▸"} {sources.length} source{sources.length > 1 ? "s" : ""}
      </button>

      {open && (
        <ol className="sources-list">
          {sources.map((s) => (
            <li key={s.n} className="source">
              <div className="source-head">
                <span className="source-marker">[{s.n}]</span>
                <span className="source-meta">
                  {s.source} · p.{s.page} · chunk {s.chunk_index}
                </span>
                <span className="source-score" title="retrieval similarity score">
                  {s.score.toFixed(3)}
                </span>
              </div>
              <p className="source-text">{s.text}</p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
