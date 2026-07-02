"use client";

import { useRef, useState } from "react";
import type { QueryResponse, Source } from "@/app/types";
import { Sources } from "./sources";

type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; sources: Source[] }
  | { role: "error"; text: string };

const SAMPLE_QUESTIONS = [
  "What PPE must operators wear?",
  "What is the lockout/tagout procedure?",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listEndRef = useRef<HTMLDivElement>(null);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      const data = await res.json();

      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: "error", text: data?.error ?? "Something went wrong." },
        ]);
      } else {
        const payload = data as QueryResponse;
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: payload.answer, sources: payload.sources ?? [] },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "error", text: "Network error talking to the app server." },
      ]);
    } finally {
      setLoading(false);
      // Let the new message render, then scroll it into view.
      requestAnimationFrame(() => listEndRef.current?.scrollIntoView({ behavior: "smooth" }));
    }
  }

  return (
    <main className="app">
      <header className="header">
        <h1>On-Prem RAG Chatbot</h1>
        <p className="subtitle">
          Answers are grounded in your indexed documents and cite where each claim came from.
        </p>
      </header>

      <section className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <p>Ask a question about the ingested documents. Try:</p>
            <div className="samples">
              {SAMPLE_QUESTIONS.map((s) => (
                <button key={s} className="sample" onClick={() => ask(s)} disabled={loading}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <div className="role">{m.role === "assistant" ? "Assistant" : m.role === "user" ? "You" : "Error"}</div>
            <div className="text">{m.text}</div>
            {m.role === "assistant" && m.sources.length > 0 && <Sources sources={m.sources} />}
          </div>
        ))}

        {loading && (
          <div className="bubble assistant pending">
            <div className="role">Assistant</div>
            <div className="text">Retrieving and generating…</div>
          </div>
        )}
        <div ref={listEndRef} />
      </section>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <input
          className="input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your documents…"
          disabled={loading}
          autoFocus
        />
        <button className="send" type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}
