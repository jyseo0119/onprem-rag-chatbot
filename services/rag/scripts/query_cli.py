"""Ask the RAG chain a question from the command line.

    uv run python -m scripts.query_cli "How do I report a safety incident?"
    uv run python -m scripts.query_cli "..." --retrieve-only   # no LLM, no API key

`--retrieve-only` embeds the query and prints the retrieved chunks with their
cosine scores — handy for tuning RETRIEVAL_SCORE_THRESHOLD, and for verifying
the retrieval half without an LLM key. The default path runs the full chain and
prints the grounded answer followed by its cited sources.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from app.config import settings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the RAG chain.")
    parser.add_argument("query", help="the question to ask")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="only run retrieval and print the matched chunks (no LLM call)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def _print_sources(sources: list[dict]) -> None:
    if not sources:
        print("\n(no sources — nothing cleared the score threshold)")
        return
    print("\nSources:")
    for s in sources:
        preview = textwrap.shorten(s["text"].replace("\n", " "), width=100)
        print(f"  [{s['n']}] {s['source']} p.{s['page']}  (score {s['score']})")
        print(f"      {preview}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.retrieve_only:
        from app.retrieval.retriever import Retriever

        chunks = Retriever.from_settings().retrieve(args.query)
        if not chunks:
            print(
                f"No chunks above threshold {settings.retrieval_score_threshold} "
                f"in collection '{settings.qdrant_collection}'."
            )
            return 0
        print(f"Top {len(chunks)} chunk(s) for: {args.query!r}\n")
        for i, c in enumerate(chunks, start=1):
            preview = textwrap.shorten(c.text.replace("\n", " "), width=100)
            print(f"  [{i}] {c.source} p.{c.page}  (score {c.score:.4f})")
            print(f"      {preview}")
        return 0

    from app.rag.chain import RAGChain

    try:
        chain = RAGChain.from_settings()
    except (RuntimeError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = chain.answer(args.query)
    print(result["answer"])
    _print_sources(result["sources"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
