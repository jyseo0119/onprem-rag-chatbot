"""Batch ingestion CLI.

    uv run python -m scripts.ingest_cli ../../data/raw
    uv run python -m scripts.ingest_cli ../../data/raw --recreate

Indexes every PDF in the given path (file or folder) into Qdrant via
app.ingest.pipeline, then prints a per-file chunk-count report.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import settings
from app.ingest.pipeline import ingest_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant.")
    parser.add_argument(
        "path",
        nargs="?",
        default="../../data/raw",
        help="PDF file or folder of PDFs (default: ../../data/raw)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and recreate the collection before ingesting",
    )
    parser.add_argument("--collection", default=None, help="override QDRANT_COLLECTION")
    parser.add_argument("--chunk-size", type=int, default=None, help="override CHUNK_SIZE")
    parser.add_argument(
        "--chunk-overlap", type=int, default=None, help="override CHUNK_OVERLAP"
    )
    parser.add_argument(
        "--ocr",
        dest="enable_ocr",
        action="store_true",
        default=None,
        help="OCR scanned PDFs instead of skipping them (needs `uv sync --extra ocr`)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    report = ingest_path(
        args.path,
        recreate=args.recreate,
        collection=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        enable_ocr=args.enable_ocr,
    )

    if not report.files:
        print(f"No PDFs found at {args.path}", file=sys.stderr)
        return 1

    collection = args.collection or settings.qdrant_collection
    print(f"\nIngested into collection '{collection}':")
    print(f"  {'file':<40} {'status':<16} {'pages':>5} {'chunks':>7}")
    print(f"  {'-' * 40} {'-' * 16} {'-' * 5} {'-' * 7}")
    for f in report.files:
        print(f"  {f.source:<40} {f.status:<16} {f.pages:>5} {f.chunks:>7}")
    print(
        f"\n{report.indexed_files}/{len(report.files)} file(s) indexed, "
        f"{report.total_chunks} chunks total."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
