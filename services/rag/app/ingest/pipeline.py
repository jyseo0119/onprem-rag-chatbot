"""Ingestion orchestration: load -> (ocr) -> chunk -> embed -> upsert.

Ties together loader, ocr, chunker, the BGE-m3 embedder and the Qdrant store.
Called by scripts/ingest_cli.py and (stage 3) the POST /ingest endpoint.

The embedder is loaded once and reused across files (the ~2.3 GB model load is
the expensive part), and each file is reported independently so one unreadable
or scanned PDF doesn't sink the whole run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.embedding.bge_m3 import BGEM3Embedder
from app.ingest.chunker import chunk_document
from app.ingest.loader import discover_pdfs, load_pdf
from app.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class FileResult:
    source: str
    # "indexed" | "indexed_ocr" | "skipped_scanned" | "empty" | "error"
    status: str
    pages: int = 0
    chunks: int = 0
    detail: str = ""


@dataclass
class IngestReport:
    files: list[FileResult] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(f.chunks for f in self.files)

    @property
    def indexed_files(self) -> int:
        return sum(1 for f in self.files if f.status.startswith("indexed"))


def ingest_path(
    path: str | Path,
    *,
    recreate: bool = False,
    collection: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    enable_ocr: bool | None = None,
    embedder: BGEM3Embedder | None = None,
    store: QdrantStore | None = None,
) -> IngestReport:
    """Ingest a single PDF or a folder of PDFs into Qdrant.

    Unspecified knobs fall back to `app.config.settings` (env-driven). The API
    server injects a shared `embedder` and `store` so the ~2.3 GB BGE-m3 model
    is loaded once and reused across ingest and query requests; when injected,
    `collection` is ignored (the store already carries its target collection).

    `enable_ocr` routes scanned/image-only PDFs through the optional PaddleOCR
    path (`app/ingest/ocr.py`); off by default and requires `uv sync --extra ocr`.
    """
    collection = collection or settings.qdrant_collection
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    enable_ocr = settings.enable_ocr if enable_ocr is None else enable_ocr

    pdfs = discover_pdfs(path)
    report = IngestReport()
    if not pdfs:
        logger.warning("no PDFs found at %s", path)
        return report

    if embedder is None:
        embedder = BGEM3Embedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
    if store is None:
        store = QdrantStore(url=settings.qdrant_url, collection=collection)
    store.ensure_collection(embedder.dimension, recreate=recreate)

    for pdf in pdfs:
        result = _ingest_file(pdf, embedder, store, chunk_size, chunk_overlap, enable_ocr)
        report.files.append(result)
        logger.info("%s -> %s (%d chunks)", result.source, result.status, result.chunks)

    return report


def _ingest_file(
    pdf: Path,
    embedder: BGEM3Embedder,
    store: QdrantStore,
    chunk_size: int,
    chunk_overlap: int,
    enable_ocr: bool,
) -> FileResult:
    try:
        doc = load_pdf(pdf)
    except Exception as exc:  # a corrupt/encrypted PDF shouldn't abort the batch
        logger.exception("failed to load %s", pdf.name)
        return FileResult(source=pdf.name, status="error", detail=str(exc))

    via_ocr = False
    if doc.is_probably_scanned:
        if not enable_ocr:
            return FileResult(
                source=doc.source,
                status="skipped_scanned",
                pages=len(doc.pages),
                detail="looks scanned; set ENABLE_OCR=true (needs `uv sync --extra ocr`)",
            )
        # Route through the optional PaddleOCR path. Imported lazily so the core
        # pipeline never needs the heavy OCR deps.
        try:
            from app.ingest.ocr import OcrUnavailableError, load_pdf_ocr

            doc = load_pdf_ocr(pdf)
        except OcrUnavailableError as exc:
            return FileResult(
                source=doc.source,
                status="skipped_scanned",
                pages=len(doc.pages),
                detail=str(exc),
            )
        except Exception as exc:  # OCR of one scan shouldn't abort the batch
            logger.exception("OCR failed for %s", pdf.name)
            return FileResult(source=pdf.name, status="error", detail=f"OCR failed: {exc}")
        via_ocr = True
        if doc.is_probably_scanned:  # OCR still yielded almost nothing
            return FileResult(
                source=doc.source,
                status="empty",
                pages=len(doc.pages),
                detail="OCR produced no usable text",
            )

    chunks = chunk_document(doc, chunk_size, chunk_overlap)
    if not chunks:
        return FileResult(source=doc.source, status="empty", pages=len(doc.pages))

    vectors = embedder.embed_documents([c.text for c in chunks])
    store.upsert_chunks(chunks, vectors)
    return FileResult(
        source=doc.source,
        status="indexed_ocr" if via_ocr else "indexed",
        pages=len(doc.pages),
        chunks=len(chunks),
    )
