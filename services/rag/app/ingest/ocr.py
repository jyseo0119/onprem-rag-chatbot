"""OCR path for scanned (image) PDFs — optional, off by default.

Text-based PDFs go through `loader.load_pdf` (pypdf). Scanned/image-only PDFs
yield almost no extractable text there (`LoadedDoc.is_probably_scanned`), so the
pipeline routes them here instead of silently indexing blank pages.

This path is heavy and stays disabled unless BOTH hold:
  1. the `ocr` extra is installed — `uv sync --extra ocr` (PaddleOCR + PyMuPDF), and
  2. `ENABLE_OCR=true`.

Both the import and the model load are lazy, so the core pipeline runs — and this
module imports — with none of the OCR dependencies present. The output is a normal
`LoadedDoc` (per-page text), so chunking, embedding and page-level citations
downstream are identical to the text path.

Status: implemented as a wired reference path, not a verified one — this demo repo
has no GPU or scanned sample on hand to exercise it end-to-end. The PaddleOCR
result shape below targets PaddleOCR 2.x `.ocr()`. See docs/decisions.md ("OCR").
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.ingest.loader import LoadedDoc, PageText, _normalize

logger = logging.getLogger(__name__)

# Rasterisation resolution. 200 dpi is the usual sweet spot for OCR: high enough
# for small body text, low enough to keep memory and latency sane on large scans.
_RENDER_DPI = 200


class OcrUnavailableError(RuntimeError):
    """Raised when the OCR path is requested but its dependencies are missing."""


def ocr_available() -> bool:
    """True when the optional OCR dependencies can be imported."""
    try:
        import fitz  # noqa: F401  (PyMuPDF — rasterises PDF pages)
        import paddleocr  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def _engine():
    """Build the PaddleOCR engine once (model load is the expensive part)."""
    from paddleocr import PaddleOCR

    # Minimal, version-portable args: language only. Angle/layout knobs differ
    # across PaddleOCR majors, so we stay on the defaults here.
    return PaddleOCR(lang=settings.ocr_lang)


def _render_page(page) -> "object":
    """Rasterise one PyMuPDF page to a BGR ndarray for PaddleOCR."""
    import fitz
    import numpy as np

    zoom = _RENDER_DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    # PaddleOCR (via OpenCV) expects BGR; PyMuPDF emits RGB(A). Drop alpha, flip.
    return img[:, :, :3][:, :, ::-1]


def _run_ocr(engine, img) -> str:
    """Run OCR on one page image and join detected lines top-to-bottom."""
    result = engine.ocr(img)
    lines: list[str] = []
    # PaddleOCR 2.x: result is [ per_image[ [box, (text, confidence)], ... ] ].
    for page_result in result or []:
        for entry in page_result or []:
            lines.append(entry[1][0])
    return "\n".join(lines)


def load_pdf_ocr(path: str | Path) -> LoadedDoc:
    """Extract per-page text from a scanned PDF via PaddleOCR.

    Returns the same `LoadedDoc` shape as the text loader, so downstream chunking
    and citation are unchanged. Raises `OcrUnavailableError` if the optional deps
    are not installed.
    """
    if not ocr_available():
        raise OcrUnavailableError(
            "OCR path requested but paddleocr/PyMuPDF are not installed. "
            "Install the optional extra: `uv sync --extra ocr`."
        )

    import fitz

    path = Path(path)
    engine = _engine()
    pages: list[PageText] = []
    with fitz.open(str(path)) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = _run_ocr(engine, _render_page(page))
            pages.append(PageText(page=i, text=_normalize(text)))

    doc = LoadedDoc(source=path.name, path=path, pages=pages)
    logger.info(
        "OCR extracted %d chars from %s (%d pages)",
        doc.char_count,
        path.name,
        len(pages),
    )
    return doc
