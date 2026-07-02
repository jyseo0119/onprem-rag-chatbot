"""Load source documents into raw text.

Handles text-based PDFs (pypdf). Text is extracted per page so a chunk can be
attributed back to a page number in the citation UI. Scanned/image PDFs yield
little or no extractable text; we flag those (`is_probably_scanned`) so the
pipeline can route them to the optional OCR path (`app/ingest/ocr.py`) instead of
silently indexing empty pages.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Below this many extracted characters per page (on average), a PDF is almost
# certainly scanned/image-only rather than text-based. Kept deliberately low so
# sparse-but-real text PDFs (title pages, forms) are not misclassified.
_SCANNED_CHARS_PER_PAGE = 40


@dataclass
class PageText:
    """Extracted text for a single 1-indexed page."""

    page: int
    text: str


@dataclass
class LoadedDoc:
    """A parsed document, ready for chunking."""

    source: str  # filename, used as the citation label
    path: Path
    pages: list[PageText]

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def is_probably_scanned(self) -> bool:
        """True when there is too little text to be a real text-based PDF."""
        if not self.pages:
            return True
        return self.char_count / len(self.pages) < _SCANNED_CHARS_PER_PAGE


def _normalize(text: str) -> str:
    """Light cleanup of pypdf output: collapse runs of blank lines/spaces.

    We keep single newlines (paragraph structure matters to the chunker) but
    drop the ragged whitespace pypdf tends to emit.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing spaces on each line.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def load_pdf(path: str | Path) -> LoadedDoc:
    """Extract per-page text from a text-based PDF."""
    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = _normalize(raw)
        pages.append(PageText(page=i, text=cleaned))

    doc = LoadedDoc(source=path.name, path=path, pages=pages)
    if doc.is_probably_scanned:
        logger.warning(
            "%s looks scanned/image-only (%.0f chars/page); OCR path needed",
            path.name,
            doc.char_count / max(len(pages), 1),
        )
    return doc


def discover_pdfs(path: str | Path) -> list[Path]:
    """Resolve a file or directory into a sorted list of PDF paths."""
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    if path.is_dir():
        return sorted(p for p in path.glob("*.pdf"))
    raise FileNotFoundError(f"No such file or directory: {path}")
