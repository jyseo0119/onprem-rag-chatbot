"""Chunking strategy.

The first place a naive RAG pipeline breaks: chunks that split mid-sentence or
mid-paragraph destroy retrieval quality — the embedding of half a sentence
rarely matches the query, and the half that does match is missing context.

Strategy (documented in docs/decisions.md):

- **Structure-aware recursive split.** We try to break on the biggest natural
  boundary that keeps a piece under `chunk_size` — paragraphs, then lines, then
  sentences, then words — instead of slicing at a fixed character offset.
- **Overlap.** Consecutive chunks share `chunk_overlap` characters so a fact
  that lands near a boundary is retrievable from either side.
- **Per-page.** We chunk one page at a time and stamp each chunk with its page
  number, so a retrieved chunk cites an exact page. The trade-off is that a
  paragraph spanning a page break is cut at the break; acceptable for clean
  page-level citations, and revisited if it ever hurts recall.

Both knobs are measured in characters (not tokens): BGE-m3 handles up to 8192
tokens, so an 800-char chunk is comfortably within budget, and character counts
keep the chunker independent of any specific tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingest.loader import LoadedDoc

# Separators tried in order, largest structural unit first. The empty string is
# the hard fallback: split between characters when a single "word" is longer
# than chunk_size (e.g. a long URL or an un-spaced CJK run).
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    """A retrievable unit of text with its citation metadata."""

    text: str
    source: str  # filename
    page: int  # 1-indexed
    chunk_index: int  # position within the document, for a stable id


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split `text` into pieces <= chunk_size, preferring earlier separators."""
    sep = separators[0]
    rest = separators[1:]

    if sep == "":
        # Hard fallback: fixed-width slices.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    pieces: list[str] = []
    for part in parts:
        if not part:
            continue
        segment = part + sep  # keep the separator so re-joined text reads naturally
        if len(segment) <= chunk_size:
            pieces.append(segment)
        else:
            pieces.extend(_split_recursive(segment, chunk_size, rest))
    return pieces


def _merge_with_overlap(
    pieces: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Greedily pack small pieces up to chunk_size, carrying overlap forward."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            # Seed the next chunk with the overlap tail of the one we just closed.
            current = (current[-chunk_overlap:] if chunk_overlap else "") + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a single block of text into overlapping, boundary-aware chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    pieces = _split_recursive(text, chunk_size, _SEPARATORS)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def chunk_document(
    doc: LoadedDoc, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk every page of a document, assigning a stable per-doc chunk index."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    index = 0
    for page in doc.pages:
        for piece in chunk_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    page=page.page,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks
