# Demo data

This folder holds the documents the RAG pipeline ingests.

> ⚠️ **Use only publicly redistributable dummy data** (e.g. public agency
> manuals / policy PDFs). Do **not** place any real client data here.

## Layout

```
data/
├─ raw/         # Source documents you drop in (PDF). Ignored by git.
└─ processed/   # Parsed / OCR'd text produced by the pipeline. Ignored by git.
```

Only `.gitkeep` and this README are committed — actual documents stay local.

## Expected format

- Put source PDFs directly in `data/raw/`, e.g. `data/raw/safety-manual.pdf`.
- Both text-based PDFs and scanned (image) PDFs are supported; scanned files
  are routed through the OCR path (PaddleOCR).
- Filenames are used as the document id / citation label, so name them
  meaningfully (`electrical-safety-guide-2024.pdf`).

## Where to get sample PDFs

Any openly licensed public manual works. Suggested sources:

- Public agency safety / operation manuals
- Government policy or guideline PDFs published for public use

Download a few into `data/raw/`, then run the ingestion CLI (see the root
`README.md`) to index them into Qdrant.
