# Architecture

Status: Phase 1 (Data Engineering) — this document will grow as each phase
lands.

## System overview

SupportIQ is built as five pipeline stages, each owned by a `src/` module:

| Stage | Module | Responsibility |
|---|---|---|
| Ingestion & validation | `src/data` | Load raw tickets, enforce a schema/data contract, reject or quarantine bad records |
| Feature engineering | `src/features` | Derive model-ready features from cleaned ticket data |
| ML triage model | `src/models` | Train/evaluate/register a category + priority classifier |
| Retrieval + generation | `src/ai` | Embed tickets/KB articles, retrieve similar cases, generate a grounded draft response via Claude |
| Serving | `src/serving` | FastAPI endpoints exposing classification + resolution-assistant functionality |

## Data flow

```
raw tweets (data/raw, untouched source data)
   → ingestion (schema validation via Pydantic, malformed records quarantined)
   → landing zone (data/landing, schema-conformant Parquet)
   → cleaning (text normalization, dedup, missing-value policy)
   → processed dataset (data/processed, versioned with DVC)
   → feature engineering
   → ML training → model registry (MLflow)
   → embeddings → vector store
   → RAG pipeline (retrieval + Claude generation) → serving layer
```

### Ingestion (`src/data/ingest.py`, `src/data/schema.py`)

The raw dataset is ~2.8M tweets (516 MB as CSV) and is never loaded into
memory in full. `ingest.py` streams it in 50k-row chunks, validates each
row against the `RawTweet` Pydantic model (the data contract — field
types, required fields, parseable dates and ID lists), and writes
conformant rows to `data/landing/tweets.parquet`. Rows that fail
validation are counted and skipped rather than aborting the run, so a
handful of malformed records at row 2 million doesn't take down the whole
ingestion job.

## Design decisions log

Decisions are recorded here as they're made, with the reasoning, so the
"why" behind the architecture is traceable without digging through commit
history.

- **2026-08-01** — Chose a single coherent dataset (customer support
  tickets) to drive every phase, rather than separate toy datasets per
  phase, so the project tells one consistent product story end-to-end.
- **2026-08-01** — Local-first infrastructure (Docker for packaging, no
  managed cloud services during build) with a cloud deploy planned only at
  the end, to keep iteration fast and cost at zero during development.
- **2026-08-01** — Dataset: Kaggle's "Twitter Customer Support" dataset —
  real brand-support exchanges on Twitter. Chosen over more structured,
  pre-labeled ticket datasets because the raw text is noisier and less
  structured, requiring real data-engineering work in Phase 1 (parsing
  conversation threads, handling missing structure, deriving category and
  priority labels rather than reading them off the schema).
- **2026-08-01** — Ingestion streams the raw CSV in chunks rather than
  loading it into a single DataFrame, to keep peak memory bounded
  regardless of dataset size. Landing-zone output uses Parquet (columnar,
  typed, compressed) instead of CSV, since every downstream stage reads
  this file repeatedly and Parquet is both smaller on disk and faster to
  read back than CSV.
- **2026-08-01** — Schema validation (`RawTweet`, a Pydantic model) is
  enforced during ingestion itself rather than deferred entirely to a
  later validation stage. Structural validity (types, parseable dates,
  well-formed ID lists) belongs at the ingestion boundary; business-rule
  validation (allowed value ranges, cross-field consistency) is handled
  separately in the validation stage.
