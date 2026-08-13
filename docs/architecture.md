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
   → validation (business-rule checks via Pandera, quality metrics report)
   → validated tier (data/validated, business-rule-passed Parquet)
   → cleaning (text normalization, missing-value policy)
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

### Validation (`src/data/validate.py`)

Ingestion enforces structural validity — types, parseable values. This
stage enforces business rules on top of that: `tweet_id` uniqueness, valid
value ranges, and a plausible `created_at` window. These are hard checks —
a row that fails one is dropped from the output and counted as rejected.

Not every data-quality issue warrants dropping a row. Referential
completeness (does `in_response_to_tweet_id` point at a tweet that
actually exists in this dataset) and near-duplicate detection
(`author_id` + `text` pairs, common with templated support responses) are
recorded as metrics in a validation report rather than used to reject
records — this dataset is a filtered slice of larger conversations, so
some broken links are expected, not anomalous. The distinction between a
hard check (gates the data) and a soft check (measures the data) is
itself a design decision, not a technical limitation.

Output: the set of rows passing all hard checks is written to
`data/validated/tweets.parquet`; the report (row counts, rejection count,
and the two quality metrics) is written to
`data/validated/validation_report.json`.

### Cleaning (`src/data/clean.py`)

Reads the validated tier and produces the final processed dataset. Two
responsibilities:

- **Text normalization** — Unicode NFKC normalization and whitespace
  collapse, written to a new `text_clean` column. The original `text`
  column is kept as-is; downstream stages choose which they need, and
  provenance isn't lost by overwriting the source field.
- **Missing-value policy, enforced not just assumed** — only
  `response_tweet_id` and `in_response_to_tweet_id` are allowed to be
  null (a tweet with no parent or no replies). Every other column is
  checked and any unexpected null is counted in the cleaning report; on
  this dataset, ingestion and validation already guarantee this, so the
  count is a defensive integrity check between pipeline stages, not
  something expected to fire.

Deduplication was deliberately left out. Validation's
`duplicate_author_text_pairs` metric flagged ~4,700 rows, but inspecting
them showed each has a distinct, unique `tweet_id` — they're a support
account sending the same templated reply to different customers, not
duplicate records. Deduplicating on `(author_id, text)` would have
silently deleted real, distinct interactions. The metric was worth
computing to know it existed; acting on it without checking what it
actually represented would have corrupted the dataset.

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
- **2026-08-01** — Business-rule validation uses Pandera instead of Great
  Expectations. Great Expectations' dependency chain requires a numpy
  build with no prebuilt wheel for this Python version, forcing a
  from-source compile; Pandera provides equivalent DataFrame schema and
  check enforcement without that overhead.
- **2026-08-01** — Validation checks are split into hard checks (schema
  violations — the row is dropped) and soft checks (referential
  completeness, near-duplicate detection — a metric is recorded, the row
  is kept). Rejecting on every anomaly would be wrong for a dataset that's
  inherently a partial slice of larger conversations; the goal is
  visibility into data quality, not zero tolerance.
- **2026-08-01** — Split what was originally a single `data/processed`
  output into `data/validated` (business-rule-passed, pre-cleaning) and
  `data/processed` (final, cleaned). Collapsing them into one tier made
  "processed" ambiguous about whether cleaning had actually happened yet.
- **2026-08-01** — No deduplication in the cleaning stage. The duplicate
  `(author_id, text)` pairs surfaced by validation are legitimate distinct
  interactions with unique `tweet_id`s (templated support replies to
  different customers), not duplicate records — confirmed by inspecting a
  sample before writing any drop logic against the metric.
