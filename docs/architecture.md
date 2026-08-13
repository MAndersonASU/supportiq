# Architecture

Status: Phase 1 (Data Engineering) — this document will grow as each phase
lands.

## System overview

SupportIQ is built as five pipeline stages, each owned by a `src/` module:

| Stage | Module | Responsibility |
|---|---|---|
| Ingestion & validation | `src/data` | Load raw tweets, enforce a schema/data contract, reject or quarantine bad records |
| Feature engineering | `src/features` | Reconstruct reply threads into ticket-level records and derive model-ready features |
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
   → processed dataset (data/processed)
   → feature engineering (thread reconstruction → ticket_features.parquet, versioned with DVC)
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

### Feature engineering (`src/features/build_ticket_features.py`)

The processed dataset is tweet-level; a support ticket doesn't exist as a
record until this stage builds one. A ticket is defined as a
customer-initiated reply thread: the root tweet (no parent) has
`inbound = True`. Threads rooted in a brand's own tweet (e.g. a marketing
post that happened to get replies) are not tickets and are excluded
entirely, not just their root row.

Thread membership is resolved by following each tweet's
`in_response_to_tweet_id` link back to its root, using an iterative
find-with-path-compression (the same technique as union-find), so each
tweet's root is computed once and reused rather than re-walked. A tweet
whose parent ID isn't present in the dataset (a broken link, ~0.19% of
responses per the validation report) is treated as its own root rather
than failing.

Each ticket aggregates its full thread into: message counts (customer vs.
brand), the brand account that first responded, time to first response,
and basic text statistics on the opening message.

**Two findings from this stage that affect Phase 2 planning:**

- `resolved` (did the thread get a brand reply) is constant — `True` for
  100% of the 789,547 tickets extracted. This dataset was compiled by its
  creators as customer/brand *exchange pairs*, so every included thread
  already has a reply by construction. It isn't a usable ML feature or
  target on this dataset as defined.
- This dataset has no `category` or `priority` ground-truth labels. Phase
  2's triage classifier will need a labeling strategy — heuristic/keyword
  labels, unsupervised clustering with manual review, or a small
  hand-labeled sample — before it can train a supervised model.

`first_response_seconds` also has a long-tailed outlier (max ~242M
seconds, ~7.7 years) alongside a reasonable median (~1,607 seconds, ~27
minutes) — flagged in `feature_report.json` for Phase 2 to handle (likely
capping or a log transform) rather than silently feeding the raw value
into a model.

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
- **2026-08-01** — A ticket is defined as a thread rooted in a
  customer-initiated tweet. Brand-initiated threads (marketing tweets that
  received replies) are excluded entirely rather than kept with a null
  ticket-opener, since they don't represent a support interaction.
- **2026-08-01** — DVC's remote is a local filesystem path outside the
  repository (`C:\Users\Ander\.dvc-storage\supportiq`), not inside the
  OneDrive-synced project folder. The project directory is already
  cloud-synced by OneDrive; pointing DVC's cache at the same tree would
  double-sync large data files. This remote will move to cloud storage
  (S3/GCS) when the project reaches its cloud-deploy phase.
- **2026-08-01** — Only `ticket_features.parquet` (83 MB, the final
  feature-engineered artifact) is tracked with DVC, not the larger
  intermediate tiers (`tweets.parquet` at 458 MB, the landing-zone file).
  Those are fully reproducible by re-running the pipeline scripts against
  the same raw source, so versioning only the artifact that's expensive
  to regenerate and directly consumed downstream avoids duplicating
  hundreds of megabytes in the DVC cache for no reproducibility benefit.
