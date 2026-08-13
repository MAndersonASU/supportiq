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

### Weak-supervision labeling (`src/models/label_tickets.py`, `src/models/explore_categories.py`)

The dataset has no ground-truth `category` or `priority` labels (identified
during feature engineering). Two constraints ruled out the usual fixes:
no hand-labeled sample exists, and there was no budget for LLM-based
labeling. The solution is weak supervision — keyword labeling functions
computed directly from ticket text, at zero cost and no scale limit.

The category keyword lists were not guessed. `explore_categories.py` runs
TF-IDF + LSA (SVD) + KMeans over a sample of ticket text with brand
handles and anonymized customer IDs stripped out first — without that
stripping, clusters mostly rediscover *which company* the tweet is
addressed to (Delta, Apple, Amazon...), which is already captured by
`brand_id` and tells us nothing about issue type. After stripping, several
clusters aligned with real issue-type vocabulary: account/password/access,
order/delivery/package, flight delays, app glitches, service-quality
complaints. Those clusters' top terms directly informed the keyword lists
in `label_tickets.py`.

Raw cluster assignment was **not** used as the label. Roughly half the
sample fell into one large, linguistically generic cluster that KMeans
could not usefully subdivide (silhouette scores stayed low, 0.05–0.09,
across every cluster count tried) — a known limitation of TF-IDF
clustering on short, informal text. Using cluster ID directly would have
produced one dominant meaningless class.

`assign_category` scores each category's keyword matches (word-boundary
regex, apostrophe-normalized so typographic quotes like `'` don't break
contraction keywords) and picks the highest-scoring category, defaulting
to `General Inquiry` when nothing matches. `assign_priority` scores
urgency language, exclamation marks, and ALL-CAPS words into Low/Medium/
High buckets — deliberately computed from ticket text only, never from an
outcome field like `first_response_seconds`, since deriving a label from
an outcome a future model might predict would leak that outcome into its
own target.

### Baseline classifiers (`src/models/split_data.py`, `src/models/train_classifier.py`)

Two separate TF-IDF + logistic regression models — one for `category`,
one for `priority` — trained on the weak-supervision labels, each
compared against a majority-class baseline and tracked in MLflow
(SQLite-backed store at `mlflow.db`; MLflow's plain filesystem backend is
in maintenance mode as of the version used here). A single stratified
70/15/15 train/val/test split (stratified on `category`, the more
imbalanced target) is reused for both models, so a ticket is never in
train for one target and test for the other.

**Read the category metric with its actual meaning, not its face value.**
Test macro F1 is 0.977, uniformly high across all seven classes
(0.94–0.997). That is the signature of a model reconstructing a
deterministic function, not evidence of learned semantic understanding —
expected, because word-level TF-IDF features overlap almost completely
with the exact keyword phrases that define the labels, and there is no
independent ground truth to check real generalization against. This
number answers "did the model recover the labeling rule," not "does the
model understand support ticket categories."

**Priority (test macro F1 0.421) is weaker for a specific, diagnosable
reason, not because it's a harder problem in the abstract.** Part of the
priority label depends on exclamation marks and ALL-CAPS words, but
scikit-learn's default `TfidfVectorizer` lowercases text and strips
punctuation before tokenizing — that signal is structurally invisible to
the model. Feeding those same signals in as explicit features would only
reproduce category's circularity under a different name, so this gap is
documented rather than papered over. Separately,
`class_weight="balanced"` is pushing hard on the rare High-priority
class: precision there is 0.115 (many false positives) against a recall
of 0.447 — a real, tunable cost of balancing against a distribution this
skewed (75% Low / 20% Medium / 5% High), left for the hyperparameter-
tuning stage rather than adjusted ad hoc here.

### Hyperparameter tuning (`src/models/tune_classifier.py`)

Sweeps regularization strength (`C` — 0.1, 1.0, 10.0) and class weighting
(`None`, `balanced`) for both classifiers. The text vectorizer is fit once
per target and reused across the grid, since re-vectorizing is the
expensive step and only the classifier itself changes between trials.
Model selection uses validation macro F1 exclusively; the test set is
scored exactly once per target, after the winning configuration is
already fixed, so tuning can't leak into the reported number.

Category's tuned result (test macro F1 0.987) is only marginally above
the untuned baseline (0.977) — consistent with the earlier finding that
this task is close to its ceiling regardless of configuration, since the
model is reconstructing a deterministic rule rather than learning a
harder decision boundary.

Priority's High-precision problem does not resolve through tuning.
`class_weight="balanced"` wins on validation macro F1 at every
regularization strength tested but one, and the tuned model's
High-priority precision (0.119) is statistically the same as the
untuned baseline (0.115) regardless of `C`. That rules out "the wrong
hyperparameter" as the explanation: macro F1 consistently rewards
balanced weighting despite its precision cost on the rare class, and the
real fix is a feature or evaluation-strategy change (probability
thresholding, or the punctuation/casing features already ruled out in
the baseline stage for circularity reasons) rather than anything in this
grid.

## Design decisions log

Decisions are recorded here as they're made, with the reasoning, so the
"why" behind the architecture is traceable without digging through commit
history.

- **2026-08-13** — Chose a single coherent dataset (customer support
  tickets) to drive every phase, rather than separate toy datasets per
  phase, so the project tells one consistent product story end-to-end.
- **2026-08-13** — Local-first infrastructure (Docker for packaging, no
  managed cloud services during build) with a cloud deploy planned only at
  the end, to keep iteration fast and cost at zero during development.
- **2026-08-13** — Dataset: Kaggle's "Twitter Customer Support" dataset —
  real brand-support exchanges on Twitter. Chosen over more structured,
  pre-labeled ticket datasets because the raw text is noisier and less
  structured, requiring real data-engineering work in Phase 1 (parsing
  conversation threads, handling missing structure, deriving category and
  priority labels rather than reading them off the schema).
- **2026-08-13** — Ingestion streams the raw CSV in chunks rather than
  loading it into a single DataFrame, to keep peak memory bounded
  regardless of dataset size. Landing-zone output uses Parquet (columnar,
  typed, compressed) instead of CSV, since every downstream stage reads
  this file repeatedly and Parquet is both smaller on disk and faster to
  read back than CSV.
- **2026-08-13** — Schema validation (`RawTweet`, a Pydantic model) is
  enforced during ingestion itself rather than deferred entirely to a
  later validation stage. Structural validity (types, parseable dates,
  well-formed ID lists) belongs at the ingestion boundary; business-rule
  validation (allowed value ranges, cross-field consistency) is handled
  separately in the validation stage.
- **2026-08-13** — Business-rule validation uses Pandera instead of Great
  Expectations. Great Expectations' dependency chain requires a numpy
  build with no prebuilt wheel for this Python version, forcing a
  from-source compile; Pandera provides equivalent DataFrame schema and
  check enforcement without that overhead.
- **2026-08-13** — Validation checks are split into hard checks (schema
  violations — the row is dropped) and soft checks (referential
  completeness, near-duplicate detection — a metric is recorded, the row
  is kept). Rejecting on every anomaly would be wrong for a dataset that's
  inherently a partial slice of larger conversations; the goal is
  visibility into data quality, not zero tolerance.
- **2026-08-13** — Split what was originally a single `data/processed`
  output into `data/validated` (business-rule-passed, pre-cleaning) and
  `data/processed` (final, cleaned). Collapsing them into one tier made
  "processed" ambiguous about whether cleaning had actually happened yet.
- **2026-08-13** — No deduplication in the cleaning stage. The duplicate
  `(author_id, text)` pairs surfaced by validation are legitimate distinct
  interactions with unique `tweet_id`s (templated support replies to
  different customers), not duplicate records — confirmed by inspecting a
  sample before writing any drop logic against the metric.
- **2026-08-13** — A ticket is defined as a thread rooted in a
  customer-initiated tweet. Brand-initiated threads (marketing tweets that
  received replies) are excluded entirely rather than kept with a null
  ticket-opener, since they don't represent a support interaction.
- **2026-08-13** — DVC's remote is a local filesystem path outside the
  repository (`C:\Users\Ander\.dvc-storage\supportiq`), not inside the
  OneDrive-synced project folder. The project directory is already
  cloud-synced by OneDrive; pointing DVC's cache at the same tree would
  double-sync large data files. This remote will move to cloud storage
  (S3/GCS) when the project reaches its cloud-deploy phase.
- **2026-08-13** — Only `ticket_features.parquet` (83 MB, the final
  feature-engineered artifact) is tracked with DVC, not the larger
  intermediate tiers (`tweets.parquet` at 458 MB, the landing-zone file).
  Those are fully reproducible by re-running the pipeline scripts against
  the same raw source, so versioning only the artifact that's expensive
  to regenerate and directly consumed downstream avoids duplicating
  hundreds of megabytes in the DVC cache for no reproducibility benefit.
- **2026-08-13** — No budget for LLM-based labeling, so category/priority
  labels are generated by weak supervision (keyword labeling functions)
  instead of an LLM API. Category keywords are grounded in an exploratory
  TF-IDF/KMeans clustering pass (with brand names and IDs stripped) rather
  than picked arbitrarily; raw cluster assignment was rejected as the
  label itself because roughly half the data falls into one
  indistinguishable cluster.
- **2026-08-13** — Priority scoring reads ticket text only, never
  `first_response_seconds` or any other post-open outcome field, even
  though response time is an intuitive urgency proxy. A label built from
  an outcome the classifier might later be asked to predict (or that
  correlates with what it predicts) leaks the answer into its own target.
- **2026-08-13** — One stratified split (on `category`), reused for both
  the category and priority classifiers, rather than splitting
  independently per target. Keeps a given ticket in the same partition
  for both models — simpler to reason about than two splits that
  disagree on which tickets are held out.
- **2026-08-13** — Switched MLflow's tracking backend from the default
  filesystem store to SQLite (`sqlite:///mlflow.db`) after the filesystem
  backend raised a maintenance-mode error on the installed MLflow
  version. This is the currently recommended backend, not a workaround.
- **2026-08-13** — Did not add exclamation-mark/ALL-CAPS features to the
  priority classifier to close its accuracy gap with category. Those
  features are literally the inputs to the priority labeling function;
  feeding them to the model would reproduce category's rule-reconstruction
  circularity rather than fix anything. The gap is documented as an
  understood, structural property of training on weak-supervision labels
  with a feature space narrower than the label's own construction.
- **2026-08-13** — Hyperparameter tuning selects on validation macro F1
  only, with the test set scored once after selection is final. Choosing
  hyperparameters against the test set would make the final reported
  number optimistic in a way that doesn't hold up on genuinely new data.
- **2026-08-13** — The text vectorizer is fit once per target and reused
  across the tuning grid instead of being refit inside a full pipeline
  per trial. Vectorizing 550k+ documents is the expensive step; only the
  classifier changes between grid points, so refitting it every trial
  wastes runtime without changing the result.
