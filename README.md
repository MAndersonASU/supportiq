# SupportIQ

[![CI](https://github.com/MAndersonASU/supportiq/actions/workflows/ci.yml/badge.svg)](https://github.com/MAndersonASU/supportiq/actions/workflows/ci.yml)

An AI-powered support ticket platform that automatically triages incoming
tickets and drafts grounded reply suggestions for support agents — built
end-to-end, from raw data to a running API, with the same rigor a
production AI engineering team would apply.

## What this is

Support teams field a constant stream of tickets that need three things
before an agent can act on them: what category is this, how urgent is it,
and what's the fastest path to a good answer. That triage work is usually
manual — an agent reads the ticket, decides where it belongs, and often
searches past cases for a similar resolution before replying.

SupportIQ automates the first two steps and assists the third:

1. **Classifies** every incoming ticket by category (Billing, Technical
   Support, Account Access, Order/Delivery, and more) and priority
   (Low/Medium/High), using a trained machine learning model — no manual
   sorting required.
2. **Drafts a grounded reply suggestion** by retrieving the most similar
   past tickets and their real resolutions, then having a locally-run AI
   model draft a response based on those — with built-in checks that
   catch and remove anything the model states without real grounding
   before it ever reaches an agent.

An agent still reviews and sends the final reply. SupportIQ's job is to
get them to a strong starting point faster, not to auto-respond
unsupervised — and everywhere the AI model could produce something
ungrounded or unsafe, that output is checked in code, not just assumed
correct because the model was told to behave.

## Why this matters for a business

- **Faster first response.** Every ticket is categorized and prioritized
  instantly instead of sitting in a general queue for manual triage.
- **Consistency at scale.** The same ticket, phrased the same way, gets
  routed the same way every time — not dependent on which agent happens
  to pick it up.
- **Less time searching, more time resolving.** Instead of an agent
  manually digging through past tickets for "have we seen this before,"
  the system retrieves the closest real precedents automatically and
  drafts a starting reply grounded in them.
- **Safety-checked by design, not by trust.** Every AI-drafted claim is
  verified before it's shown to anyone: citations the model claims are
  checked against what was actually retrieved, any link or customer
  handle the model generates is stripped rather than assumed safe, and
  a reply claiming an action was already taken (a refund issued, a DM
  already sent) — or one that had to be redacted — is routed to a human
  instead of going out unreviewed. Every one of these protections exists
  because I found the underlying problem by testing the running system,
  including a real prompt-injection attempt that pulled another
  customer's data into a draft reply — see
  [`docs/rag-link-fabrication.md`](docs/rag-link-fabrication.md),
  [`docs/rag-completed-action-claims.md`](docs/rag-completed-action-claims.md),
  and [`docs/rag-prompt-injection.md`](docs/rag-prompt-injection.md)
  for three examples, start to finish.

## What I built

The full pipeline, in the order it runs:

1. **Data engineering** — ~2.8M raw support tweets ingested, schema-
   validated, cleaned, and reconstructed into 789,547 individual support
   tickets by rebuilding each customer's reply thread.
2. **Machine learning triage model** — a trained classifier predicts
   category and priority for any new ticket, tracked and versioned
   through an experiment-tracking and model registry system (MLflow) the
   way a production ML team manages model versions and promotion.
3. **AI resolution assistant** — retrieves the most similar past tickets
   from a vector database and uses a locally-run AI model to draft a
   grounded reply, citing which past tickets informed it. Runs entirely
   on free, local infrastructure — no paid API required.
4. **API and deployment** — the system runs behind a REST API (FastAPI),
   containerized with Docker, with automated tests and a build check
   running on every code change (CI/CD via GitHub Actions).
5. **Safety and monitoring** — every AI-drafted claim is checked, not
   trusted: cited tickets are verified against what was actually
   retrieved, hallucination rates are tracked over a fixed evaluation
   set, and generated links are stripped before a reply is ever returned.

Every stage, and every real bug found and fixed along the way, is
documented in [`docs/`](docs/) — see
[`docs/engineering-log/`](docs/engineering-log/) for the full build
history and [`docs/architecture.md`](docs/architecture.md) for the system
design and every engineering decision behind it.

## Results

- **789,547 tickets** reconstructed from ~2.8M raw tweets by rebuilding
  reply threads, cleaned and feature-engineered through a validated
  pipeline (data contract enforced at ingestion, business rules enforced
  separately, both reported and tracked).
- **Category classifier**: 0.987 test macro F1. Reflects the model
  reconstructing the weak-supervision labeling rule closely (there's no
  independent ground truth to check generalization against) — the
  honest read on this number, not just the number itself, is in
  `docs/architecture.md`.
- **Priority classifier**: 0.429 test macro F1, with a diagnosed,
  unfixed cause: part of the priority label depends on punctuation/
  capitalization that the default TF-IDF tokenizer strips before the
  model ever sees it. Documented as a real limitation rather than
  patched around.
- **RAG resolution assistant**: grounded replies over 96,536 embedded
  past resolutions, with citation claims cross-checked against what was
  actually retrieved — 16.7% hallucination rate on a fixed 6-query
  evaluation set, caught and reported rather than assumed to be zero.
- **Multiple real bugs, one false-positive test result, and a documented
  AI-safety finding** — all caught by actually running the built system,
  not by reading the code. Full catalogue in
  [`docs/architecture.md`](docs/architecture.md#design-decisions-log);
  the link-fabrication finding has its own writeup at
  [`docs/rag-link-fabrication.md`](docs/rag-link-fabrication.md),
  including a measured reproduction rate and post-fix verification.

## Architecture

```
raw tickets → validation → cleaning → feature engineering → [ML model] → category/priority
                                              │
                                              ▼
                              embeddings → vector store → [RAG + local LLM] → draft resolution
```

Full breakdown, including a module-level diagram, in
[`docs/architecture.md`](docs/architecture.md).

## Project layout

```
src/
├── data/       ingestion, validation, cleaning
├── features/   thread reconstruction, ticket-level feature engineering
├── ai/         knowledge base, vector index, RAG resolution assistant, triage pipeline
├── models/     training, evaluation, model registry
└── serving/    FastAPI application, structured logging
tests/          unit and pipeline tests
docs/           architecture notes, the engineering log, and demo output
.github/workflows/  CI: lint, test, Docker build check on every push
data/
├── raw/        untouched source data (gitignored)
├── landing/    schema-conformant Parquet, pre-validation (gitignored)
├── validated/  business-rule-validated Parquet (gitignored; report tracked)
└── processed/  cleaned tweets, ticket_features.parquet, labeled_tickets.parquet
              (gitignored; reports and the DVC pointer file are tracked)
models/         trained classifier artifacts (gitignored, reproducible via train_classifier)
mlflow.db       local MLflow tracking store (gitignored)
data/vector_store/  local Chroma vector index (gitignored, reproducible via build_vector_index)
```

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Fetching the raw dataset requires a Kaggle API token at
`~/.kaggle/kaggle.json` (see [Kaggle API docs](https://www.kaggle.com/docs/api)):

```
.venv\Scripts\python -m kaggle datasets download -d thoughtvector/customer-support-on-twitter -f "twcs/twcs.csv" -p data/raw
```

Then run the pipeline stages in order:

```
.venv\Scripts\python -m src.data.ingest
.venv\Scripts\python -m src.data.validate
.venv\Scripts\python -m src.data.clean
.venv\Scripts\python -m src.features.build_ticket_features
.venv\Scripts\python -m src.models.label_tickets
.venv\Scripts\python -m src.models.train_classifier
.venv\Scripts\python -m src.models.tune_classifier
.venv\Scripts\python -m src.models.register_model
```

Registered models and their `staging`/`production` aliases are visible via `mlflow ui --backend-store-uri sqlite:///mlflow.db`, under the Models tab.

The RAG resolution assistant runs entirely locally — no external API calls. It needs [Ollama](https://ollama.com) installed with a local model pulled:

```
ollama pull llama3.2:3b
.venv\Scripts\python -m src.ai.build_knowledge_base
.venv\Scripts\python -m src.ai.build_vector_index
.venv\Scripts\python -m src.ai.rag_assistant "my order never arrived"
.venv\Scripts\python -m src.ai.evaluate_rag
.venv\Scripts\python -m src.ai.triage_pipeline "my order never arrived"
```

`rag_assistant` returns a validated structured object (reply, cited ticket IDs cross-checked against what was actually retrieved, an escalation flag) rather than free text, with any generated link stripped before it's returned — see [`docs/rag-link-fabrication.md`](docs/rag-link-fabrication.md). `triage_pipeline` combines the production classifiers with the resolution assistant into a single classify-then-draft call. It checks the MLflow registry's `production` alias for each model (confirming a version is actually promoted) but loads the model weights from the mounted `.joblib` file — see [Running with Docker](#running-with-docker) for why.

## Running the API

```
.venv\Scripts\python -m uvicorn src.serving.app:app --reload
```

`GET /health` for a liveness check, `POST /triage` with `{"text": "..."}` for a full classify-and-draft response, interactive docs at `/docs`. Models are not loaded at startup — the first request pays the one-time load cost (embedding model, vector index, classifiers); subsequent requests reuse the cached models.

## Running with Docker

Requires the training/registry pipeline to have already been run locally at least once (`models/`, `mlflow.db`, `mlruns/`, and `data/vector_store/` must exist — these are mounted into the container, not rebuilt inside it).

```
docker compose up -d
docker compose exec ollama ollama pull llama3.2:3b   # one-time, persists in a named volume
```

`GET http://localhost:8000/health` and `POST http://localhost:8000/triage` work the same as running locally. The app container waits for Ollama's healthcheck before starting. First request after a fresh container start is slow (cold model load — the embedding model downloads into a cached volume on first use, and the local LLM itself takes real time to generate on CPU); subsequent requests are faster.

The serving image installs `requirements-serving.txt`, not the full `requirements.txt`, to keep training-only and test-only tools out of the image. The model registry's `production` alias is still checked at container startup-time inference (confirming a production version is registered), but the actual model weights load from the mounted `.joblib` files: MLflow's local file-based artifact store records absolute host filesystem paths, which don't resolve inside a container.

The dataset ships with no ground-truth category/priority labels. `label_tickets` applies weak supervision (keyword labeling functions) rather than calling a paid LLM API; the category keywords were grounded by an exploratory clustering pass — see `src/models/explore_categories.py` and `docs/architecture.md`.

`train_classifier` trains a baseline TF-IDF + logistic regression classifier for category and for priority, tracked in MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db` to view runs locally). Read `docs/architecture.md` before trusting the category metric at face value — it's explained there.

The Anthropic SDK and a `.env` slot for `ANTHROPIC_API_KEY` are present
in the codebase but unused by any current pipeline stage — the RAG
assistant and weak-supervision labeling were both built against local,
free tooling instead (see `docs/architecture.md` for why), so nothing
in this project requires a paid API key to run.

The final ticket-level feature set is version-controlled with DVC (see
`data/processed/ticket_features.parquet.dvc`). Run `dvc pull` to fetch the
exact version referenced by that pointer file from the configured remote.

## Continuous integration

Every push and pull request against `main` runs lint (`ruff check`) and the
test suite, plus a Docker build check for the serving image — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

```
.venv\Scripts\python -m ruff check src/ tests/
.venv\Scripts\python -m pytest
```

## Status

Feature-complete: data pipeline, ML triage model, RAG resolution
assistant, FastAPI serving, Docker packaging, and CI are all built and
verified end to end. Still under active review — testing the deployed
system continues to surface real findings, each investigated,
documented, and fixed in place; see the
[engineering log](docs/engineering-log/) for the full history and
[`docs/demo.md`](docs/demo.md) for real output from a live run.

## License

MIT — see [LICENSE](LICENSE).
