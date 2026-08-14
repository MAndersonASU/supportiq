# SupportIQ

[![CI](https://github.com/MAndersonASU/supportiq/actions/workflows/ci.yml/badge.svg)](https://github.com/MAndersonASU/supportiq/actions/workflows/ci.yml)

An end-to-end AI engineering project: a customer support ticket platform that
combines a production-style data pipeline, a classical ML triage model, and a
retrieval-augmented resolution assistant running entirely on local, free
infrastructure — no paid API in the loop.

Built incrementally, one pipeline stage at a time, with each stage documented
as it's completed. See [`docs/engineering-log/`](docs/engineering-log/) for
the development log, [`docs/architecture.md`](docs/architecture.md) for the
system design, [`docs/demo.md`](docs/demo.md) for real captured output, and
[`docs/rag-link-fabrication.md`](docs/rag-link-fabrication.md) for a real
safety finding and fix caught by live-testing the deployed API.

## Problem

Support teams triage incoming tickets (categorize, prioritize, route) and
then resolve them, often by searching past tickets and internal docs for a
similar case. SupportIQ automates both halves:

1. **Classify** an incoming ticket (category, priority) with a trained ML
   model.
2. **Assist resolution** by retrieving similar past tickets / knowledge-base
   entries and generating a grounded, cited draft response with an LLM.

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
- **Priority classifier**: 0.428 test macro F1, with a diagnosed,
  unfixed cause: part of the priority label depends on punctuation/
  capitalization that the default TF-IDF tokenizer strips before the
  model ever sees it. Documented as a real limitation rather than
  patched around.
- **RAG resolution assistant**: grounded replies over 96,536 embedded
  past resolutions, with citation claims cross-checked against what was
  actually retrieved — 16.7% hallucination rate on a fixed 6-query
  evaluation set, caught and reported rather than assumed to be zero.
- **Two real bugs and one false-positive test result** caught only by
  actually running the built system, not by reading the code — details
  in [`docs/architecture.md`](docs/architecture.md#design-decisions-log)
  and [`docs/demo.md`](docs/demo.md).

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

`rag_assistant` returns a validated structured object (reply, cited ticket IDs cross-checked against what was actually retrieved, an escalation flag) rather than free text. `triage_pipeline` combines the production classifiers with the resolution assistant into a single classify-then-draft call. It checks the MLflow registry's `production` alias for each model (confirming a version is actually promoted) but loads the model weights from the mounted `.joblib` file — see [Running with Docker](#running-with-docker) for why.

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
verified end to end. See the [engineering log](docs/engineering-log/)
for the full build history and [`docs/demo.md`](docs/demo.md) for real
output from a live run.

## License

MIT — see [LICENSE](LICENSE).
