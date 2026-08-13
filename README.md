# SupportIQ

An end-to-end AI engineering project: a customer support ticket platform that
combines a production-style data pipeline, a classical ML triage model, and a
Claude-powered retrieval-augmented resolution assistant.

Built incrementally, one pipeline stage at a time, with each stage documented
as it's completed. See [`docs/engineering-log/`](docs/engineering-log/) for
the development log and [`docs/architecture.md`](docs/architecture.md) for
the system design.

## Problem

Support teams triage incoming tickets (categorize, prioritize, route) and
then resolve them, often by searching past tickets and internal docs for a
similar case. SupportIQ automates both halves:

1. **Classify** an incoming ticket (category, priority) with a trained ML
   model.
2. **Assist resolution** by retrieving similar past tickets / knowledge-base
   entries and generating a grounded, cited draft response with an LLM.

## Architecture

```
raw tickets → validation → cleaning → feature engineering → [ML model] → category/priority
                                              │
                                              ▼
                                     embeddings → vector store → [RAG + Claude] → draft resolution
```

Full breakdown in [`docs/architecture.md`](docs/architecture.md).

## Project layout

```
src/
├── data/       ingestion, validation, cleaning
├── features/   thread reconstruction, ticket-level feature engineering
├── models/     training, evaluation, model registry
├── ai/         embeddings, vector store, RAG, agent logic
└── serving/    FastAPI application
tests/          unit and pipeline tests
docs/           architecture notes and the engineering log
data/
├── raw/        untouched source data (gitignored)
├── landing/    schema-conformant Parquet, pre-validation (gitignored)
├── validated/  business-rule-validated Parquet (gitignored; report tracked)
└── processed/  cleaned tweets and the final ticket_features.parquet
              (gitignored; reports and the DVC pointer file are tracked)
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
```

The final ticket-level feature set is version-controlled with DVC (see
`data/processed/ticket_features.parquet.dvc`). Run `dvc pull` to fetch the
exact version referenced by that pointer file from the configured remote.

## Status

Actively in progress. See the [engineering log](docs/engineering-log/) for
current phase and the most recent entry for what's done so far.

## License

MIT — see [LICENSE](LICENSE).
