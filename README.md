# SupportIQ

An end-to-end AI engineering project: a customer support ticket platform that
combines a production-style data pipeline, a classical ML triage model, and a
Claude-powered retrieval-augmented resolution assistant.

Built incrementally, one pipeline stage at a time, with each stage documented
as it's built. See [`docs/engineering-log/`](docs/engineering-log/) for the
day-by-day build log and [`docs/architecture.md`](docs/architecture.md) for
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
├── features/   feature engineering
├── models/     training, evaluation, model registry
├── ai/         embeddings, vector store, RAG, agent logic
└── serving/    FastAPI application
tests/          unit and pipeline tests
docs/           architecture notes and the daily engineering log
```

## Status

Actively in progress. See the [engineering log](docs/engineering-log/) for
current phase and the most recent entry for what's done so far.

## License

MIT — see [LICENSE](LICENSE).
