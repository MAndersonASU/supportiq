# Demo

Real output from a local run of the full stack — the training pipeline
already executed once to produce `models/`, `mlflow.db`, and
`data/vector_store/`, then the serving layer queried directly. Nothing
below is hand-written; it's copied from actual command output.

## Classification + resolution draft

```
$ python -m src.ai.triage_pipeline "my order never arrived and I've been waiting two weeks"

{
  "ticket_text": "my order never arrived and I've been waiting two weeks",
  "predicted_category": "Order / Delivery / Refund",
  "predicted_priority": "High",
  "draft_reply": "Apologies! Orders placed with us are basically processed as per the estimates provided. Please speak with our team directly so they can help further.",
  "cited_ticket_ids": [],
  "needs_human_escalation": true
}
```

## Retrieval step

`rag_assistant.py` run directly, showing what the resolution assistant
actually retrieves before generating a reply — the three nearest past
tickets by embedding distance, each tagged with its resolved category:

```
$ python -m src.ai.rag_assistant "What are your customer service hours?"

Query: What are your customer service hours?

Draft reply:
Hi! We have sent you a Direct Message via Twitter with further instructions.

Needs human escalation: True
Cited tickets: []

Retrieved examples:
  ticket 1019996 (Customer Service Complaint, distance 0.747)
  ticket 2222752 (Customer Service Complaint, distance 0.777)
  ticket 1031672 (Customer Service Complaint, distance 0.824)
```

`cited_ticket_ids` is empty in both examples above — the model retrieved
relevant prior tickets but didn't explicitly reference their IDs in this
reply. The evaluation harness (`src/ai/evaluate_rag.py`) measures this
across a fixed query set rather than relying on any single example: of 6
test queries, one produced 3 citations, all real (0 hallucinated); the
aggregate hallucination rate — cited an ID that wasn't actually
retrieved — is 16.7%, tracked and reported, not hidden. Full numbers in
[`rag_evaluation_report.json`](../data/processed/rag_evaluation_report.json).

## Live API call

FastAPI service running locally (`uvicorn src.serving.app:app`):

```
$ curl http://127.0.0.1:8000/health
{"status":"ok"}

$ curl -X POST http://127.0.0.1:8000/triage \
    -H "Content-Type: application/json" \
    -d '{"text": "my order never arrived and I have been waiting two weeks"}'

{
  "ticket_text": "my order never arrived and I have been waiting two weeks",
  "predicted_category": "Order / Delivery / Refund",
  "predicted_priority": "High",
  "draft_reply": "Sorry to hear you haven't received your order. Please speak with our team directly so they can help further.",
  "cited_ticket_ids": [],
  "needs_human_escalation": true
}
```

Structured logs for that request — note the ticket text itself never
appears, only its length:

```
{"timestamp": "2026-08-13T21:13:46", "level": "INFO", "logger": "supportiq.triage", "message": "ticket triaged", "text_length": 56, "predicted_category": "Order / Delivery / Refund", "predicted_priority": "High", "needs_human_escalation": true}
{"timestamp": "2026-08-13T21:13:46", "level": "INFO", "logger": "supportiq.request", "message": "request completed", "path": "/triage", "method": "POST", "status_code": 200, "duration_ms": 17617.27}
```

The 17.6s duration is a cold-start cost — first request after startup
loads the embedding model, vector index, and both classifiers. See
[`architecture.md`](architecture.md) for the caching behavior and the
Docker cold-start numbers.
