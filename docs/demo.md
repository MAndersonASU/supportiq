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
  "draft_reply": "Apologies! Orders placed with us are basically shipped and delivered as per the estimates provided. 1/2\nWe're sorry to hear you haven't received your order. Can you please DM us with some further details regarding your query, such as the order number or any tracking information? We'll do our best to help.",
  "cited_ticket_ids": ["315238", "2886650"],
  "link_redacted": false,
  "mention_redacted": false,
  "completed_action_claimed": false,
  "ticket_severity_signaled": false,
  "needs_human_escalation": true
}
```

The four `*_redacted`/`*_claimed`/`*_signaled` fields are independent
safety checks, each documented as its own investigation in
[`docs/`](.) — see `docs/rag-link-fabrication.md`,
`docs/rag-completed-action-claims.md`, `docs/rag-prompt-injection.md`,
and `docs/rag-escalation-reliability.md`. All four are `false` here,
which is the common case — most individual responses don't trigger any
of them.

## Retrieval step, and a real hallucinated citation caught live

`rag_assistant.py` run directly, showing what the resolution assistant
actually retrieves before generating a reply — the three nearest past
tickets by embedding distance, each tagged with its resolved category:

```
$ python -m src.ai.rag_assistant "What are your customer service hours?"

Query: What are your customer service hours?

Draft reply:
Hi! We are here to help. Our customer service hours are available 24/7. Please send us a DM with any further questions or concerns.

Needs human escalation: False
Cited tickets: []
Hallucinated citations (dropped): ['We sure are and we are here to help! 24/7. :) Need help? Send me a DM! https://t.co/dPHUAru2pc *MikeVance', '[24/7] We have sent you a Direct Message via Twitter with further instructions.']

Retrieved examples:
  ticket 1019996 (Customer Service Complaint, distance 0.747)
  ticket 2222752 (Customer Service Complaint, distance 0.777)
  ticket 1031672 (Customer Service Complaint, distance 0.824)
```

This particular run shows `verify_citations` doing real work, not just
existing in theory: the model put two retrieved resolution snippets
into the `cited_ticket_ids` field instead of actual ticket numbers.
Neither matches anything that was really retrieved, so both are
classified as hallucinated and dropped before the reply is returned —
the final `cited_ticket_ids` sent to the caller is empty, not the
fabricated text.

The evaluation harness (`src/ai/evaluate_rag.py`) measures this kind of
event systematically across a fixed 6-query set rather than relying on
any single example. Hallucination rate isn't a fixed number — repeated
runs against the same 6 queries have come back anywhere from 0/6 to
1/6, since the local model's generation isn't deterministic — which is
exactly why this is tracked automatically rather than checked once and
assumed. Full numbers in
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
  "draft_reply": "Hello, Sorry to hear you haven't received your order. We can assist you further, please can you DM us with some details regarding your query. Thanks, CD",
  "cited_ticket_ids": ["1920576", "234126", "2886650"],
  "link_redacted": false,
  "mention_redacted": false,
  "completed_action_claimed": false,
  "ticket_severity_signaled": false,
  "needs_human_escalation": false
}
```

Structured logs for that request — note the ticket text itself never
appears, only its length:

```
{"timestamp": "2026-08-14T14:56:56", "level": "INFO", "logger": "supportiq.triage", "message": "ticket triaged", "text_length": 56, "predicted_category": "Order / Delivery / Refund", "predicted_priority": "High", "link_redacted": false, "mention_redacted": false, "completed_action_claimed": false, "ticket_severity_signaled": false, "needs_human_escalation": false}
{"timestamp": "2026-08-14T14:56:56", "level": "INFO", "logger": "supportiq.request", "message": "request completed", "path": "/triage", "method": "POST", "status_code": 200, "duration_ms": 12623.09}
```

The 12.6s duration is a cold-start cost — first request after startup
loads the embedding model, vector index, and both classifiers. See
[`architecture.md`](architecture.md) for the caching behavior and the
Docker cold-start numbers.

## Web UI

A static single-page interface is served at `GET /` — see
`src/serving/static/index.html` and the "Serving layer" section of
[`architecture.md`](architecture.md) for what it does and how it was
verified.
