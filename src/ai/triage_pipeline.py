"""
End-to-end triage pipeline: classifies an incoming ticket (category,
priority) using the production models from the MLflow registry, then
drafts a grounded resolution with the RAG assistant. Ties Phase 2's
classifiers and Phase 3's resolution assistant into the single product
surface the project set out to build. Loading models via their registry
alias (rather than the local joblib files directly) exercises the
registry in an actual inference path, not just at training time.
"""

from __future__ import annotations

import json
import sys

import mlflow
import mlflow.sklearn

from src.ai.rag_assistant import generate_response
from src.models.train_classifier import MLFLOW_DB_PATH

MODEL_ALIAS_URIS = {
    "category": "models:/ticket-category-classifier@production",
    "priority": "models:/ticket-priority-classifier@production",
}

_classifiers: dict[str, object] = {}


def get_classifier(target_col: str):
    if target_col not in _classifiers:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        _classifiers[target_col] = mlflow.sklearn.load_model(MODEL_ALIAS_URIS[target_col])
    return _classifiers[target_col]


def triage(ticket_text: str) -> dict:
    predicted_category = get_classifier("category").predict([ticket_text])[0]
    predicted_priority = get_classifier("priority").predict([ticket_text])[0]

    resolution = generate_response(ticket_text)

    return {
        "ticket_text": ticket_text,
        "predicted_category": predicted_category,
        "predicted_priority": predicted_priority,
        "draft_reply": resolution["reply"],
        "cited_ticket_ids": resolution["cited_ticket_ids"],
        "needs_human_escalation": resolution["needs_human_escalation"],
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ticket_text = (
        sys.argv[1] if len(sys.argv) > 1 else "My order never arrived and it's been two weeks."
    )
    result = triage(ticket_text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
