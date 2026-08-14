"""
End-to-end triage pipeline: classifies an incoming ticket (category,
priority) using the production models, then drafts a grounded resolution
with the RAG assistant. Ties Phase 2's classifiers and Phase 3's
resolution assistant into the single product surface the project set
out to build.

The MLflow registry's `production` alias is still consulted at
inference time — that's a cheap metadata lookup confirming a production
version exists — but the actual model weights are loaded from the
mounted `.joblib` file, not via MLflow's artifact-store resolution.
MLflow's local file-based artifact store records absolute host
filesystem paths at logging time, which don't exist inside a container;
resolving them there fails, and fixing that properly would mean running
a real MLflow tracking server with its own artifact storage (S3 or
similar) instead of a local SQLite file, which is outside this
project's local-first, zero-budget scope.
"""

from __future__ import annotations

import json
import sys

import joblib
import mlflow
from mlflow import MlflowClient

from src.ai.rag_assistant import generate_response
from src.models.register_model import REGISTERED_MODEL_NAMES
from src.models.train_classifier import MLFLOW_DB_PATH, MODEL_DIR

_classifiers: dict[str, object] = {}


def get_classifier(target_col: str):
    if target_col not in _classifiers:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        model_name = REGISTERED_MODEL_NAMES[target_col]
        # Metadata-only lookup: confirms a production version is registered
        # and raises if not, without downloading any artifact.
        MlflowClient().get_model_version_by_alias(model_name, "production")

        model_path = MODEL_DIR / f"{target_col}_classifier.joblib"
        _classifiers[target_col] = joblib.load(model_path)
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
        "link_redacted": resolution["link_redacted"],
        "mention_redacted": resolution["mention_redacted"],
        "completed_action_claimed": resolution["completed_action_claimed"],
        "ticket_severity_signaled": resolution["ticket_severity_signaled"],
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
