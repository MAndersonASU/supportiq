"""
Model registry stage. Registers each tuned classifier as a new version in
the MLflow Model Registry and applies explicit promotion criteria via
registry aliases (the current MLflow registry API — the older stage-based
transitions are legacy). A version reaches the "staging" alias only if
it beats its majority-class baseline by a minimum margin, and reaches
"production" only if it also matches or beats whatever is currently in
production, or there is no incumbent yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.models.train_classifier import MLFLOW_DB_PATH, MODEL_DIR

DEFAULT_TUNING_REPORT_PATH = Path("data/processed/tuning_report.json")
DEFAULT_EVAL_REPORT_PATH = Path("data/processed/model_evaluation_report.json")
DEFAULT_REPORT_PATH = Path("data/processed/registry_report.json")

MIN_IMPROVEMENT_OVER_BASELINE = 0.05

REGISTERED_MODEL_NAMES = {
    "category": "ticket-category-classifier",
    "priority": "ticket-priority-classifier",
}


def decide_promotion(
    test_macro_f1: float,
    baseline_macro_f1: float,
    production_macro_f1: float | None,
) -> dict:
    lift = test_macro_f1 - baseline_macro_f1
    if lift < MIN_IMPROVEMENT_OVER_BASELINE:
        return {
            "outcome": "rejected",
            "reason": (
                f"macro F1 lift over baseline ({lift:.4f}) is below the "
                f"required minimum ({MIN_IMPROVEMENT_OVER_BASELINE})"
            ),
        }

    if production_macro_f1 is None:
        return {
            "outcome": "promoted-to-production",
            "reason": "no existing production version; first qualifying version promoted directly",
        }

    if test_macro_f1 >= production_macro_f1:
        return {
            "outcome": "promoted-to-production",
            "reason": f"matched or beat incumbent production macro F1 ({production_macro_f1:.4f})",
        }

    return {
        "outcome": "staged-only",
        "reason": f"below incumbent production macro F1 ({production_macro_f1:.4f}); held for review",
    }


def _get_production_macro_f1(client: MlflowClient, model_name: str) -> float | None:
    try:
        version = client.get_model_version_by_alias(model_name, "production")
    except MlflowException:
        return None
    return float(version.tags.get("test_macro_f1", "0"))


def register_and_promote(target_col: str, test_macro_f1: float, baseline_macro_f1: float) -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    client = MlflowClient()
    model_name = REGISTERED_MODEL_NAMES[target_col]
    model_path = MODEL_DIR / f"{target_col}_classifier.joblib"

    with mlflow.start_run(run_name=f"{target_col}-register"):
        mlflow.log_metric("test_macro_f1", test_macro_f1)
        mlflow.sklearn.log_model(
            joblib.load(model_path),
            name="model",
            registered_model_name=model_name,
        )

    versions = client.search_model_versions(f"name='{model_name}'")
    new_version = max(versions, key=lambda v: int(v.version))
    client.set_model_version_tag(model_name, new_version.version, "test_macro_f1", str(test_macro_f1))
    client.set_model_version_tag(model_name, new_version.version, "baseline_macro_f1", str(baseline_macro_f1))

    production_macro_f1 = _get_production_macro_f1(client, model_name)
    decision = decide_promotion(test_macro_f1, baseline_macro_f1, production_macro_f1)

    if decision["outcome"] in ("promoted-to-production", "staged-only"):
        client.set_registered_model_alias(model_name, "staging", new_version.version)
    if decision["outcome"] == "promoted-to-production":
        client.set_registered_model_alias(model_name, "production", new_version.version)

    return {
        "model_name": model_name,
        "version": new_version.version,
        "test_macro_f1": test_macro_f1,
        "baseline_macro_f1": baseline_macro_f1,
        **decision,
    }


def run(
    tuning_report_path: Path = DEFAULT_TUNING_REPORT_PATH,
    eval_report_path: Path = DEFAULT_EVAL_REPORT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    tuning_report = json.loads(tuning_report_path.read_text())
    eval_report = json.loads(eval_report_path.read_text())

    report = {}
    for target_col in ("category", "priority"):
        test_macro_f1 = tuning_report[target_col]["test_metrics"]["macro_f1"]
        baseline_macro_f1 = eval_report[target_col]["baseline"]["macro_f1"]
        report[target_col] = register_and_promote(target_col, test_macro_f1, baseline_macro_f1)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
