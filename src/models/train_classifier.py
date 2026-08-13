"""
Baseline ticket classifiers: TF-IDF + logistic regression for category
and for priority, each compared against a majority-class baseline and
tracked in MLflow. Trained on weak-supervision labels
(src/models/label_tickets.py) rather than human-annotated ground truth —
these metrics measure how well the model generalizes past the exact
keyword rules that produced the labels, not agreement with a true
category or priority, since no ground truth exists for this dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.models.split_data import stratified_split

DEFAULT_INPUT_PATH = Path("data/processed/labeled_tickets.parquet")
DEFAULT_REPORT_PATH = Path("data/processed/model_evaluation_report.json")
MODEL_DIR = Path("models")
MLFLOW_DB_PATH = Path("mlflow.db")
EXPERIMENT_NAME = "ticket-triage"

TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_MAX_ITER = 1000


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    min_df=3,
                ),
            ),
            ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER, class_weight="balanced")),
        ]
    )


def evaluate(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(stats["precision"]),
                "recall": float(stats["recall"]),
                "f1": float(stats["f1-score"]),
                "support": int(stats["support"]),
            }
            for label, stats in report.items()
            if label not in ("accuracy", "macro avg", "weighted avg")
        },
    }


def train_and_evaluate(
    target_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train = train_df["text"], train_df[target_col]
    X_val, y_val = val_df["text"], val_df[target_col]
    X_test, y_test = test_df["text"], test_df[target_col]

    results: dict = {}

    with mlflow.start_run(run_name=f"{target_col}-majority-baseline"):
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_metrics = evaluate(y_test, baseline.predict(X_test))
        mlflow.log_param("target", target_col)
        mlflow.log_param("model", "majority-baseline")
        mlflow.log_metrics(
            {"accuracy": baseline_metrics["accuracy"], "macro_f1": baseline_metrics["macro_f1"]}
        )
        results["baseline"] = baseline_metrics

    with mlflow.start_run(run_name=f"{target_col}-tfidf-logreg"):
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        val_metrics = evaluate(y_val, pipeline.predict(X_val))
        test_metrics = evaluate(y_test, pipeline.predict(X_test))

        mlflow.log_param("target", target_col)
        mlflow.log_param("model", "tfidf+logreg")
        mlflow.log_param("max_features", TFIDF_MAX_FEATURES)
        mlflow.log_param("ngram_range", str(TFIDF_NGRAM_RANGE))
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics(
            {f"val_{k}": v for k, v in val_metrics.items() if k != "per_class"}
        )
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_metrics.items() if k != "per_class"}
        )
        mlflow.sklearn.log_model(pipeline, name="model")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_DIR / f"{target_col}_classifier.joblib")

        results["val"] = val_metrics
        results["test"] = test_metrics

    return results


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    df = pd.read_parquet(input_path)
    train_df, val_df, test_df = stratified_split(df, stratify_col="category")

    report = {
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "category": train_and_evaluate("category", train_df, val_df, test_df),
        "priority": train_and_evaluate("priority", train_df, val_df, test_df),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run()
    summary = {
        "split_sizes": report["split_sizes"],
        "category_baseline_macro_f1": report["category"]["baseline"]["macro_f1"],
        "category_test_macro_f1": report["category"]["test"]["macro_f1"],
        "priority_baseline_macro_f1": report["priority"]["baseline"]["macro_f1"],
        "priority_test_macro_f1": report["priority"]["test"]["macro_f1"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
