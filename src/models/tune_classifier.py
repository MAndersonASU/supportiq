"""
Hyperparameter tuning for the category and priority classifiers. The
text vectorizer is fit once per target and reused across the grid (the
expensive step), and only the classifier hyperparameters are swept.
Each configuration is scored on the validation set; the test set is
touched exactly once per target, after the winning configuration is
already chosen, so model selection never contaminates the final number.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.models.split_data import stratified_split
from src.models.train_classifier import (
    DEFAULT_INPUT_PATH,
    EXPERIMENT_NAME,
    LOGREG_MAX_ITER,
    MLFLOW_DB_PATH,
    MODEL_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    evaluate,
)

DEFAULT_REPORT_PATH = Path("data/processed/tuning_report.json")

PARAM_GRID = [
    {"C": c, "class_weight": cw}
    for c in (0.1, 1.0, 10.0)
    for cw in (None, "balanced")
]


def select_best_trial_index(trials: list[dict]) -> int:
    return max(range(len(trials)), key=lambda i: trials[i]["val_macro_f1"])


def tune_target(
    target_col: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=3,
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train, y_val, y_test = train_df[target_col], val_df[target_col], test_df[target_col]

    trials: list[dict] = []
    fitted_models: list[LogisticRegression] = []

    for params in PARAM_GRID:
        with mlflow.start_run(run_name=f"{target_col}-tune-C{params['C']}-{params['class_weight']}"):
            clf = LogisticRegression(max_iter=LOGREG_MAX_ITER, **params)
            clf.fit(X_train, y_train)
            val_metrics = evaluate(y_val, clf.predict(X_val))

            mlflow.log_param("target", target_col)
            mlflow.log_params(params)
            mlflow.log_metrics(
                {"val_macro_f1": val_metrics["macro_f1"], "val_accuracy": val_metrics["accuracy"]}
            )

            trials.append({"params": params, "val_macro_f1": val_metrics["macro_f1"]})
            fitted_models.append(clf)

    best_index = select_best_trial_index(trials)
    best_params = trials[best_index]["params"]
    best_model = fitted_models[best_index]

    with mlflow.start_run(run_name=f"{target_col}-tuned-best"):
        test_metrics = evaluate(y_test, best_model.predict(X_test))

        mlflow.log_param("target", target_col)
        mlflow.log_params(best_params)
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_metrics.items() if k != "per_class"}
        )

        pipeline = Pipeline([("tfidf", vectorizer), ("clf", best_model)])
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_DIR / f"{target_col}_classifier.joblib")

    return {
        "trials": trials,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "best_val_macro_f1": trials[best_index]["val_macro_f1"],
        "test_metrics": test_metrics,
    }


def run(
    input_path: Path = DEFAULT_INPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    df = pd.read_parquet(input_path)
    train_df, val_df, test_df = stratified_split(df, stratify_col="category")

    report = {
        "category": tune_target("category", train_df, val_df, test_df),
        "priority": tune_target("priority", train_df, val_df, test_df),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run()
    summary = {
        target: {
            "best_params": data["best_params"],
            "best_val_macro_f1": data["best_val_macro_f1"],
            "test_macro_f1": data["test_metrics"]["macro_f1"],
        }
        for target, data in report.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
