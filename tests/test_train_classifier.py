"""
Tests for the classifier evaluation helper: correct accuracy/macro-F1
on a known small example, and JSON-serializable per-class stats (numpy
scalar types cast to plain Python types).
"""

from src.models.train_classifier import evaluate


def test_perfect_predictions_score_1_0():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]

    metrics = evaluate(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_all_wrong_predictions_score_below_perfect():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["B", "B", "A", "A"]

    metrics = evaluate(y_true, y_pred)

    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_per_class_stats_use_plain_python_types():
    y_true = ["A", "A", "B"]
    y_pred = ["A", "B", "B"]

    metrics = evaluate(y_true, y_pred)

    assert isinstance(metrics["per_class"]["A"]["support"], int)
    assert isinstance(metrics["per_class"]["A"]["precision"], float)
    assert metrics["per_class"]["A"]["support"] == 2
    assert metrics["per_class"]["B"]["support"] == 1
