"""
Tests for the promotion-decision policy: rejects insufficient lift over
baseline, promotes the first qualifying version directly when there's no
incumbent, promotes a version that matches or beats the incumbent, and
holds a version at staging when it regresses against production.
"""

from src.models.register_model import MIN_IMPROVEMENT_OVER_BASELINE, decide_promotion


def test_rejected_when_lift_over_baseline_is_too_small():
    decision = decide_promotion(
        test_macro_f1=0.30,
        baseline_macro_f1=0.28,
        production_macro_f1=None,
    )

    assert decision["outcome"] == "rejected"


def test_promoted_directly_when_no_incumbent_exists():
    decision = decide_promotion(
        test_macro_f1=0.90,
        baseline_macro_f1=0.10,
        production_macro_f1=None,
    )

    assert decision["outcome"] == "promoted-to-production"


def test_promoted_when_matching_or_beating_incumbent():
    decision = decide_promotion(
        test_macro_f1=0.95,
        baseline_macro_f1=0.10,
        production_macro_f1=0.90,
    )

    assert decision["outcome"] == "promoted-to-production"


def test_staged_only_when_it_regresses_against_incumbent():
    decision = decide_promotion(
        test_macro_f1=0.85,
        baseline_macro_f1=0.10,
        production_macro_f1=0.90,
    )

    assert decision["outcome"] == "staged-only"


def test_lift_just_above_minimum_threshold_qualifies():
    baseline = 0.30
    decision = decide_promotion(
        test_macro_f1=baseline + MIN_IMPROVEMENT_OVER_BASELINE + 0.001,
        baseline_macro_f1=baseline,
        production_macro_f1=None,
    )

    assert decision["outcome"] == "promoted-to-production"


def test_lift_just_below_minimum_threshold_is_rejected():
    baseline = 0.30
    decision = decide_promotion(
        test_macro_f1=baseline + MIN_IMPROVEMENT_OVER_BASELINE - 0.001,
        baseline_macro_f1=baseline,
        production_macro_f1=None,
    )

    assert decision["outcome"] == "rejected"
