"""
Tests for the tuning-trial selection logic: picks the configuration with
the highest validation macro F1, and breaks ties by keeping the first
one encountered (the earlier, simpler configuration in the grid).
"""

from src.models.tune_classifier import select_best_trial_index


def test_selects_highest_val_macro_f1():
    trials = [
        {"params": {"C": 0.1}, "val_macro_f1": 0.40},
        {"params": {"C": 1.0}, "val_macro_f1": 0.55},
        {"params": {"C": 10.0}, "val_macro_f1": 0.50},
    ]

    assert select_best_trial_index(trials) == 1


def test_tie_breaks_to_first_occurrence():
    trials = [
        {"params": {"C": 0.1}, "val_macro_f1": 0.50},
        {"params": {"C": 1.0}, "val_macro_f1": 0.50},
    ]

    assert select_best_trial_index(trials) == 0
