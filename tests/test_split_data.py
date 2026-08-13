"""
Tests for the stratified train/val/test split: proportions land close to
target, every class appears in every split, no ticket appears twice, and
the split is deterministic given the same random state.
"""

import pandas as pd

from src.models.split_data import stratified_split


def make_df(n_per_class: int = 200) -> pd.DataFrame:
    rows = []
    ticket_id = 0
    for category in ["A", "B", "C"]:
        for _ in range(n_per_class):
            rows.append({"ticket_id": ticket_id, "category": category, "text": f"ticket {ticket_id}"})
            ticket_id += 1
    return pd.DataFrame(rows)


def test_split_proportions_are_approximately_70_15_15():
    df = make_df()
    train_df, val_df, test_df = stratified_split(df)

    total = len(df)
    assert abs(len(train_df) / total - 0.70) < 0.02
    assert abs(len(val_df) / total - 0.15) < 0.02
    assert abs(len(test_df) / total - 0.15) < 0.02


def test_every_class_present_in_every_split():
    df = make_df()
    train_df, val_df, test_df = stratified_split(df)

    for split in (train_df, val_df, test_df):
        assert set(split["category"]) == {"A", "B", "C"}


def test_no_ticket_appears_in_more_than_one_split():
    df = make_df()
    train_df, val_df, test_df = stratified_split(df)

    train_ids = set(train_df["ticket_id"])
    val_ids = set(val_df["ticket_id"])
    test_ids = set(test_df["ticket_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(df)


def test_split_is_deterministic_given_same_random_state():
    df = make_df()
    train_a, _, _ = stratified_split(df, random_state=7)
    train_b, _, _ = stratified_split(df, random_state=7)

    assert sorted(train_a["ticket_id"]) == sorted(train_b["ticket_id"])
