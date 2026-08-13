"""
Tests for the cleaning stage: text normalization, the missing-value
policy check, and dropping of rows that are empty after normalization.
"""

import pandas as pd

from src.data.clean import clean_dataset, normalize_text


def make_row(**overrides: object) -> dict:
    row = {
        "tweet_id": 1,
        "author_id": "sprintcare",
        "inbound": False,
        "created_at": pd.Timestamp("2017-10-31", tz="UTC"),
        "text": "hello there",
        "response_tweet_id": None,
        "in_response_to_tweet_id": None,
    }
    row.update(overrides)
    return row


def test_normalize_text_collapses_whitespace():
    assert normalize_text("hello   there\n\tfriend") == "hello there friend"


def test_normalize_text_applies_unicode_normalization():
    assert normalize_text("café") == "café"


def test_clean_dataset_adds_normalized_column_and_keeps_original():
    df = pd.DataFrame([make_row(text="  hi   there  ")])

    cleaned_df, report = clean_dataset(df)

    assert cleaned_df.loc[0, "text"] == "  hi   there  "
    assert cleaned_df.loc[0, "text_clean"] == "hi there"
    assert report["rows_out"] == 1


def test_row_empty_after_normalization_is_dropped():
    df = pd.DataFrame([make_row(tweet_id=1), make_row(tweet_id=2, text="   \n\t  ")])

    cleaned_df, report = clean_dataset(df)

    assert report["rows_in"] == 2
    assert report["rows_out"] == 1
    assert report["rows_dropped_empty_after_cleaning"] == 1
    assert list(cleaned_df["tweet_id"]) == [1]


def test_null_in_nullable_linkage_field_is_not_flagged():
    df = pd.DataFrame([make_row(in_response_to_tweet_id=None, response_tweet_id=None)])

    _, report = clean_dataset(df)

    assert report["unexpected_nulls_in_required_columns"] == 0


def test_null_in_required_field_is_flagged():
    df = pd.DataFrame([make_row(author_id=None)])

    _, report = clean_dataset(df)

    assert report["unexpected_nulls_in_required_columns"] == 1
