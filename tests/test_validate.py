"""
Tests for the validation stage: hard business-rule violations are
quarantined out of the processed dataset, while referential and
duplication issues are surfaced as quality metrics without dropping
rows.
"""

import pandas as pd

from src.data.validate import check_dataset


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


def test_valid_row_passes_through_unmodified() -> None:
    df = pd.DataFrame([make_row()])

    valid_df, report = check_dataset(df)

    assert report["rows_in"] == 1
    assert report["rows_valid"] == 1
    assert report["rows_rejected"] == 0
    assert len(valid_df) == 1


def test_duplicate_tweet_id_is_rejected() -> None:
    df = pd.DataFrame([make_row(tweet_id=1), make_row(tweet_id=1)])

    valid_df, report = check_dataset(df)

    assert report["rows_rejected"] >= 1
    assert valid_df["tweet_id"].is_unique


def test_empty_text_is_rejected() -> None:
    df = pd.DataFrame([make_row(tweet_id=1), make_row(tweet_id=2, text="")])

    valid_df, report = check_dataset(df)

    assert report["rows_valid"] == 1
    assert report["rows_rejected"] == 1


def test_out_of_range_date_is_rejected() -> None:
    far_future = pd.Timestamp("2999-01-01", tz="UTC")
    df = pd.DataFrame([make_row(tweet_id=1), make_row(tweet_id=2, created_at=far_future)])

    valid_df, report = check_dataset(df)

    assert report["rows_valid"] == 1
    assert report["rows_rejected"] == 1


def test_broken_response_link_is_a_metric_not_a_rejection() -> None:
    df = pd.DataFrame(
        [make_row(tweet_id=1, in_response_to_tweet_id=999.0)]
    )

    valid_df, report = check_dataset(df)

    assert report["rows_valid"] == 1
    assert report["rows_rejected"] == 0
    assert report["broken_response_links"] == 1
    assert report["broken_response_link_rate"] == 1.0


def test_duplicate_author_text_pair_is_a_metric_not_a_rejection() -> None:
    df = pd.DataFrame(
        [
            make_row(tweet_id=1, author_id="sprintcare", text="same text"),
            make_row(tweet_id=2, author_id="sprintcare", text="same text"),
        ]
    )

    valid_df, report = check_dataset(df)

    assert report["rows_valid"] == 2
    assert report["duplicate_author_text_pairs"] == 1
