"""
Tests for knowledge-base construction: a ticket is paired with the first
brand reply in its thread by creation time, and tickets with no
reconstructable reply are dropped rather than kept with a null
resolution.
"""

import pandas as pd

from src.ai.build_knowledge_base import build_knowledge_base


def make_tweet(**overrides: object) -> dict:
    row = {
        "tweet_id": 1,
        "author_id": "customer1",
        "inbound": True,
        "created_at": pd.Timestamp("2017-10-31T00:00:00", tz="UTC"),
        "in_response_to_tweet_id": None,
        "text_clean": "root message",
    }
    row.update(overrides)
    return row


def make_labeled_row(**overrides: object) -> dict:
    row = {
        "ticket_id": 1,
        "category": "Technical Support",
        "priority": "Low",
        "text": "root message",
    }
    row.update(overrides)
    return row


def test_ticket_paired_with_first_brand_reply_by_time():
    tweets_df = pd.DataFrame(
        [
            make_tweet(tweet_id=1),
            make_tweet(
                tweet_id=2,
                author_id="brandco",
                inbound=False,
                created_at=pd.Timestamp("2017-10-31T00:10:00", tz="UTC"),
                in_response_to_tweet_id=1,
                text_clean="later reply",
            ),
            make_tweet(
                tweet_id=3,
                author_id="brandco",
                inbound=False,
                created_at=pd.Timestamp("2017-10-31T00:05:00", tz="UTC"),
                in_response_to_tweet_id=1,
                text_clean="earlier reply",
            ),
        ]
    )
    labeled_df = pd.DataFrame([make_labeled_row(ticket_id=1)])

    kb, report = build_knowledge_base(tweets_df, labeled_df)

    assert len(kb) == 1
    assert kb.iloc[0]["resolution_text"] == "earlier reply"
    assert report["resolutions_found"] == 1
    assert report["resolutions_missing"] == 0


def test_ticket_with_no_reply_is_dropped():
    tweets_df = pd.DataFrame([make_tweet(tweet_id=1)])
    labeled_df = pd.DataFrame([make_labeled_row(ticket_id=1)])

    kb, report = build_knowledge_base(tweets_df, labeled_df)

    assert len(kb) == 0
    assert report["resolutions_missing"] == 1


def test_output_columns_match_expected_schema():
    tweets_df = pd.DataFrame(
        [
            make_tweet(tweet_id=1),
            make_tweet(
                tweet_id=2,
                author_id="brandco",
                inbound=False,
                created_at=pd.Timestamp("2017-10-31T00:10:00", tz="UTC"),
                in_response_to_tweet_id=1,
                text_clean="the fix",
            ),
        ]
    )
    labeled_df = pd.DataFrame([make_labeled_row(ticket_id=1)])

    kb, _ = build_knowledge_base(tweets_df, labeled_df)

    assert set(kb.columns) == {
        "ticket_id",
        "category",
        "priority",
        "customer_message",
        "resolution_text",
    }
