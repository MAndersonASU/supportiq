"""
Tests for ticket feature engineering: thread reconstruction across reply
chains, exclusion of brand-initiated threads, and the resolved /
first-response features derived from each thread.
"""

import pandas as pd

from src.features.build_ticket_features import assign_thread_roots, build_ticket_features


def make_row(**overrides: object) -> dict:
    row = {
        "tweet_id": 1,
        "author_id": "customer1",
        "inbound": True,
        "created_at": pd.Timestamp("2017-10-31T00:00:00", tz="UTC"),
        "in_response_to_tweet_id": None,
        "text_clean": "help me",
    }
    row.update(overrides)
    return row


def test_assign_thread_roots_follows_chain_to_the_original_tweet():
    df = pd.DataFrame(
        [
            make_row(tweet_id=1, in_response_to_tweet_id=None),
            make_row(tweet_id=2, inbound=False, in_response_to_tweet_id=1),
            make_row(tweet_id=3, in_response_to_tweet_id=2),
        ]
    )

    roots = assign_thread_roots(df)

    assert list(roots) == [1, 1, 1]


def test_assign_thread_roots_treats_broken_link_as_its_own_root():
    df = pd.DataFrame([make_row(tweet_id=5, in_response_to_tweet_id=999)])

    roots = assign_thread_roots(df)

    assert list(roots) == [5]


def test_resolved_ticket_has_first_response_time():
    df = pd.DataFrame(
        [
            make_row(
                tweet_id=1,
                created_at=pd.Timestamp("2017-10-31T00:00:00", tz="UTC"),
            ),
            make_row(
                tweet_id=2,
                author_id="brandco",
                inbound=False,
                created_at=pd.Timestamp("2017-10-31T00:05:00", tz="UTC"),
                in_response_to_tweet_id=1,
                text_clean="how can we help",
            ),
        ]
    )

    tickets, report = build_ticket_features(df)

    assert len(tickets) == 1
    ticket = tickets.iloc[0]
    assert ticket["ticket_id"] == 1
    assert ticket["resolved"] == True
    assert ticket["num_messages"] == 2
    assert ticket["first_response_seconds"] == 300.0
    assert ticket["brand_id"] == "brandco"
    assert report["tickets_extracted"] == 1
    assert report["tickets_with_brand_response"] == 1


def test_unresolved_ticket_has_no_first_response_time():
    df = pd.DataFrame([make_row(tweet_id=1)])

    tickets, report = build_ticket_features(df)

    assert len(tickets) == 1
    assert tickets.iloc[0]["resolved"] == False
    assert pd.isna(tickets.iloc[0]["first_response_seconds"])


def test_brand_initiated_thread_is_not_a_ticket():
    df = pd.DataFrame(
        [
            make_row(tweet_id=1, author_id="brandco", inbound=False),
            make_row(tweet_id=2, in_response_to_tweet_id=1),
        ]
    )

    tickets, report = build_ticket_features(df)

    assert len(tickets) == 0
    assert report["tickets_extracted"] == 0
