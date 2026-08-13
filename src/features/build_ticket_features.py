"""
Feature engineering stage. The processed dataset is tweet-level; a
support ticket doesn't exist as a record yet. This module reconstructs
reply threads from the response-linkage fields, keeps only threads
opened by a customer (a real ticket, as opposed to a brand's own tweet
that happened to get replies), and aggregates each thread into one
ticket-level feature row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_PROCESSED_PATH = Path("data/processed/tweets.parquet")
DEFAULT_FEATURES_PATH = Path("data/processed/ticket_features.parquet")
DEFAULT_REPORT_PATH = Path("data/processed/feature_report.json")

FEATURE_COLUMNS = [
    "ticket_id",
    "customer_id",
    "brand_id",
    "opened_at",
    "text",
    "text_length",
    "word_count",
    "num_messages",
    "num_customer_messages",
    "num_brand_messages",
    "first_response_seconds",
    "resolved",
]


def assign_thread_roots(df: pd.DataFrame) -> pd.Series:
    parent_of = df.set_index("tweet_id")["in_response_to_tweet_id"].to_dict()
    root_cache: dict[int, int] = {}

    def find_root(tweet_id: int) -> int:
        path: list[int] = []
        current = tweet_id
        while current not in root_cache:
            if current in path:
                root_cache[current] = current
                break
            parent = parent_of.get(current)
            if parent is None or parent not in parent_of:
                root_cache[current] = current
                break
            path.append(current)
            current = int(parent)
        root = root_cache[current]
        for node in path:
            root_cache[node] = root
        return root

    return df["tweet_id"].map(find_root)


def build_ticket_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["thread_root_id"] = assign_thread_roots(df)

    thread_size = df.groupby("thread_root_id").agg(
        num_messages=("tweet_id", "size"),
        num_customer_messages=("inbound", "sum"),
    )
    thread_size["num_brand_messages"] = thread_size["num_messages"] - thread_size["num_customer_messages"]

    root_rows = df.loc[df["tweet_id"] == df["thread_root_id"]].set_index("thread_root_id")

    brand_messages = df.loc[~df["inbound"]].sort_values("created_at")
    first_brand_response_at = brand_messages.groupby("thread_root_id")["created_at"].first()
    first_brand_id = brand_messages.groupby("thread_root_id")["author_id"].first()

    tickets = root_rows.join(thread_size)
    tickets = tickets.join(first_brand_response_at.rename("first_brand_response_at"))
    tickets = tickets.join(first_brand_id.rename("brand_id"))
    tickets = tickets[tickets["inbound"]]
    tickets = tickets.drop(columns=["text"], errors="ignore")

    tickets["first_response_seconds"] = (
        tickets["first_brand_response_at"] - tickets["created_at"]
    ).dt.total_seconds()
    tickets["resolved"] = tickets["num_brand_messages"] > 0
    tickets["text_length"] = tickets["text_clean"].str.len()
    tickets["word_count"] = tickets["text_clean"].str.split().str.len()

    tickets = tickets.reset_index().rename(
        columns={
            "thread_root_id": "ticket_id",
            "author_id": "customer_id",
            "created_at": "opened_at",
            "text_clean": "text",
        }
    )
    tickets = tickets[FEATURE_COLUMNS]

    response_times = tickets["first_response_seconds"].dropna()
    report = {
        "total_tweets": len(df),
        "total_threads": int(df["thread_root_id"].nunique()),
        "tickets_extracted": len(tickets),
        "tickets_with_brand_response": int(tickets["resolved"].sum()),
        "median_first_response_seconds": float(response_times.median()) if len(response_times) else None,
        "max_first_response_seconds": float(response_times.max()) if len(response_times) else None,
    }
    return tickets, report


def run(
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    df = pd.read_parquet(processed_path)
    tickets, report = build_ticket_features(df)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    tickets.to_parquet(features_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    report = run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
