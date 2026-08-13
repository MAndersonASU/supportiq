"""
Builds the resolution knowledge base the RAG assistant retrieves from:
each labeled ticket's opening message paired with the first brand reply
in its thread. ticket_features.parquet carries reply counts and timing
but not reply text, so thread roots are recomputed here against the full
cleaned tweet dataset to recover the actual resolution text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.build_ticket_features import assign_thread_roots

DEFAULT_TWEETS_PATH = Path("data/processed/tweets.parquet")
DEFAULT_LABELED_PATH = Path("data/processed/labeled_tickets.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/knowledge_base.parquet")
DEFAULT_REPORT_PATH = Path("data/processed/knowledge_base_report.json")


def build_knowledge_base(
    tweets_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    tweets_df = tweets_df.copy()
    tweets_df["thread_root_id"] = assign_thread_roots(tweets_df)

    brand_replies = tweets_df.loc[~tweets_df["inbound"]].sort_values("created_at")
    first_reply_text = brand_replies.groupby("thread_root_id")["text_clean"].first()

    kb = labeled_df[["ticket_id", "category", "priority", "text"]].copy()
    kb = kb.rename(columns={"text": "customer_message"})
    kb["resolution_text"] = kb["ticket_id"].map(first_reply_text)

    missing = int(kb["resolution_text"].isna().sum())
    kb = kb.dropna(subset=["resolution_text"]).reset_index(drop=True)

    report = {
        "tickets_in": len(labeled_df),
        "resolutions_found": len(kb),
        "resolutions_missing": missing,
    }
    return kb, report


def run(
    tweets_path: Path = DEFAULT_TWEETS_PATH,
    labeled_path: Path = DEFAULT_LABELED_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    tweets_df = pd.read_parquet(tweets_path)
    labeled_df = pd.read_parquet(labeled_path)
    kb, report = build_knowledge_base(tweets_df, labeled_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kb.to_parquet(output_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
