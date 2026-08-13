"""
Cleaning stage. Normalizes text for downstream NLP/embedding use without
discarding the original, and enforces the dataset's missing-value policy:
only the two response-linkage fields are allowed to be null. Reads the
validated dataset and writes the final processed dataset.

Note on deduplication: validation reports repeated (author_id, text)
pairs as a quality metric, but each has a distinct, unique tweet_id and
represents a separate real interaction (e.g. a support account sending
the same templated reply to different customers). These are not
duplicate records and are not removed here.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

DEFAULT_VALIDATED_PATH = Path("data/validated/tweets.parquet")
DEFAULT_PROCESSED_PATH = Path("data/processed/tweets.parquet")
DEFAULT_REPORT_PATH = Path("data/processed/cleaning_report.json")

NULLABLE_COLUMNS = {"response_tweet_id", "in_response_to_tweet_id"}
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required_columns = [c for c in df.columns if c not in NULLABLE_COLUMNS]
    unexpected_nulls = int(df[required_columns].isna().sum().sum())

    df = df.copy()
    df["text_clean"] = df["text"].map(normalize_text)

    empty_after_cleaning = df["text_clean"] == ""
    dropped = int(empty_after_cleaning.sum())
    cleaned_df = df.loc[~empty_after_cleaning].reset_index(drop=True)

    report = {
        "rows_in": len(df),
        "rows_out": len(cleaned_df),
        "rows_dropped_empty_after_cleaning": dropped,
        "unexpected_nulls_in_required_columns": unexpected_nulls,
    }
    return cleaned_df, report


def clean(
    validated_path: Path = DEFAULT_VALIDATED_PATH,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    df = pd.read_parquet(validated_path)
    cleaned_df, report = clean_dataset(df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(processed_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    report = clean()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
