"""
Validation stage. Applies business-rule checks on top of the structural
schema already enforced during ingestion: uniqueness, value ranges, and
cross-field consistency. Rows that fail a hard check are quarantined;
rows that pass are written to the processed dataset. Referential and
duplication issues that don't warrant dropping a row are recorded as
quality metrics in the validation report instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

DEFAULT_LANDING_PATH = Path("data/landing/tweets.parquet")
DEFAULT_PROCESSED_PATH = Path("data/processed/tweets.parquet")
DEFAULT_REPORT_PATH = Path("data/processed/validation_report.json")

MIN_VALID_DATE = pd.Timestamp("2006-03-21", tz="UTC")
MAX_VALID_DATE = pd.Timestamp("2020-01-01", tz="UTC")

SCHEMA = pa.DataFrameSchema(
    {
        "tweet_id": pa.Column(int, unique=True, checks=pa.Check.ge(0)),
        "author_id": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "inbound": pa.Column(bool),
        "created_at": pa.Column(
            checks=[
                pa.Check(lambda s: s >= MIN_VALID_DATE),
                pa.Check(lambda s: s <= MAX_VALID_DATE),
            ]
        ),
        "text": pa.Column(str, checks=pa.Check.str_length(min_value=1)),
        "response_tweet_id": pa.Column(object, nullable=True),
        "in_response_to_tweet_id": pa.Column(float, nullable=True),
    },
    strict=True,
)


def check_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    try:
        SCHEMA.validate(df, lazy=True)
        failing_indices: set[int] = set()
    except pa.errors.SchemaErrors as exc:
        failing_indices = set(exc.failure_cases["index"].dropna().astype(int))

    valid_df = df.drop(index=failing_indices)

    known_ids = set(valid_df["tweet_id"])
    responses = valid_df["in_response_to_tweet_id"].dropna()
    broken_links = int((~responses.isin(known_ids)).sum())
    broken_link_rate = broken_links / len(responses) if len(responses) else 0.0

    duplicate_pairs = int(valid_df.duplicated(subset=["author_id", "text"]).sum())

    report = {
        "rows_in": len(df),
        "rows_valid": len(valid_df),
        "rows_rejected": len(failing_indices),
        "broken_response_links": broken_links,
        "broken_response_link_rate": round(broken_link_rate, 4),
        "duplicate_author_text_pairs": duplicate_pairs,
    }
    return valid_df, report


def validate(
    landing_path: Path = DEFAULT_LANDING_PATH,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict:
    df = pd.read_parquet(landing_path)
    valid_df, report = check_dataset(df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_parquet(processed_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    return report


def main() -> None:
    report = validate()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
