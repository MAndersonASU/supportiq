"""
Ingestion stage. Streams the raw Twitter customer-support CSV in chunks,
validates each record against the RawTweet schema, and writes
schema-conformant records to the landing zone as Parquet. Records that
fail schema validation are counted and skipped rather than aborting the
run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from src.data.schema import RawTweet

DEFAULT_RAW_PATH = Path("data/raw/twcs.csv")
DEFAULT_LANDING_PATH = Path("data/landing/tweets.parquet")
CHUNK_SIZE = 50_000

LANDING_SCHEMA = pa.schema(
    [
        ("tweet_id", pa.int64()),
        ("author_id", pa.string()),
        ("inbound", pa.bool_()),
        ("created_at", pa.timestamp("s", tz="UTC")),
        ("text", pa.string()),
        ("response_tweet_id", pa.list_(pa.int64())),
        ("in_response_to_tweet_id", pa.int64()),
    ]
)


def parse_chunk(chunk: pd.DataFrame) -> tuple[list[dict], int]:
    valid_records = []
    rejected = 0
    for row in chunk.to_dict("records"):
        try:
            record = RawTweet.model_validate(row)
        except ValidationError:
            rejected += 1
            continue
        valid_records.append(record.model_dump())
    return valid_records, rejected


def ingest(
    raw_path: Path = DEFAULT_RAW_PATH,
    landing_path: Path = DEFAULT_LANDING_PATH,
) -> dict[str, int]:
    landing_path.parent.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv(raw_path, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE)

    writer: pq.ParquetWriter | None = None
    rows_read = 0
    rows_rejected = 0

    try:
        for chunk in reader:
            valid_records, rejected = parse_chunk(chunk)
            rows_read += len(chunk)
            rows_rejected += rejected

            if valid_records:
                table = pa.Table.from_pylist(valid_records, schema=LANDING_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(landing_path, LANDING_SCHEMA)
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return {
        "rows_read": rows_read,
        "rows_written": rows_read - rows_rejected,
        "rows_rejected": rows_rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw tweets into the landing zone.")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--landing-path", type=Path, default=DEFAULT_LANDING_PATH)
    args = parser.parse_args()

    summary = ingest(args.raw_path, args.landing_path)
    print(
        f"rows_read={summary['rows_read']} "
        f"rows_written={summary['rows_written']} "
        f"rows_rejected={summary['rows_rejected']}"
    )


if __name__ == "__main__":
    main()
