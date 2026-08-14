"""
Tests for the ingestion stage: valid records are written to the landing
zone in the expected shape, and malformed records are rejected without
aborting the run.
"""

from pathlib import Path

import pandas as pd

from src.data.ingest import ingest

CSV_CONTENT = """tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,sprintcare,False,Tue Oct 31 22:10:47 +0000 2017,hello there,2,
2,115712,True,Tue Oct 31 22:11:45 +0000 2017,thanks,"3,4",1
3,115712,True,not-a-real-date,broken row,,1
"""


def test_ingest_writes_valid_rows_and_rejects_malformed(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    landing_path = tmp_path / "landing.parquet"
    raw_path.write_text(CSV_CONTENT)

    summary = ingest(raw_path, landing_path)

    assert summary == {"rows_read": 3, "rows_written": 2, "rows_rejected": 1}

    written = pd.read_parquet(landing_path)
    assert set(written["tweet_id"]) == {1, 2}

    row_one = written.loc[written["tweet_id"] == 1].iloc[0]
    assert not row_one["inbound"]
    assert pd.isna(row_one["in_response_to_tweet_id"])

    row_two = written.loc[written["tweet_id"] == 2].iloc[0]
    assert list(row_two["response_tweet_id"]) == [3, 4]
