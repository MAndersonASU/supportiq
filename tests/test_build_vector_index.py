"""
Tests for the per-category sampling used to build the vector index: caps
oversized categories at the limit, keeps every row of undersized
categories, and never drops a category entirely.
"""

import pandas as pd

from src.ai.build_vector_index import sample_knowledge_base


def make_df(counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    ticket_id = 0
    for category, count in counts.items():
        for _ in range(count):
            rows.append({"ticket_id": ticket_id, "category": category, "customer_message": "x"})
            ticket_id += 1
    return pd.DataFrame(rows)


def test_oversized_category_is_capped():
    df = make_df({"A": 1000, "B": 50})

    sampled = sample_knowledge_base(df, max_per_category=100)

    assert (sampled["category"] == "A").sum() == 100


def test_undersized_category_keeps_all_rows():
    df = make_df({"A": 1000, "B": 50})

    sampled = sample_knowledge_base(df, max_per_category=100)

    assert (sampled["category"] == "B").sum() == 50


def test_every_category_present_in_sample():
    df = make_df({"A": 1000, "B": 50, "C": 5})

    sampled = sample_knowledge_base(df, max_per_category=100)

    assert set(sampled["category"]) == {"A", "B", "C"}
