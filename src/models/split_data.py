"""
Stratified train/val/test split for the labeled ticket dataset.
Stratifies on category, the more imbalanced of the two prediction
targets, so every split preserves class proportions down to the rarest
category. The same split is reused for both the category and priority
models so a given ticket is never in train for one target and test for
the other.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
RANDOM_STATE = 42


def stratified_split(
    df: pd.DataFrame,
    stratify_col: str = "category",
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df[stratify_col],
        random_state=random_state,
    )
    val_share_of_temp = val_frac / (1 - train_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share_of_temp,
        stratify=temp_df[stratify_col],
        random_state=random_state,
    )
    return train_df, val_df, test_df
