"""Loading and summarising the UCI Wine dataset.

The dataset is bundled with scikit-learn, so no download is required.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_wine

TARGET_COLUMN = "target"
TARGET_NAME_COLUMN = "target_name"


def load_wine_data() -> pd.DataFrame:
    """Return the wine dataset as a DataFrame with a readable class label column."""
    bunch = load_wine(as_frame=True)
    df = bunch.frame.copy()
    class_names = dict(enumerate(bunch.target_names))
    df[TARGET_NAME_COLUMN] = df[TARGET_COLUMN].map(class_names)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of the input features, excluding the target columns."""
    excluded = {TARGET_COLUMN, TARGET_NAME_COLUMN}
    return [column for column in df.columns if column not in excluded]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-feature summary statistics."""
    stats = df[feature_columns(df)].describe().T
    return stats[["mean", "std", "min", "max"]]


def class_balance(df: pd.DataFrame) -> pd.Series:
    """Return the number of samples in each class."""
    return df[TARGET_NAME_COLUMN].value_counts()
