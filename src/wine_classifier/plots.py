"""Plots for the classification results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402


def plot_confusion_matrix(
    matrix: list[list[int]], class_names: list[str], path: Path
) -> None:
    """Save a labelled confusion matrix heatmap."""
    data = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(data, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted cultivar")
    ax.set_ylabel("True cultivar")
    ax.set_title("Confusion matrix (test set)")

    threshold = data.max() / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                str(data[i, j]),
                ha="center",
                va="center",
                color="white" if data[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def permutation_feature_importance(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, seed: int = 42
) -> pd.Series:
    """Measure how much test accuracy drops when each feature is shuffled."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=seed, scoring="accuracy"
    )
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(
        ascending=False
    )


def plot_feature_importance(importances: pd.Series, path: Path, top_n: int = 10) -> None:
    """Save a horizontal bar chart of the most important features."""
    top = importances.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("Drop in test accuracy when shuffled")
    ax.set_title(f"Top {len(top)} features (permutation importance)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
