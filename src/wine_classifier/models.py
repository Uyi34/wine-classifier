"""Model definitions, cross-validated comparison and test-set evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5


def build_models() -> dict[str, Any]:
    """Return the candidate models, keyed by name.

    Logistic regression is sensitive to feature scale, so it is wrapped in a
    pipeline that standardises the features. The random forest does not need
    scaling.
    """
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }


def split_data(X: pd.DataFrame, y: pd.Series):
    """Split into stratified train and test sets."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Score each candidate with stratified k-fold cross-validation.

    Only the training data is used here, so the test set stays untouched.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in build_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        rows.append(
            {
                "model": name,
                "cv_mean_accuracy": scores.mean(),
                "cv_std": scores.std(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("cv_mean_accuracy", ascending=False)
        .reset_index(drop=True)
    )


def fit_model(name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Build the named model and fit it on the training data."""
    model = build_models()[name]
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute test-set metrics. Returns plain Python types so they can be saved as JSON."""
    y_pred = model.predict(X_test)
    labels = np.unique(y_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "labels": [int(label) for label in labels],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }
