"""End-to-end pipeline: load, compare, evaluate, explain and save results."""

from __future__ import annotations

import json
from pathlib import Path

from wine_classifier.data import (
    TARGET_COLUMN,
    TARGET_NAME_COLUMN,
    class_balance,
    feature_columns,
    load_wine_data,
    summarise,
)
from wine_classifier.models import (
    compare_models,
    evaluate_model,
    fit_model,
    split_data,
)
from wine_classifier.plots import (
    permutation_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def main(reports_dir: Path = REPORTS_DIR) -> dict:
    """Run the full pipeline and return the metrics it saved."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and explore
    df = load_wine_data()
    print("\n=== Class balance ===")
    print(class_balance(df).to_string())
    print("\n=== Feature summary ===")
    print(summarise(df).round(2).to_string())

    # 2. Split
    features = feature_columns(df)
    X, y = df[features], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 3. Compare models with cross-validation on the training set
    comparison = compare_models(X_train, y_train)
    print("\n=== Cross-validated comparison (training data only) ===")
    print(comparison.round(4).to_string(index=False))

    # 4. Refit the best model and evaluate once on the test set
    best_name = comparison.iloc[0]["model"]
    model = fit_model(best_name, X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"\nBest model: {best_name}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")

    # 5. Explain the model
    importances = permutation_feature_importance(model, X_test, y_test)

    # 6. Save outputs
    class_names = list(df.groupby(TARGET_COLUMN)[TARGET_NAME_COLUMN].first())
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        reports_dir / "confusion_matrix.png",
    )
    plot_feature_importance(importances, reports_dir / "feature_importance.png")

    summary = {
        "dataset": "UCI Wine (bundled with scikit-learn)",
        "n_samples": int(len(df)),
        "n_features": len(features),
        "best_model": best_name,
        "comparison": comparison.round(6).to_dict(orient="records"),
        "test_metrics": metrics,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results to {reports_dir.resolve()}")
    return summary


if __name__ == "__main__":
    main()
