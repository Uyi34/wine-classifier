from wine_classifier.data import TARGET_COLUMN, feature_columns, load_wine_data
from wine_classifier.models import (
    build_models,
    compare_models,
    evaluate_model,
    fit_model,
    split_data,
)


def _split():
    df = load_wine_data()
    X = df[feature_columns(df)]
    y = df[TARGET_COLUMN]
    return split_data(X, y)


def test_split_is_stratified_and_sized_correctly():
    X_train, X_test, y_train, y_test = _split()
    assert len(X_train) + len(X_test) == 178
    assert abs(len(X_test) / 178 - 0.25) < 0.02
    # Every class appears in both splits
    assert set(y_train.unique()) == set(y_test.unique()) == {0, 1, 2}


def test_build_models_returns_expected_candidates():
    assert set(build_models()) == {"logistic_regression", "random_forest"}


def test_comparison_is_sorted_best_first():
    X_train, _, y_train, _ = _split()
    comparison = compare_models(X_train, y_train)
    scores = list(comparison["cv_mean_accuracy"])
    assert scores == sorted(scores, reverse=True)
    assert len(comparison) == 2


def test_best_model_meets_accuracy_threshold_on_test_set():
    X_train, X_test, y_train, y_test = _split()
    model = fit_model("logistic_regression", X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] >= 0.90
    assert len(metrics["confusion_matrix"]) == 3
