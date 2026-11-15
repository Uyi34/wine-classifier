from wine_classifier.data import (
    TARGET_COLUMN,
    TARGET_NAME_COLUMN,
    class_balance,
    feature_columns,
    load_wine_data,
    summarise,
)


def test_dataset_shape():
    df = load_wine_data()
    assert df.shape == (178, 15)  # 13 features + target + target_name


def test_no_missing_values():
    df = load_wine_data()
    assert not df.isna().any().any()


def test_three_classes_with_expected_names():
    df = load_wine_data()
    assert set(df[TARGET_COLUMN].unique()) == {0, 1, 2}
    assert set(df[TARGET_NAME_COLUMN].unique()) == {"class_0", "class_1", "class_2"}


def test_class_balance_matches_known_counts():
    counts = class_balance(load_wine_data()).sort_index()
    assert list(counts) == [59, 71, 48]


def test_feature_columns_exclude_targets():
    df = load_wine_data()
    features = feature_columns(df)
    assert len(features) == 13
    assert TARGET_COLUMN not in features
    assert TARGET_NAME_COLUMN not in features


def test_summary_has_one_row_per_feature():
    df = load_wine_data()
    stats = summarise(df)
    assert len(stats) == 13
    assert list(stats.columns) == ["mean", "std", "min", "max"]
