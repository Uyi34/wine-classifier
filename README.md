# Wine Classifier

A small, end-to-end machine learning project in Python. It loads the classic
**UCI Wine recognition dataset**, explores it, compares two classifiers with
cross-validation, evaluates the best one on a held-out test set, and produces
reports and plots.

The dataset contains 178 samples of Italian wines from three cultivars,
described by 13 chemical measurements (alcohol, malic acid, flavanoids,
colour intensity, proline, and so on). The task is to predict which cultivar
a wine comes from.

## Project structure

```
wine-classifier/
├── src/wine_classifier/
│   ├── __init__.py
│   ├── __main__.py      # enables `python -m wine_classifier`
│   ├── data.py          # loading and summarising the dataset
│   ├── models.py        # model definitions, comparison and evaluation
│   ├── plots.py         # confusion matrix and feature importance charts
│   └── main.py          # end-to-end pipeline
├── tests/
│   ├── test_data.py
│   └── test_models.py
├── reports/             # generated metrics and plots (git-ignored)
├── pytest.ini
├── requirements.txt
└── README.md
```

## Quick start

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/wine-classifier.git
cd wine-classifier

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m wine_classifier        # run the full pipeline
pytest                           # run the test suite
```

The dataset ships with scikit-learn, so there is nothing to download.

## What the pipeline does

1. **Load and summarise** the data, printing per-feature statistics.
2. **Split** into stratified train and test sets (75% / 25%).
3. **Compare** a logistic regression (with feature scaling) and a random
   forest using 5-fold stratified cross-validation on the training set.
4. **Fit** the better model on the full training set.
5. **Evaluate** on the untouched test set, reporting accuracy, macro F1 and
   the confusion matrix.
6. **Explain** the model with permutation feature importance.
7. **Save** `reports/metrics.json`, `reports/confusion_matrix.png` and
   `reports/feature_importance.png`.

## Example results

Results from a run with scikit-learn 1.9 and a fixed random seed:

| Model               | CV accuracy (5-fold, train) | Test accuracy | Test macro F1 |
|---------------------|-----------------------------|---------------|---------------|
| Random forest       | 0.985 ± 0.030               | 1.00          | 1.00          |
| Logistic regression | 0.978 ± 0.018               | 1.00          | 1.00          |

The random forest is selected on cross-validation accuracy. The test set has
only 45 samples, so a perfect score is encouraging but does not prove the
models are equally good. Cross-validation gives a more reliable comparison.
Both models are evaluated on wines they were not trained on.

## Design choices

- **Pipelines.** Scaling is part of the model pipeline and is fitted only on
  training data, which prevents data leakage into the test set.
- **Stratified splitting.** Class proportions are preserved in every split.
- **Fixed random seeds.** Every run gives the same results.
- **Cross-validation for model selection.** The test set is used once, at the
  end, so it is not used to choose between models.

## Running the tests

```bash
pytest -v
```

The tests check the dataset shape and class balance, the stratified split,
and that the trained model meets a minimum accuracy threshold.

## Possible extensions

- Add hyperparameter tuning with `GridSearchCV`.
- Try gradient boosting (`HistGradientBoostingClassifier`).
- Replace the bundled dataset with a CSV file from the UCI repository.
- Wrap the pipeline in a command-line interface with `argparse`.

## Licence

This project is released under the MIT Licence. See [LICENSE](LICENSE).
