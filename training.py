"""Train, compare, evaluate and save two loan-approval classifiers.

Run: python training.py
The Colab notebook calls these same functions one stage at a time.
"""

import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import time
import urllib.request
import zipfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay, make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from loan_features import (
    FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET, TARGET_MAPPING, clean_features,
)

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SOURCE_URL = "https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset"
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "architsharma01/loan-approval-prediction-dataset?datasetVersionNumber=1"
)
DATA_NOTE = (
    "Educational public Kaggle dataset. Authentic Indian customer/bureau "
    "provenance is unverified. Monetary and loan-term units are not documented "
    "in the publisher description; use the dataset's original units."
)


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def download_dataset(csv_path):
    """Use a local CSV if present; otherwise download the public source."""
    csv_path = Path(csv_path)
    if csv_path.is_file():
        return csv_path
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "LoanApprovalProject/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read(10 * 1024 * 1024)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            contents = bundle.read("loan_approval_dataset.csv")
    except Exception as exc:
        raise RuntimeError(
            f"Download failed. Download the CSV from {SOURCE_URL} and place it at {csv_path}."
        ) from exc
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(contents)
    return csv_path


def load_and_inspect(csv_path):
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    if not data.columns.is_unique:
        raise ValueError("Duplicate column names after removing whitespace.")
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    unexpected = sorted(set(data.columns) - set(required + ["loan_id"]))
    if missing or unexpected:
        raise ValueError(f"Dataset schema differs. Missing={missing}; unexpected={unexpected}")
    for column in CATEGORICAL_FEATURES + [TARGET]:
        data[column] = data[column].astype("string").str.strip()
    if data[TARGET].isna().any() or not data[TARGET].isin(TARGET_MAPPING).all():
        raise ValueError("Every training row must have an Approved or Rejected label.")

    # This cleaning is only for inspection and duplicate detection. It learns
    # nothing. The same function runs inside every fitted model pipeline.
    clean = clean_features(data[FEATURES])
    audit = {
        "original_rows": len(data),
        "negative_residential_asset_rows": int((pd.to_numeric(
            data["residential_assets_value"], errors="coerce") < 0).sum()),
        "missing_values_after_stateless_cleaning": clean.isna().sum().astype(int).to_dict(),
    }
    check = clean.copy()
    check[TARGET] = data[TARGET]
    conflicts = check.groupby(FEATURES, dropna=False, observed=True)[TARGET].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Identical applicant profiles have conflicting labels; review before splitting.")
    duplicate = clean.duplicated(keep="first")
    audit["duplicate_profiles_removed"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    audit["usable_rows"] = len(data)
    audit["target_counts"] = data[TARGET].value_counts().astype(int).to_dict()
    X = data.loc[:, FEATURES].copy()  # loan_id and loan_status never enter X.
    y = data[TARGET].map(TARGET_MAPPING).astype(int)
    if y.value_counts().min() < 10:
        raise ValueError("Too few examples per class for a holdout and five-fold CV.")
    print("Dataset shape:", data.shape)
    print(data.head().to_string(index=False))
    print("\nColumn types:")
    print(data.dtypes.to_string())
    print("\nTarget counts:", audit["target_counts"])
    print("Missing/invalid feature values:", audit["missing_values_after_stateless_cleaning"])
    print("Negative asset values treated as missing:", audit["negative_residential_asset_rows"])
    print(DATA_NOTE)
    return data, X, y, audit


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    assert set(X_train.index).isdisjoint(X_test.index)
    print(f"Training rows: {len(X_train)}; held-out test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_pipeline(classifier):
    numerical = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numerical, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")
    return Pipeline([
        ("clean", FunctionTransformer(clean_features, validate=False, feature_names_out="one-to-one")),
        ("preprocess", preprocess),
        ("model", classifier),
    ])


def tune_models(X_train, y_train, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "Logistic Regression": (
            LogisticRegression(solver="lbfgs", l1_ratio=0.0, max_iter=3000, random_state=SEED),
            {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
             "model__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"model__max_depth": [3, 5, 8, 12],
             "model__min_samples_leaf": [5, 15, 30],
             "model__ccp_alpha": [0.0, 0.001],
             "model__class_weight": [None, "balanced"]},
        ),
    }
    scoring = {
        "roc_auc": "roc_auc", "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    searches = {}
    for name, (model, grid) in candidates.items():
        started = time.perf_counter()
        search = GridSearchCV(
            build_pipeline(model), param_grid=grid, scoring=scoring,
            refit="roc_auc", cv=cv, n_jobs=1, error_score="raise",
            return_train_score=True,
        )
        # All imputation, encoding and scaling are fitted inside each CV fold.
        search.fit(X_train, y_train)
        searches[name] = search
        slug = name.lower().replace(" ", "_")
        pd.DataFrame(search.cv_results_).to_csv(report_dir / f"{slug}_cv_results.csv", index=False)
        print(f"\n{name}: CV ROC-AUC={search.best_score_:.4f}; elapsed={time.perf_counter()-started:.1f}s")
        print("Best parameters:", search.best_params_)
    # Select before looking at the test set. Ties favor the earlier model (LR).
    winner = max(searches, key=lambda name: searches[name].best_score_)
    print("\nSelected using training-set CV only:", winner)
    return searches, winner


def approval_scores(pipeline, X):
    positive_column = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(X)[:, positive_column]


def evaluate_models(searches, X_train, y_train, X_test, y_test, report_dir):
    """One final comparison on the untouched test set; never retune from it."""
    report_dir = Path(report_dir)
    rows, reports, predictions = [], {}, pd.DataFrame({"actual": y_test})
    cm_figure, cm_axes = plt.subplots(1, 2, figsize=(10, 4))
    roc_figure, roc_axis = plt.subplots(figsize=(6, 5))
    for position, (name, search) in enumerate(searches.items()):
        fitted = search.best_estimator_
        predicted = fitted.predict(X_test)
        scores = approval_scores(fitted, X_test)
        best = search.best_index_
        rows.append({
            "model": name,
            "cv_roc_auc_mean": float(search.best_score_),
            "cv_roc_auc_std": float(search.cv_results_["std_test_roc_auc"][best]),
            "train_accuracy": float(accuracy_score(y_train, fitted.predict(X_train))),
            "test_accuracy": float(accuracy_score(y_test, predicted)),
            "test_precision": float(precision_score(y_test, predicted, zero_division=0)),
            "test_recall": float(recall_score(y_test, predicted, zero_division=0)),
            "test_f1": float(f1_score(y_test, predicted, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, scores)),
        })
        reports[name] = {
            "confusion_matrix_labels": ["Rejected", "Approved"],
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y_test, predicted, labels=[0, 1], target_names=["Rejected", "Approved"],
                zero_division=0, output_dict=True,
            ),
        }
        predictions[f"{name}_prediction"] = predicted
        predictions[f"{name}_approval_score"] = scores
        ConfusionMatrixDisplay.from_predictions(
            y_test, predicted, labels=[0, 1], display_labels=["Rejected", "Approved"],
            ax=cm_axes[position], colorbar=False, cmap="Blues",
        )
        cm_axes[position].set_title(name)
        RocCurveDisplay.from_predictions(y_test, scores, name=name, ax=roc_axis)
    roc_axis.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    roc_axis.set_title("Held-out test ROC curves")
    roc_axis.legend(loc="lower right")
    cm_figure.tight_layout()
    roc_figure.tight_layout()
    cm_figure.savefig(report_dir / "confusion_matrices.png", dpi=160)
    roc_figure.savefig(report_dir / "roc_curves.png", dpi=160)
    plt.close(cm_figure)
    plt.close(roc_figure)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    predictions.to_csv(report_dir / "test_predictions.csv", index_label="source_row_index")
    write_json(report_dir / "classification_reports.json", reports)
    print("\nPositive class for precision/recall/F1: Approved (1)")
    print(comparison.round(4).to_string(index=False))
    for name, result in reports.items():
        print(f"\n{name} confusion matrix (rows=actual, columns=predicted; Rejected, Approved):")
        print(np.array(result["confusion_matrix"]))
    return comparison


def save_model(searches, winner, comparison, X_train, X_test, y_train, y_test,
               audit, csv_path, project_dir):
    project_dir = Path(project_dir)
    artifact_dir = project_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pipeline = searches[winner].best_estimator_
    # Save the exact estimator evaluated on the holdout. Do not fit on X_test.
    model_path = artifact_dir / "loan_pipeline.joblib"
    temporary = artifact_dir / "loan_pipeline.joblib.tmp"
    joblib.dump(pipeline, temporary, compress=3)
    temporary.replace(model_path)
    clean_train = clean_features(X_train)
    numeric_ranges = {}
    for name in NUMERIC_FEATURES:
        values = clean_train[name].dropna()
        numeric_ranges[name] = None if values.empty else {"min": float(values.min()), "max": float(values.max())}
    metadata = {
        "model_name": winner, "feature_names": FEATURES,
        "target_mapping": TARGET_MAPPING, "positive_class": "Approved",
        "selection_metric": "five-fold training-set cross-validation ROC-AUC",
        "best_parameters": searches[winner].best_params_,
        "cv_roc_auc": float(searches[winner].best_score_),
        "held_out_metrics": comparison.set_index("model").loc[winner].to_dict(),
        "random_seed": SEED, "test_fraction": TEST_SIZE,
        "training_rows": len(X_train), "test_rows": len(X_test),
        "probabilities_calibrated": False,
        "training_numeric_ranges": numeric_ranges,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "scipy", "pandas", "scikit-learn", "joblib"]},
        "python_version": platform.python_version(),
        "dataset_source": SOURCE_URL, "dataset_note": DATA_NOTE,
        "dataset_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "audit": audit,
    }
    write_json(artifact_dir / "model_metadata.json", metadata)
    write_json(project_dir / "reports" / "split_indices.json", {
        "train": X_train.index.astype(int).tolist(), "test": X_test.index.astype(int).tolist(),
    })
    write_json(project_dir / "data" / "source.json", {
        "url": SOURCE_URL, "download_url": DOWNLOAD_URL,
        "sha256": metadata["dataset_sha256"], "note": DATA_NOTE,
        "license_as_listed_by_uploader": "MIT",
    })
    restored = joblib.load(model_path)
    np.testing.assert_allclose(
        approval_scores(restored, X_test), approval_scores(pipeline, X_test), rtol=0, atol=1e-12,
    )
    print("\nSaved and reload-checked:", model_path)
    return model_path, metadata


def unseen_prediction_test(model_path, report_dir):
    # Illustrative new inputs, not measured customers. No outcome is known.
    examples = pd.DataFrame([
        {"no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
         "income_annum": 2400000, "loan_amount": 6000000, "loan_term": 10,
         "cibil_score": 780, "residential_assets_value": 5000000,
         "commercial_assets_value": 0, "luxury_assets_value": 3000000,
         "bank_asset_value": 2000000},
        {"no_of_dependents": 1, "education": "Not Graduate", "self_employed": "Yes",
         "income_annum": 1800000, "loan_amount": 6000000, "loan_term": 12,
         "cibil_score": 420, "residential_assets_value": 2000000,
         "commercial_assets_value": 1000000, "luxury_assets_value": 2000000,
         "bank_asset_value": 500000},
        {"no_of_dependents": None, "education": None, "self_employed": "No",
         "income_annum": 3500000, "loan_amount": 8000000, "loan_term": 8,
         "cibil_score": 720, "residential_assets_value": None,
         "commercial_assets_value": 0, "luxury_assets_value": 4000000,
         "bank_asset_value": None},
    ], columns=FEATURES)
    restored = joblib.load(model_path)
    result = examples.copy()
    result["prediction"] = ["Approved" if value == 1 else "Rejected" for value in restored.predict(examples)]
    result["approval_score"] = approval_scores(restored, examples)
    result.to_csv(Path(report_dir) / "unseen_predictions.csv", index=False)
    print("\nIllustrative unseen predictions (uncalibrated model scores):")
    print(result[["cibil_score", "prediction", "approval_score"]].to_string(index=False))
    return result


def main():
    project_dir = Path(os.environ.get("LOAN_PROJECT_DIR", Path(__file__).resolve().parent))
    csv_path = download_dataset(project_dir / "data" / "loan_approval_dataset.csv")
    data, X, y, audit = load_and_inspect(csv_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    report_dir = project_dir / "reports"
    searches, winner = tune_models(X_train, y_train, report_dir)
    comparison = evaluate_models(searches, X_train, y_train, X_test, y_test, report_dir)
    model_path, metadata = save_model(
        searches, winner, comparison, X_train, X_test, y_train, y_test,
        audit, csv_path, project_dir,
    )
    unseen_prediction_test(model_path, report_dir)


if __name__ == "__main__":
    main()
