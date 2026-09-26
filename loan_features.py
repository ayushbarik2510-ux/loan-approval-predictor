"""Dataset columns and stateless cleaning shared by training and inference."""

import numpy as np
import pandas as pd

FEATURES = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
]
CATEGORICAL_FEATURES = ["education", "self_employed"]
NUMERIC_FEATURES = [name for name in FEATURES if name not in CATEGORICAL_FEATURES]
TARGET = "loan_status"
TARGET_MAPPING = {"Rejected": 0, "Approved": 1}


def clean_features(frame):
    """Clean values without learning statistics; imputation happens later.

    Keeping this function in an importable module makes joblib loading work in
    a fresh Python process. Ship this file with the saved model.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Provide applicant data as a pandas DataFrame.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    result = frame.loc[:, FEATURES].copy()
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(result[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        values = values.mask(values < 0)
        if column in {"income_annum", "loan_amount", "loan_term"}:
            values = values.mask(values == 0)
        if column == "cibil_score":
            values = values.mask(~values.between(300, 900))
        if column in {"no_of_dependents", "loan_term", "cibil_score"}:
            values = values.mask(values.notna() & (values % 1 != 0))
        result[column] = values.astype(float)
    for column in CATEGORICAL_FEATURES:
        values = result[column].astype("string").str.strip()
        values = values.mask(values.eq(""))
        result[column] = values.astype(object).where(values.notna(), np.nan)
    return result
