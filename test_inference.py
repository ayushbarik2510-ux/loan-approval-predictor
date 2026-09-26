"""Run after training: python -m unittest discover -s tests -v"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pandas as pd

from api import app
from loan_features import FEATURES, NUMERIC_FEATURES, clean_features

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = {
    "no_of_dependents": 2, "education": "Graduate", "self_employed": "No",
    "income_annum": 2400000, "loan_amount": 6000000, "loan_term": 10,
    "cibil_score": 780, "residential_assets_value": 5000000,
    "commercial_assets_value": 0, "luxury_assets_value": 3000000,
    "bank_asset_value": 2000000,
}


class InferenceTests(unittest.TestCase):
    def test_api_matches_saved_pipeline_without_fitting(self):
        pipeline = joblib.load(ROOT / "artifacts" / "loan_pipeline.joblib")
        samples = [SAMPLE, {**SAMPLE, "cibil_score": 420}, {
            **SAMPLE, "education": None, "no_of_dependents": None,
            "residential_assets_value": None, "bank_asset_value": None,
        }]
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            with patch.object(app.state.pipeline, "fit", side_effect=AssertionError("Inference must not fit")):
                for sample in samples:
                    with self.subTest(sample=sample):
                        # Deliberately reverse JSON keys to verify feature ordering.
                        response = client.post("/predict", json=dict(reversed(list(sample.items()))))
                        self.assertEqual(response.status_code, 200, response.text)
                        result = response.json()
                        frame = pd.DataFrame([sample], columns=FEATURES)
                        self.assertEqual(result["predicted_class"], int(pipeline.predict(frame)[0]))
                        self.assertAlmostEqual(result["approval_score"], pipeline.predict_proba(frame)[0, 1])
                        self.assertAlmostEqual(result["approval_score"] + result["rejection_score"], 1)
                        self.assertEqual(set(result["imputed_fields"]), {k for k, v in sample.items() if v is None})

    def test_invalid_applicants_return_422(self):
        invalid = [
            {**SAMPLE, "cibil_score": 299}, {**SAMPLE, "cibil_score": 901},
            {**SAMPLE, "cibil_score": 750.5}, {**SAMPLE, "cibil_score": "750"},
            {**SAMPLE, "cibil_score": True}, {**SAMPLE, "cibil_score": None},
            {**SAMPLE, "income_annum": 0}, {**SAMPLE, "loan_amount": -1},
            {**SAMPLE, "loan_term": 2.5}, {**SAMPLE, "no_of_dependents": -1},
            {**SAMPLE, "residential_assets_value": -100000},
            {**SAMPLE, "education": "Unknown"}, {**SAMPLE, "unexpected": 1},
            {k: v for k, v in SAMPLE.items() if k != "loan_amount"},
        ]
        with TestClient(app) as client:
            for sample in invalid:
                with self.subTest(sample=sample):
                    self.assertEqual(client.post("/predict", json=sample).status_code, 422)

    def test_missing_artifacts_return_503(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LOAN_ARTIFACT_DIR": directory}):
                with self.assertLogs("api", level="ERROR"):
                    with TestClient(app) as client:
                        self.assertEqual(client.get("/health").status_code, 503)
                        self.assertEqual(client.post("/predict", json=SAMPLE).status_code, 503)

    def test_saved_preprocessing_uses_training_rows_only(self):
        data = pd.read_csv(ROOT / "data" / "loan_approval_dataset.csv")
        data.columns = data.columns.str.strip()
        split = json.loads((ROOT / "reports" / "split_indices.json").read_text())
        self.assertFalse(set(split["train"]) & set(split["test"]))
        pipeline = joblib.load(ROOT / "artifacts" / "loan_pipeline.joblib")
        cleaned_train = clean_features(data.loc[split["train"], FEATURES])
        learned = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_
        np.testing.assert_allclose(learned, cleaned_train[NUMERIC_FEATURES].median().to_numpy())
        self.assertEqual(pipeline.named_steps["preprocess"].feature_names_in_.tolist(), FEATURES)
        self.assertNotIn("loan_id", FEATURES)
        self.assertNotIn("loan_status", FEATURES)


if __name__ == "__main__":
    unittest.main()
