"""FastAPI inference service. Run: uvicorn api:app --host 0.0.0.0 --port 8000"""

from contextlib import asynccontextmanager
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from loan_features import FEATURES

logger = logging.getLogger(__name__)
NonNegativeNumber = Annotated[float, Field(ge=0, strict=True)]
PositiveNumber = Annotated[float, Field(gt=0, strict=True)]
NonNegativeInteger = Annotated[int, Field(ge=0, strict=True)]
PositiveInteger = Annotated[int, Field(gt=0, strict=True)]


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)

    no_of_dependents: NonNegativeInteger | None
    education: Literal["Graduate", "Not Graduate"] | None
    self_employed: Literal["Yes", "No"] | None
    income_annum: PositiveNumber
    loan_amount: PositiveNumber
    loan_term: PositiveInteger
    cibil_score: Annotated[int, Field(ge=300, le=900, strict=True)]
    residential_assets_value: NonNegativeNumber | None
    commercial_assets_value: NonNegativeNumber | None
    luxury_assets_value: NonNegativeNumber | None
    bank_asset_value: NonNegativeNumber | None


class Prediction(BaseModel):
    prediction: Literal["Approved", "Rejected"]
    predicted_class: Literal[0, 1]
    approval_score: float = Field(ge=0, le=1)
    rejection_score: float = Field(ge=0, le=1)
    score_type: Literal["uncalibrated_model_probability"]
    model_name: str
    imputed_fields: list[str]


@asynccontextmanager
async def lifespan(app):
    app.state.pipeline = None
    app.state.metadata = None
    artifact_dir = Path(os.environ.get(
        "LOAN_ARTIFACT_DIR", str(Path(__file__).resolve().parent / "artifacts")
    ))
    try:
        model_path = artifact_dir / "loan_pipeline.joblib"
        metadata = json.loads((artifact_dir / "model_metadata.json").read_text())
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
            raise ValueError("The model and metadata do not match. Rerun training.")
        for package, expected in metadata["versions"].items():
            if importlib.metadata.version(package) != expected:
                raise RuntimeError(f"Install requirements.txt: {package} must be {expected}.")
        # This is a trusted, locally trained artifact. Never load user uploads.
        pipeline = joblib.load(model_path)
        if metadata["feature_names"] != FEATURES or list(pipeline.classes_) != [0, 1]:
            raise ValueError("Saved model schema or class mapping differs from this API.")
        app.state.pipeline = pipeline
        app.state.metadata = metadata
    except Exception:
        logger.exception("Model unavailable. Run training.py and check requirements.txt.")
    yield
    app.state.pipeline = None
    app.state.metadata = None


app = FastAPI(title="Loan Approval Predictor", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if app.state.pipeline is None:
        return JSONResponse(status_code=503, content={
            "status": "unavailable", "model_loaded": False,
            "detail": "Model unavailable. Run training.py and check the server log.",
        })
    return {"status": "ok", "model_loaded": True,
            "model_name": app.state.metadata["model_name"]}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    if app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="The trained model is unavailable.")
    values = applicant.model_dump()
    frame = pd.DataFrame([values], columns=FEATURES)
    pipeline = app.state.pipeline
    try:
        # Inference only. No fit(), tuning, or separate preprocessing here.
        predicted = int(pipeline.predict(frame)[0])
        positive_column = list(pipeline.classes_).index(1)
        score = float(pipeline.predict_proba(frame)[0, positive_column])
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check the server log.")
    return Prediction(
        prediction="Approved" if predicted == 1 else "Rejected",
        predicted_class=predicted,
        approval_score=score, rejection_score=1.0 - score,
        score_type="uncalibrated_model_probability",
        model_name=app.state.metadata["model_name"],
        imputed_fields=[name for name, value in values.items() if value is None],
    )
