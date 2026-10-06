"""FastAPI app exposing /predict and /explain endpoints."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException

from app.encoders import encode_dict
from app.model import load_encoding_map, load_features, load_models, predict_all
from app.schemas import (
    ExplainResponse, PatientFeatures, PredictionResponse,
    SCHEMA_TO_COLUMN, VesselProbability,
)
from app.shap_explainer import ShapExplainer

app = FastAPI(
    title="CardioVision API",
    description="Cardiovascular risk prediction (Track A) — decision support only.",
    version="0.1.0",
)

SAFETY = (
    "Predictions are for decision support / educational purposes only and are "
    "not a substitute for formal diagnostic imaging."
)


def to_dataset_columns(features: PatientFeatures) -> dict:
    """Convert schema field names to dataset column names."""
    raw = features.model_dump()
    out = {}
    for key, val in raw.items():
        col = SCHEMA_TO_COLUMN.get(key, key)
        out[col] = val
    return out


@lru_cache
def _models():
    return load_models()


@lru_cache
def _features():
    return load_features()


@lru_cache
def _encoding():
    return load_encoding_map()


@lru_cache
def _explainers() -> dict[str, ShapExplainer]:
    return {target: ShapExplainer(model) for target, model in _models().items()}


@app.get("/health")
def health():
    return {"status": "ok", "safety_disclaimer": SAFETY}


@app.get("/features")
def features():
    """Return the feature list and categorical encoding (for the frontend form)."""
    return {"features": _features(), "encoding": _encoding(), "safety_disclaimer": SAFETY}


def _predict(features: PatientFeatures) -> dict:
    ds_features = to_dataset_columns(features)
    return predict_all(_models(), _features(), _encoding(), ds_features)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    try:
        result = _predict(features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return PredictionResponse(
        cad_probability=result["cad"]["probability"],
        vessel_probabilities=VesselProbability(
            lad=result["lad"]["probability"],
            lcx=result["lcx"]["probability"],
            rca=result["rca"]["probability"],
        ),
        cad_prediction=result["cad"]["prediction"],
        vessel_predictions=VesselProbability(
            lad=result["lad"]["prediction"],
            lcx=result["lcx"]["prediction"],
            rca=result["rca"]["prediction"],
        ),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(features: PatientFeatures):
    try:
        result = _predict(features)
        ds_features = to_dataset_columns(features)
        vec = encode_dict(ds_features, _encoding(), _features())
        x = np.array([vec], dtype=float)
        exps = {t: e.explain(x, _features()) for t, e in _explainers().items()}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return ExplainResponse(
        cad_probability=result["cad"]["probability"],
        vessel_probabilities=VesselProbability(
            lad=result["lad"]["probability"],
            lcx=result["lcx"]["probability"],
            rca=result["rca"]["probability"],
        ),
        cad_prediction=result["cad"]["prediction"],
        vessel_predictions=VesselProbability(
            lad=result["lad"]["prediction"],
            lcx=result["lcx"]["prediction"],
            rca=result["rca"]["prediction"],
        ),
        explanation=exps["cad"],
        explanations=exps,
    )
