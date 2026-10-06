"""FastAPI app exposing /predict and /explain endpoints."""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from .model import load_features, load_models, predict_all
from .schemas import (
    ExplainResponse,
    PatientFeatures,
    PredictionResponse,
    VesselProbability,
)
from .shap_explainer import ShapExplainer

app = FastAPI(
    title="CardioVision API",
    description="Cardiovascular risk prediction (Track A) — decision support only.",
)

# Safety disclaimer surfaced in the API itself.
SAFETY = (
    "Predictions are for decision support / educational purposes only and are "
    "not a substitute for formal diagnostic imaging."
)


@lru_cache
def _models():
    return load_models()


@lru_cache
def _features():
    return load_features()


@lru_cache
def _explainer():
    return ShapExplainer(_models()["cad"])


@app.get("/health")
def health():
    return {"status": "ok", "safety_disclaimer": SAFETY}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    try:
        feature_names = _features()
        result = predict_all(_models(), feature_names, features.model_dump())
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
    from .model import features_to_vector
    import numpy as np

    try:
        feature_names = _features()
        result = predict_all(_models(), feature_names, features.model_dump())
        x = features_to_vector(features.model_dump(), feature_names)
        exp = _explainer().explain(np.asarray(x), feature_names)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    base = ExplainResponse(
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
        explanation=exp,
    )
    return base
