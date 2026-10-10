"""FastAPI app exposing the prediction, explanation and model-information endpoints."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.cardio_models import DERIVED_SOURCES
from app.config import METRICS_PATH, TARGETS
from app.model import (
    load_encoding_map, load_feature_info, load_features, load_metrics, load_models,
    predict_all, to_frame,
)
from app.schemas import (
    COLUMN_TO_SCHEMA, FIXED_COLUMNS, QUICK_FIELDS, SCHEMA_TO_COLUMN, ExplainResponse,
    PatientFeatures, PredictionResponse, RangeWarning, ShapExplanation, TargetResult,
    VesselProbability, obesity_flag
)
from app.shap_explainer import explain as explain_row

SAFETY = (
    "Decision support / educational use only. Estimates come from a model trained on 303 patients "
    "referred for coronary angiography at a single centre (71% had CAD); it is not validated for "
    "screening and is not a diagnosis. It does not replace angiography, formal diagnostic imaging "
    "or professional medical judgement."
)


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
def _metrics():
    return load_metrics()


@lru_cache
def _feature_info():
    return load_feature_info()


def _model_version() -> str:
    try:
        mtime = METRICS_PATH.stat().st_mtime
    except FileNotFoundError:
        return "untrained"
    return "trained " + datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the caches so the first request is fast; tolerate missing models (health reports it).
    try:
        _models(), _features(), _encoding(), _metrics(), _feature_info()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="CardioVision API",
    description="Coronary artery disease and vessel-level stenosis risk (Track A). " + SAFETY,
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _require_models():
    try:
        return _models(), _metrics(), _feature_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


def to_dataset_columns(features: PatientFeatures, info: dict) -> tuple[dict, list[str]]:
    """Schema fields -> dataset columns, filling omitted values with the training-cohort default."""
    raw = features.model_dump()
    out, imputed = {}, []
    for key, val in raw.items():
        col = SCHEMA_TO_COLUMN.get(key, key)
        if val is None:
            val = info["features"][col]["default"]
            imputed.append(key)
        out[col] = val
    # BMI may have been imputed: keep Obesity consistent with the BMI actually used.
    if "Obesity" in imputed:
        out["Obesity"] = obesity_flag(out["BMI"])
    out.update(FIXED_COLUMNS)
    return out, imputed


def range_warnings(columns: dict, imputed: list[str], info: dict) -> list[RangeWarning]:
    warnings = []
    for col, spec in info["features"].items():
        if spec["type"] != "numeric" or COLUMN_TO_SCHEMA.get(col, col) in imputed:
            continue
        v = float(columns[col])
        if v < spec["min"] or v > spec["max"]:
            warnings.append(RangeWarning(field=COLUMN_TO_SCHEMA.get(col, col), column=col, value=v,
                                         train_min=spec["min"], train_max=spec["max"]))
    return warnings


def _predict(features: PatientFeatures):
    models, metrics, info = _require_models()
    columns, imputed = to_dataset_columns(features, info)
    X = to_frame(columns, _encoding(), _features())
    result = predict_all(models, X, metrics)
    response = dict(
        cad_probability=result["cad"]["probability"],
        vessel_probabilities=VesselProbability(**{v: result[v]["probability"] for v in ("lad", "lcx", "rca")}),
        cad_prediction=result["cad"]["prediction"],
        vessel_predictions=VesselProbability(**{v: result[v]["prediction"] for v in ("lad", "lcx", "rca")}),
        results={t: TargetResult(**result[t]) for t in TARGETS},
        imputed=imputed,
        warnings=range_warnings(columns, imputed, info),
        model_version=_model_version(),
        safety_disclaimer=SAFETY,
    )
    return response, models, X


@app.get("/health")
def health():
    try:
        _models()
        return {"status": "ok", "models_loaded": True, "n_features": len(_features()),
                "model_version": _model_version(), "safety_disclaimer": SAFETY}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail={"status": "models_missing", "models_loaded": False,
                                                     "message": str(e)})


@app.get("/features")
def features():
    """Input columns, category codes, training ranges / defaults and model influence per feature."""
    _, _, info = _require_models()
    return {
        "features": _features(),
        "encoding": _encoding(),
        "schema_to_column": SCHEMA_TO_COLUMN,
        "fixed_columns": FIXED_COLUMNS,
        "quick_fields": QUICK_FIELDS,
        "feature_info": info["features"],
        "influence": info["influence"],
        "derived_sources": DERIVED_SOURCES,
        "safety_disclaimer": SAFETY,
    }


@app.get("/metrics")
def metrics():
    """Cross-validated performance, operating points and calibration bands per target."""
    _require_models()
    return {"metrics": _metrics(), "model_version": _model_version(), "safety_disclaimer": SAFETY}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    response, _, _ = _predict(features)
    return PredictionResponse(**response)


@app.post("/explain", response_model=ExplainResponse)
def explain(features: PatientFeatures):
    response, models, X = _predict(features)
    exps = {t: ShapExplanation(**explain_row(models[t], X)) for t in TARGETS}
    return ExplainResponse(**response, explanation=exps["cad"], explanations=exps)
