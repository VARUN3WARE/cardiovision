"""Model loading and inference."""
from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd

from app.config import FEATURE_INFO_PATH, FEATURES_PATH, METRICS_PATH, MODELS_DIR, TARGETS
from app.encoders import encode_dict, load_encoding


def load_features() -> list[str]:
    with open(FEATURES_PATH) as f:
        return json.load(f)


def load_metrics() -> dict:
    with open(METRICS_PATH) as f:
        return json.load(f)


def load_feature_info() -> dict:
    with open(FEATURE_INFO_PATH) as f:
        return json.load(f)


def load_models() -> dict[str, Any]:
    models = {}
    for target in TARGETS:
        path = MODELS_DIR / f"{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model for '{target}' not found at {path}. Run `python train.py` first."
            )
        models[target] = joblib.load(path)
    return models


def to_frame(features: dict[str, Any], encoding: dict, feature_names: list[str]) -> pd.DataFrame:
    """One encoded row as a DataFrame (the models read columns by name)."""
    return pd.DataFrame([encode_dict(features, encoding, feature_names)], columns=feature_names, dtype=float)


def band_for(p: float, bands: list[dict]) -> dict:
    for b in bands:
        if b["lo"] <= p < b["hi"] or (b["hi"] >= 1 and p >= b["lo"]):
            return b
    return bands[-1]


def predict_all(models: dict[str, Any], X: pd.DataFrame, metrics: dict) -> dict:
    result = {}
    for target in TARGETS:
        model = models[target]
        p = float(model.predict_proba(X)[0, 1])
        threshold = float(getattr(model, "threshold_", 0.5))
        band = band_for(p, metrics[target]["bands"])
        result[target] = {
            "probability": p,
            "prediction": int(p >= threshold),
            "threshold": threshold,
            "band": band["label"],
            "band_observed_rate": band.get("observed_rate"),
        }
    return result


def load_encoding_map() -> dict:
    return load_encoding()
