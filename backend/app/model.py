"""Model loading and inference."""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np

from app.config import FEATURES_PATH, TARGETS
from app.encoders import encode_dict, load_encoding


def load_features() -> list[str]:
    with open(FEATURES_PATH) as f:
        return json.load(f)


def load_models() -> dict[str, Any]:
    from app.config import MODELS_DIR
    models = {}
    for target in TARGETS:
        path = MODELS_DIR / f"{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model for '{target}' not found at {path}. Run `python train.py` first."
            )
        models[target] = joblib.load(path)
    return models


def predict_all(models: dict[str, Any], feature_names: list[str],
                encoding: dict, features: dict[str, Any]) -> dict:
    vec = encode_dict(features, encoding, feature_names)
    x = np.array([vec], dtype=float)
    result = {}
    for target in TARGETS:
        model = models[target]
        proba = float(model.predict_proba(x)[0, 1])
        pred = int(proba >= 0.5)
        result[target] = {"probability": proba, "prediction": pred}
    return result


def load_encoding_map() -> dict:
    return load_encoding()
