"""Model loading and inference.

Trains/loads 4 classifiers:
  - CAD (overall coronary artery disease)
  - LAD stenosis
  - LCX stenosis
  - RCA stenosis

Leakage guard: input feature list must NOT include LAD, LCX, RCA, or Cath.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURES_PATH = MODELS_DIR / "features.json"

# Columns that must NEVER be used as inputs (target leakage).
LEAKAGE_COLUMNS = {"LAD", "LCX", "RCA", "Cath", "lad", "lcx", "rca", "cath"}

TARGETS = ["cad", "lad", "lcx", "rca"]


def load_features() -> list[str]:
    """Return the ordered list of input feature names used at training time."""
    with open(FEATURES_PATH) as f:
        return json.load(f)


def load_models() -> dict[str, Any]:
    """Load all 4 trained classifiers from disk."""
    models = {}
    for target in TARGETS:
        path = MODELS_DIR / f"{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model for '{target}' not found at {path}. Run `python train.py` first."
            )
        models[target] = joblib.load(path)
    return models


def features_to_vector(features: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    """Order a feature dict into the model's expected input vector."""
    return np.array([[float(features[name]) for name in feature_names]])


def predict_all(models: dict[str, Any], feature_names: list[str], features: dict[str, Any]):
    """Return probabilities and class predictions for all 4 targets."""
    x = features_to_vector(features, feature_names)
    result = {}
    for target in TARGETS:
        model = models[target]
        proba = float(model.predict_proba(x)[0, 1])
        pred = int(proba >= 0.5)
        result[target] = {"probability": proba, "prediction": pred}
    return result
