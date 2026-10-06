"""Per-feature attributions for the production models.

The production models are log-odds blends of logistic regressions and gradient-boosted trees,
so attributions are computed exactly, member by member (app.cardio_models.contributions):
linear terms coef * (x - mean), TreeSHAP from LightGBM / XGBoost natively, averaged in log-odds
and passed through the Platt recalibration where present. They satisfy

    base_value + sum(shap_values) == model log-odds   (to float precision)

No `shap` package is involved (shap.TreeExplainer is incompatible with xgboost 3.x).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.cardio_models import ClinicalFeatures, contributions, display_attributions, margin

_DERIVED = ClinicalFeatures(fix_encodings=False, engineered=True, log_labs=False)


def explain(model: Any, X: pd.DataFrame) -> dict:
    """Attributions for a single encoded row X (DataFrame with the 55 input columns)."""
    C, base = contributions(model, X)
    C = display_attributions(C)
    raw = X.iloc[0]
    values_frame = _DERIVED.transform(X).iloc[0]  # raw columns + derived scores, untransformed
    names = list(C.columns)
    shap_values = C.iloc[0].to_numpy(float)
    base_value = float(base[0])
    # Guard the additivity property the UI relies on. XGBoost computes TreeSHAP in float32, so allow
    # a small relative error (1e-4 of the log-odds is < 1e-6 in probability).
    logit = float(margin(model, X)[0])
    if not np.isclose(base_value + shap_values.sum(), logit, rtol=1e-4, atol=1e-3):
        raise RuntimeError(f"attributions do not add up: {base_value + shap_values.sum()} vs {logit}")
    return {
        "base_value": base_value,
        "feature_names": names,
        "shap_values": [float(v) for v in shap_values],
        "feature_values": [float(raw[c]) if c in raw.index else float(values_frame[c]) for c in names],
        "derived": [c not in raw.index for c in names],
    }
