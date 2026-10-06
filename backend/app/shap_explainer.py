"""SHAP-based clinical interpretability.

Uses XGBoost's native SHAP (pred_contribs) rather than shap.TreeExplainer,
to avoid the shap 0.49 / xgboost 3.x incompatibility where base_score is
stored as a string like '[5E-1]'.

Returns per-feature SHAP values for the CAD prediction (overall risk),
which the frontend renders as a waterfall/bar breakdown.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb


class ShapExplainer:
    """Wraps an XGBoost model and computes SHAP values via pred_contribs.

    The last column of pred_contribs output is the base value (expected output);
    the rest are per-feature SHAP contributions.
    """

    def __init__(self, model: Any):
        self.model = model
        self.booster = model.get_booster()

    def explain(self, x: np.ndarray, feature_names: list[str]) -> dict:
        dmatrix = xgb.DMatrix(x, feature_names=feature_names)
        contribs = self.booster.predict(dmatrix, pred_contribs=True)
        row = contribs[0]
        base = float(row[-1])
        shap_values = row[:-1]
        return {
            "base_value": base,
            "feature_names": list(feature_names),
            "shap_values": [float(v) for v in shap_values],
            "feature_values": [float(v) for v in x[0]],
        }
