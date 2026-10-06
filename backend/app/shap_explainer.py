"""SHAP-based clinical interpretability.

Uses TreeExplainer for the XGBoost / gradient-boosting classifiers.
Returns per-feature SHAP values for the CAD prediction (overall risk),
which the frontend renders as a waterfall/bar breakdown.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import shap


class ShapExplainer:
    """Wraps a TreeExplainer for the CAD model.

    The explainer is built once on model load and reused across requests.
    """

    def __init__(self, cad_model: Any):
        self.explainer = shap.TreeExplainer(cad_model)

    def explain(self, x: np.ndarray, feature_names: list[str]) -> dict:
        """Return base value, feature names, shap values, and feature values."""
        shap_values = self.explainer.shap_values(x)[0]
        base = float(self.explainer.expected_value)
        if isinstance(base, list):  # some explainers return a list
            base = float(base[1] if len(base) > 1 else base[0])
        return {
            "base_value": base,
            "feature_names": list(feature_names),
            "shap_values": [float(v) for v in shap_values],
            "feature_values": [float(v) for v in x[0]],
        }
