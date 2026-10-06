"""In-process test of the full pipeline (no HTTP / no httpx needed).

Calls the model + encoder + SHAP logic directly, exactly as the API does.
Run:  .venv/bin/python test_pipeline.py
"""
from __future__ import annotations

from app.encoders import encode_dict
from app.model import load_encoding_map, load_features, load_models, predict_all
from app.schemas import SCHEMA_TO_COLUMN, PatientFeatures
from app.shap_explainer import ShapExplainer
import numpy as np

# Sample patient using schema field names.
patient = {
    "Age": 60, "Weight": 78, "Length": 170, "Sex": "Male", "BMI": 27.0,
    "DM": 1, "HTN": 1, "CurrentSmoker": 1, "EX_Smoker": 0, "FH": 1,
    "Obesity": "Y", "CRF": "N", "CVA": "N", "AirwayDisease": "N", "ThyroidDisease": "N",
    "CHF": "N", "DLP": "Y",
    "BP": 145, "PR": 80, "Edema": 0, "WeakPeripheralPulse": "N", "LungRales": "N",
    "SystolicMurmur": "N", "DiastolicMurmur": "N", "TypicalChestPain": 1,
    "Dyspnea": "Y", "FunctionClass": 2, "Atypical": "N", "Nonanginal": "N",
    "ExertionalCP": "N", "LowTHAng": "Y",
    "QWave": 0, "StElevation": 1, "StDepression": 1, "Tinversion": 1,
    "LVH": "N", "PoorRProgression": "N", "BBB": "N",
    "FBS": 120, "CR": 1.1, "TG": 180, "LDL": 150, "HDL": 38, "BUN": 16, "ESR": 12,
    "HB": 13.5, "K": 4.2, "Na": 140, "WBC": 7500, "Lymph": 30, "Neut": 65, "PLT": 250,
    "EF_TTE": 50, "RegionRWMA": 1, "VHD": "N",
}

# Every schema field must map onto a model feature and vice versa; a missing
# mapping would silently encode the feature as 0 for every request.
_mapped = {SCHEMA_TO_COLUMN.get(f, f) for f in PatientFeatures.model_fields}
_features = set(load_features())
assert _mapped == _features, (
    f"schema/feature mismatch: unmapped schema fields {_mapped - _features}, "
    f"features without a schema field {_features - _mapped}"
)
PatientFeatures(**patient)  # sample patient must pass validation

# Convert schema field names -> dataset column names (same as main.to_dataset_columns).
ds_features = {SCHEMA_TO_COLUMN.get(k, k): v for k, v in patient.items()}

print("=== Loading models ===")
models = load_models()
feature_names = load_features()
encoding = load_encoding_map()
print(f"Loaded {len(models)} models, {len(feature_names)} features, "
      f"{len(encoding)} categorical encodings.")

print("\n=== PREDICT ===")
result = predict_all(models, feature_names, encoding, ds_features)
for tgt in ["cad", "lad", "lcx", "rca"]:
    print(f"  {tgt.upper():4s} prob={result[tgt]['probability']:.3f}  pred={result[tgt]['prediction']}")

print("\n=== EXPLAIN (SHAP for CAD model) ===")
vec = encode_dict(ds_features, encoding, feature_names)
x = np.array([vec], dtype=float)
explainer = ShapExplainer(models["cad"])
exp = explainer.explain(x, feature_names)
print(f"  base value: {exp['base_value']:.3f}")
ranked = sorted(
    zip(exp["feature_names"], exp["shap_values"], exp["feature_values"]),
    key=lambda t: -abs(t[1]),
)[:10]
print("  Top 10 SHAP contributors (CAD):")
for n, s, v in ranked:
    print(f"    {n:22s} shap={s:+.3f}  value={v}")

print("\n=== Pipeline test PASSED ===")
