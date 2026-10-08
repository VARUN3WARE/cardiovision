"""In-process test of the full pipeline (no HTTP server needed).

Exercises the same code paths as the API: schema validation, imputation, encoding, the four
production models, decision thresholds and exact attributions.
Run:  .venv/bin/python test_pipeline.py      (after python train.py)
"""
from __future__ import annotations

import math

from pydantic import ValidationError

from app.encoders import encode_value
from app.main import _encoding, _feature_info, _features, _predict, explain as explain_endpoint
from app.schemas import QUICK_FIELDS, SCHEMA_TO_COLUMN, PatientFeatures

# Full sample patient using schema field names.
patient = {
    "Age": 60, "Weight": 78, "Length": 170, "Sex": "Male",
    "DM": 1, "HTN": 1, "CurrentSmoker": 1, "EX_Smoker": 0, "FH": 1,
    "CRF": "N", "CVA": "N", "AirwayDisease": "N", "ThyroidDisease": "N",
    "CHF": "N", "DLP": "Y",
    "BP": 145, "PR": 80, "Edema": 0, "WeakPeripheralPulse": "N", "LungRales": "N",
    "SystolicMurmur": "N", "DiastolicMurmur": "N", "TypicalChestPain": 1,
    "Dyspnea": "Y", "FunctionClass": 2, "Atypical": "N", "Nonanginal": "N", "LowTHAng": "Y",
    "QWave": 0, "StElevation": 1, "StDepression": 1, "Tinversion": 1,
    "LVH": "N", "PoorRProgression": "N", "BBB": "N",
    "FBS": 120, "CR": 1.1, "TG": 180, "LDL": 150, "HDL": 38, "BUN": 16, "ESR": 12,
    "HB": 13.5, "K": 4.2, "Na": 140, "WBC": 7500, "Lymph": 30, "Neut": 65, "PLT": 250,
    "EF_TTE": 50, "RegionRWMA": 1, "VHD": "N",
}


def expect_invalid(data: dict, fragment: str) -> None:
    try:
        PatientFeatures(**data)
    except ValidationError as e:
        assert fragment in str(e), f"expected {fragment!r} in {e}"
        return
    raise AssertionError(f"accepted invalid input (expected error containing {fragment!r})")


# 1. Every schema field maps onto a model feature and vice versa (Exertional CP is fixed).
mapped = {SCHEMA_TO_COLUMN.get(f, f) for f in PatientFeatures.model_fields} | {"Exertional CP"}
assert mapped == set(_features()), f"schema/feature mismatch: {mapped ^ set(_features())}"
assert set(QUICK_FIELDS) <= set(PatientFeatures.model_fields)

# 2. Validation rules.
expect_invalid({**patient, "Atypical": "Y"}, "at most one chest-pain type")
expect_invalid({**patient, "EX_Smoker": 1}, "current smoker and an ex-smoker")
expect_invalid({**patient, "BMI": 20}, "does not match weight and height")
expect_invalid({**patient, "Foo": 1}, "Extra inputs are not permitted")
expect_invalid({**patient, "Age": 5}, "greater than or equal to 18")
features = PatientFeatures(**patient)
assert math.isclose(features.BMI, 78 / 1.7 ** 2) and features.Obesity == "Y"
PatientFeatures(**{**patient, "Weight": 78.5, "WBC": 7.1e3})  # decimals accepted

# 3. Unknown categories never fall back to code 0 silently.
try:
    encode_value("BBB", "XYZ", _encoding())
    raise AssertionError("unknown category accepted")
except ValueError:
    pass

# 4. Full prediction + exact attributions.
response = explain_endpoint(features)
print("=== FULL INPUT ===")
for t, r in response.results.items():
    print(f"  {t:3}  p={r.probability:.3f}  threshold={r.threshold:.2f}  prediction={r.prediction}  "
          f"band={r.band} (observed {r.band_observed_rate})")
    exp = response.explanations[t]
    logit = math.log(r.probability / (1 - r.probability))
    total = exp.base_value + sum(exp.shap_values)
    assert abs(total - logit) < 1e-3 + 1e-4 * abs(logit), f"{t}: attributions {total} != logit {logit}"
    assert r.prediction == int(r.probability >= r.threshold)
    top = sorted(zip(exp.feature_names, exp.shap_values, exp.derived), key=lambda z: -abs(z[1]))[:5]
    print("       top: " + ", ".join(f"{n}{'*' if d else ''} {v:+.2f}" for n, v, d in top))
assert response.imputed == [] and response.warnings == []
assert response.safety_disclaimer

# 5. Quick mode: only the quick fields -> the rest is imputed and reported.
quick = {k: v for k, v in patient.items() if k in QUICK_FIELDS}
q, _, _ = _predict(PatientFeatures(**quick))
assert set(q["imputed"]) == set(PatientFeatures.model_fields) - set(quick) - {"BMI", "Obesity"}, q["imputed"]
print(f"\n=== QUICK MODE ({len(quick)} fields, {len(q['imputed'])} imputed) ===")
print("  " + "  ".join(f"{t}={r.probability:.3f}" for t, r in q["results"].items()))

# 6. Minimal input (Age + Sex) still works, and out-of-range values are flagged.
m, _, _ = _predict(PatientFeatures(Age=55, Sex="Fmale"))
assert len(m["imputed"]) > 40
w, _, _ = _predict(PatientFeatures(**{**patient, "Age": 100, "FunctionClass": 4}))
flagged = {x.field for x in w["warnings"]}
assert {"Age", "FunctionClass"} <= flagged, flagged
print(f"\n=== RANGE WARNINGS === {sorted(flagged)}")

assert set(_feature_info()["influence"]) == {"cad", "lad", "lcx", "rca"}

# 7. Exact attributions for every bundle-branch-block category and every demo preset. In the CAD
#    blend the linear member attributes to BBB one-hots and the tree member to raw BBB; both must be
#    summed onto the displayed BBB row (an overwrite once dropped the tree's share and broke /explain).
import json
from pathlib import Path

presets = json.loads((Path(__file__).parent / "../frontend/src/presets.json").read_text())
cases = [{**patient, "BBB": b} for b in ("N", "LBBB", "RBBB")]
cases += [{k: v for k, v in p["features"].items() if k != "BMI"} for p in presets]
for case in cases:
    r = explain_endpoint(PatientFeatures(**case))
    for t, res in r.results.items():
        exp = r.explanations[t]
        logit = math.log(res.probability / (1 - res.probability))
        assert abs(exp.base_value + sum(exp.shap_values) - logit) < 1e-4, f"{t}: attributions do not add up"
print(f"\n=== EXACT ATTRIBUTIONS === {len(cases)} cases x 4 models within 1e-4")
print("\n=== Pipeline test PASSED ===")
