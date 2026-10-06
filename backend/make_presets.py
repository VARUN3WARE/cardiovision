"""Pick real dataset patients as demo presets for the frontend.

Selection uses OUT-OF-FOLD probabilities (each patient scored by a model that
never saw them), so presets are not cherry-picked from in-sample fits.
Writes ../frontend/src/presets.json. Run after train.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from app.config import TARGETS, TARGET_COLUMN_MAP, TARGET_POSITIVE
from app.encoders import binarize_target, encode_dataframe, load_encoding
from app.model import load_features
from app.schemas import SCHEMA_TO_COLUMN, PatientFeatures
from train import load_dataset

OUT_PATH = Path(__file__).resolve().parent.parent / "frontend" / "src" / "presets.json"


def oof_probabilities(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for target in TARGETS:
        col = TARGET_COLUMN_MAP[target]
        y = binarize_target(df[col], TARGET_POSITIVE[col])
        pos = int(y.sum())
        model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(y) - pos) / pos, eval_metric="logloss",
            random_state=42, n_jobs=4,
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        out[target] = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return pd.DataFrame(out, index=df.index)


def to_schema(row: pd.Series) -> dict:
    values = {}
    for name in PatientFeatures.model_fields:
        v = row[SCHEMA_TO_COLUMN.get(name, name)]
        values[name] = v.item() if hasattr(v, "item") else v
    if isinstance(values["BMI"], float):
        values["BMI"] = round(values["BMI"], 1)
    PatientFeatures(**values)  # validate
    return values


def main():
    df = load_dataset()
    X = encode_dataframe(df, load_encoding(), load_features())
    P = oof_probabilities(df, X)
    sten = {v: df[v.upper()] == "Stenotic" for v in ("lad", "lcx", "rca")}
    normal = (df["Cath"] == "Normal") & ~sten["lad"] & ~sten["lcx"] & ~sten["rca"]

    vessels = P[["lad", "lcx", "rca"]]
    candidates = [
        ("low", "Low risk", "Normal angiogram, no stenosis",
         normal, -P[["cad", "lad", "lcx", "rca"]].max(axis=1)),
        ("lad", "Single-vessel (LAD)", "Isolated LAD stenosis",
         sten["lad"] & ~sten["lcx"] & ~sten["rca"], P["lad"] - P[["lcx", "rca"]].max(axis=1)),
        ("rca", "Single-vessel (RCA)", "Isolated RCA stenosis",
         ~sten["lad"] & ~sten["lcx"] & sten["rca"], P["rca"] - P[["lad", "lcx"]].max(axis=1)),
        ("triple", "Three-vessel disease", "LAD + LCX + RCA stenosis",
         sten["lad"] & sten["lcx"] & sten["rca"], vessels.min(axis=1)),
    ]

    presets = []
    for key, label, truth, mask, score in candidates:
        idx = score[mask].idxmax()
        row = df.loc[idx]
        presets.append({
            "id": key,
            "label": label,
            "description": truth,
            "patient_index": int(idx),
            "ground_truth": {
                "cad": row["Cath"] == "CAD",
                "lad": bool(sten["lad"][idx]),
                "lcx": bool(sten["lcx"][idx]),
                "rca": bool(sten["rca"][idx]),
            },
            "oof_probabilities": {k: round(float(P.loc[idx, k]), 3) for k in P.columns},
            "features": to_schema(row),
        })
        print(f"{label:22} row={idx:3}  oof={presets[-1]['oof_probabilities']}")

    with open(OUT_PATH, "w") as f:
        json.dump(presets, f, indent=2)
    print(f"\nWrote {len(presets)} presets to {OUT_PATH}")


if __name__ == "__main__":
    main()
