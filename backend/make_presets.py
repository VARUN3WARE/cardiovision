"""Pick real dataset patients as demo presets for the frontend.

Uses the OUT-OF-FOLD probabilities that train.py saved (each patient scored by production-model
fits that never saw them, averaged over the 5 CV repeats), so presets are not cherry-picked
from in-sample fits and the UI can show honest "held-out" estimates next to the live score.
Writes ../frontend/src/presets.json. Run after train.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import MODELS_DIR, TARGETS
from app.schemas import COLUMN_TO_SCHEMA, FIXED_COLUMNS, PatientFeatures
from train import load_dataset

OUT_PATH = Path(__file__).resolve().parent.parent / "frontend" / "src" / "presets.json"


def to_schema(row: pd.Series) -> dict:
    values = {}
    for col, v in row.items():
        name = COLUMN_TO_SCHEMA.get(col, col)
        if name not in PatientFeatures.model_fields or col in FIXED_COLUMNS:
            continue
        values[name] = v.item() if hasattr(v, "item") else v
    values["BMI"] = round(values["BMI"], 1)
    PatientFeatures(**values)  # validate against the API schema
    return values


def main():
    df = load_dataset()
    oof = np.load(MODELS_DIR / "oof_predictions.npz")
    P = pd.DataFrame({t: oof[t].mean(axis=0) for t in TARGETS}, index=df.index)
    sten = {v: df[v.upper()] == "Stenotic" for v in ("lad", "lcx", "rca")}
    normal = (df["Cath"] == "Normal") & ~sten["lad"] & ~sten["lcx"] & ~sten["rca"]

    vessels = P[["lad", "lcx", "rca"]]
    candidates = [
        ("low", "Low risk", "no ≥50% stenosis in any artery",
         normal, -P.max(axis=1)),
        ("lad", "Single-vessel (LAD)", "≥50% stenosis of the LAD only",
         sten["lad"] & ~sten["lcx"] & ~sten["rca"], P["lad"] - P[["lcx", "rca"]].max(axis=1)),
        ("rca", "Single-vessel (RCA)", "≥50% stenosis of the RCA only",
         ~sten["lad"] & ~sten["lcx"] & sten["rca"], P["rca"] - P[["lad", "lcx"]].max(axis=1)),
        ("triple", "Three-vessel disease", "≥50% stenosis of the LAD, LCX and RCA",
         sten["lad"] & sten["lcx"] & sten["rca"], vessels.min(axis=1)),
        # An honest borderline case: CAD present but the vessel models are unsure.
        ("uncertain", "Uncertain case", "CAD with a borderline vessel picture",
         df["Cath"] == "CAD", -(vessels - 0.5).abs().mean(axis=1)),
    ]

    # Labelled examples should be cases the held-out models call correctly, at each model's own
    # learned decision threshold, for the targets the label is about; the "uncertain" preset
    # exists to show the opposite. (No isolated-RCA patient is called correctly on all four
    # targets: the vessel models also over-call LAD, which the UI shows as a false alarm.)
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    thresholds = {t: metrics[t]["thresholds"]["f1"] for t in TARGETS}
    truth_df = pd.DataFrame({"cad": df["Cath"] == "CAD", **sten})
    correct_on = {t: (P[t] >= thresholds[t]) == truth_df[t] for t in TARGETS}
    about = {"low": TARGETS, "lad": ["cad", "lad"], "rca": ["cad", "rca"], "triple": TARGETS}

    presets = []
    for key, label, truth, mask, score in candidates:
        if key in about:
            ok = pd.concat([correct_on[t] for t in about[key]], axis=1).all(axis=1)
            if (mask & ok).any():
                mask = mask & ok
        idx = score[mask].idxmax()
        row = df.loc[idx]
        gt = {"cad": bool(row["Cath"] == "CAD"), **{v: bool(sten[v][idx]) for v in ("lad", "lcx", "rca")}}
        if key == "uncertain":
            truth = "CAD; ≥50% stenosis of the " + (", ".join(v.upper() for v in ("lad", "lcx", "rca") if gt[v]) or "no major vessel")
        presets.append({
            "id": key,
            "label": label,
            "description": truth,
            "patient_index": int(idx),
            "ground_truth": gt,
            "oof_probabilities": {t: round(float(P.loc[idx, t]), 3) for t in TARGETS},
            "features": to_schema(row.drop(labels=["Cath", "LAD", "LCX", "RCA"])),
        })
        print(f"{label:22} row={idx:3}  held-out={presets[-1]['oof_probabilities']}  truth={gt}")

    # UTF-8 as written ("≥" stays readable), so a fresh run reproduces the committed file byte for byte.
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(presets)} presets to {OUT_PATH}")


if __name__ == "__main__":
    main()
