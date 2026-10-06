"""Training pipeline for Track A cardiovascular risk prediction.

Trains 4 classifiers on the Z-Alizadeh Sani extension dataset:
  - CAD (overall coronary artery disease)
  - LAD stenosis
  - LCX stenosis
  - RCA stenosis

Leakage guard: LAD, LCX, RCA, and Cath columns are dropped from inputs.
Run once:  python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from app.model import LEAKAGE_COLUMNS, MODELS_DIR, TARGETS

DATA_PATH = Path(__file__).resolve().parent / "data" / "ZAlizadehSani.csv"

# Map dataset target columns to our target names. Adjust after EDA.
TARGET_COLUMN_MAP = {
    "cad": "Cath",   # overall CAD label (dataset uses Cath for disease presence)
    "lad": "LAD",
    "lcx": "LCX",
    "rca": "RCA",
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def build_features(df: pd.DataFrame) -> list[str]:
    """All columns except leakage + target columns."""
    drop = LEAKAGE_COLUMNS | set(TARGET_COLUMN_MAP.values())
    features = [c for c in df.columns if c not in drop]
    return features


def binarize_target(series: pd.Series) -> pd.Series:
    """Convert vessel stenosis labels to binary (>=50% narrowing = 1)."""
    # Z-Alizadeh Sani uses values like 'No', 'Mild', 'Moderate', 'Severe', 'Total'
    # or numeric percentages. Adjust this logic after EDA.
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["severe", "total", "100", "90", "80", "70", "60", "50"]).astype(int)
    return (series >= 50).astype(int)


def train_one(df: pd.DataFrame, feature_names: list[str], target_col: str):
    y = binarize_target(df[target_col])
    X = df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    # Cross-validated metrics for honest reporting.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:
        auc = float("nan")
    print(f"\n=== {target_col} ===")
    print(classification_report(y, pred, zero_division=0))
    print(f"CV ROC-AUC: {auc:.3f}")
    # Refit on full data for the served model.
    model.fit(X, y)
    return model


def main():
    df = load_dataset()
    feature_names = build_features(df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "features.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Using {len(feature_names)} input features (leakage columns excluded).")
    for target in TARGETS:
        target_col = TARGET_COLUMN_MAP[target]
        if target_col not in df.columns:
            print(f"WARNING: target column '{target_col}' not found — skipping {target}.")
            continue
        model = train_one(df, feature_names, target_col)
        joblib.dump(model, MODELS_DIR / f"{target}.joblib")
        print(f"Saved {target}.joblib")
    print("\nDone. Start the API with:  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
