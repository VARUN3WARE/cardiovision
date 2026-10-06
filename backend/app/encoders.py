"""Categorical encoding for the Z-Alizadeh Sani dataset."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.config import (
    ENCODING_PATH, LEAKAGE_COLUMNS, TARGET_POSITIVE,
)


def learn_encoding(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build a category->code map for every object column except leakage/targets."""
    encoding: dict[str, dict[str, float]] = {}
    for col in df.columns:
        if col in LEAKAGE_COLUMNS:
            continue
        if df[col].dtype == object:
            cats = sorted(df[col].dropna().astype(str).unique().tolist())
            encoding[col] = {c: float(i) for i, c in enumerate(cats)}
    return encoding


def save_encoding(encoding: dict[str, dict[str, float]]) -> None:
    ENCODING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENCODING_PATH, "w") as f:
        json.dump(encoding, f, indent=2)


def load_encoding() -> dict[str, dict[str, float]]:
    with open(ENCODING_PATH) as f:
        return json.load(f)


def encode_value(col: str, value: Any, encoding: dict[str, dict[str, float]]) -> float:
    if col in encoding:
        key = str(value)
        if key not in encoding[col]:
            # Code 0 is a real category (e.g. Sex=Fmale, BBB=LBBB), so never fall back silently.
            raise ValueError(f"unknown category {value!r} for {col!r}; expected one of {list(encoding[col])}")
        return encoding[col][key]
    v = float(value)
    if v != v:
        raise ValueError(f"missing value for {col!r}")
    return v


def encode_dataframe(df: pd.DataFrame, encoding: dict[str, dict[str, float]],
                     feature_names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in feature_names:
        if col in encoding:
            out[col] = df[col].astype(str).map(encoding[col]).fillna(0.0)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return out


def encode_dict(features: dict[str, Any], encoding: dict[str, dict[str, float]],
                feature_names: list[str]) -> list[float]:
    missing = [col for col in feature_names if col not in features]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    return [encode_value(col, features[col], encoding) for col in feature_names]


def binarize_target(series: pd.Series, positive_label: str) -> pd.Series:
    return (series.astype(str) == positive_label).astype(int)
