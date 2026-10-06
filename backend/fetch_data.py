"""Load the Extension of Z-Alizadeh Sani dataset (UCI id=411) from the local
xlsx and save as CSV. Prints an EDA summary.

The xlsx is downloaded manually from UCI (the ucimlrepo API does not serve
this dataset). See README for the source:
  https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
XLSX = DATA_DIR / "extention of Z-Alizadeh sani dataset.xlsx"
CSV_PATH = DATA_DIR / "ZAlizadehSani.csv"

print(f"Reading {XLSX.name} ...")
df = pd.read_excel(XLSX)
df.to_csv(CSV_PATH, index=False)
print(f"Saved -> {CSV_PATH.name}  ({df.shape[0]} rows x {df.shape[1]} cols)")

print("\n=== COLUMNS (index, name, dtype) ===")
for i, c in enumerate(df.columns):
    print(f"  {i:2d}  {c}  ({df[c].dtype})")

print("\n=== TARGET / LEAKAGE COLUMN VALUES ===")
for col in ["Cath", "LAD", "LCX", "RCA"]:
    if col in df.columns:
        print(f"\n-- {col} --")
        print(df[col].value_counts(dropna=False))

print("\n=== MISSING VALUES (cols with any) ===")
miss = df.isna().sum()
print(miss[miss > 0].sort_values(ascending=False) if miss.sum() > 0 else "No missing values")

print("\n=== NUMERIC SUMMARY (first 12 numeric cols) ===")
num = df.select_dtypes(include="number")
print(num.describe().T.head(12))

print("\n=== CAD label balance (Cath == 'Cad' or 'CAD') ===")
if "Cath" in df.columns:
    cad = df["Cath"].astype(str).str.lower().eq("cad")
    print(cad.value_counts())
    print(f"CAD prevalence: {cad.mean():.3f}")

print("\nDone.")
