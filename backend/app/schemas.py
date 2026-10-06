"""Pydantic schemas for the prediction API.

Field names mirror the Extension of Z-Alizadeh Sani dataset columns.
LAD, LCX, RCA, and Cath are intentionally EXCLUDED (leakage guard).

Categorical fields use the raw dataset string values:
  Sex: "Male" | "Fmale"
  Y/N columns: "Y" | "N"
  BBB: "N" | "LBBB" | "RBBB"
  VHD: "N" | "mild" | "Moderate" | "Severe"
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    # Demographic
    Age: int = Field(..., ge=0, le=120)
    Weight: int = Field(..., ge=20, le=250)
    Length: int = Field(..., ge=100, le=220, description="Height (cm)")
    Sex: Literal["Male", "Fmale"]
    BMI: float = Field(..., ge=10, le=60)
    # History / risk factors
    DM: int = Field(0, ge=0, le=1, description="Diabetes mellitus 0/1")
    HTN: int = Field(0, ge=0, le=1, description="Hypertension 0/1")
    CurrentSmoker: int = Field(0, ge=0, le=1, description="Rename from 'Current Smoker'")
    EX_Smoker: int = Field(0, ge=0, le=1)
    FH: int = Field(0, ge=0, le=1, description="Family history 0/1")
    Obesity: Literal["Y", "N"] = "N"
    CRF: Literal["Y", "N"] = "N"
    CVA: Literal["Y", "N"] = "N"
    AirwayDisease: Literal["Y", "N"] = "N"
    ThyroidDisease: Literal["Y", "N"] = "N"
    CHF: Literal["Y", "N"] = "N"
    DLP: Literal["Y", "N"] = "N"
    # Examination
    BP: int = Field(..., ge=60, le=250, description="Blood pressure (mmHg)")
    PR: int = Field(..., ge=30, le=220, description="Pulse rate (bpm)")
    Edema: int = Field(0, ge=0, le=1)
    WeakPeripheralPulse: Literal["Y", "N"] = "N"
    LungRales: Literal["Y", "N"] = "N"
    SystolicMurmur: Literal["Y", "N"] = "N"
    DiastolicMurmur: Literal["Y", "N"] = "N"
    TypicalChestPain: int = Field(0, ge=0, le=1)
    Dyspnea: Literal["Y", "N"] = "N"
    FunctionClass: int = Field(0, ge=0, le=4, description="NYHA function class")
    Atypical: Literal["Y", "N"] = "N"
    Nonanginal: Literal["Y", "N"] = "N"
    ExertionalCP: Literal["Y", "N"] = "N"
    LowTHAng: Literal["Y", "N"] = "N"
    # ECG
    QWave: int = Field(0, ge=0, le=1)
    StElevation: int = Field(0, ge=0, le=1)
    StDepression: int = Field(0, ge=0, le=1)
    Tinversion: int = Field(0, ge=0, le=1)
    LVH: Literal["Y", "N"] = "N"
    PoorRProgression: Literal["Y", "N"] = "N"
    BBB: Literal["N", "LBBB", "RBBB"] = "N"
    # Laboratory
    FBS: int = Field(..., ge=40, le=500, description="Fasting blood sugar")
    CR: float = Field(..., ge=0.1, le=20, description="Creatinine")
    TG: int = Field(..., ge=20, le=2000, description="Triglycerides")
    LDL: int = Field(..., ge=10, le=500)
    HDL: float = Field(..., ge=5, le=200)
    BUN: int = Field(..., ge=3, le=200)
    ESR: int = Field(..., ge=0, le=200)
    HB: float = Field(..., ge=3, le=25, description="Hemoglobin")
    K: float = Field(..., ge=1, le=10, description="Potassium")
    Na: int = Field(..., ge=120, le=170, description="Sodium")
    WBC: int = Field(..., ge=1000, le=50000, description="White blood cells")
    Lymph: int = Field(..., ge=0, le=100, description="Lymphocyte %")
    Neut: int = Field(..., ge=0, le=100, description="Neutrophil %")
    PLT: int = Field(..., ge=10, le=1500, description="Platelets (x1000/uL)")
    # Echo
    EF_TTE: int = Field(..., ge=5, le=90, description="Ejection fraction (TTE)")
    RegionRWMA: int = Field(0, ge=0, le=4, description="Regional wall motion abnormality (0 = none, 1-4 = region)")
    VHD: Literal["N", "mild", "Moderate", "Severe"] = "N"


# Map our schema field names back to dataset column names (spaces etc.).
SCHEMA_TO_COLUMN = {
    "CurrentSmoker": "Current Smoker",
    "EX_Smoker": "EX-Smoker",
    "AirwayDisease": "Airway disease",
    "ThyroidDisease": "Thyroid Disease",
    "WeakPeripheralPulse": "Weak Peripheral Pulse",
    "LungRales": "Lung rales",
    "SystolicMurmur": "Systolic Murmur",
    "DiastolicMurmur": "Diastolic Murmur",
    "TypicalChestPain": "Typical Chest Pain",
    "FunctionClass": "Function Class",
    "ExertionalCP": "Exertional CP",
    "LowTHAng": "LowTH Ang",
    "QWave": "Q Wave",
    "StElevation": "St Elevation",
    "StDepression": "St Depression",
    "PoorRProgression": "Poor R Progression",
    "EF_TTE": "EF-TTE",
    "RegionRWMA": "Region RWMA",
}


class VesselProbability(BaseModel):
    lad: float = Field(..., ge=0.0, le=1.0)
    lcx: float = Field(..., ge=0.0, le=1.0)
    rca: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    cad_probability: float = Field(..., ge=0.0, le=1.0)
    vessel_probabilities: VesselProbability
    cad_prediction: int
    vessel_predictions: VesselProbability


class ShapExplanation(BaseModel):
    base_value: float
    feature_names: list[str]
    shap_values: list[float]
    feature_values: list[float]


class ExplainResponse(PredictionResponse):
    explanation: ShapExplanation  # overall CAD model (kept for compatibility)
    explanations: dict[str, ShapExplanation]  # per model: cad, lad, lcx, rca
