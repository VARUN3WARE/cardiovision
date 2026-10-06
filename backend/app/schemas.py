"""Pydantic schemas for the prediction API.

Feature field names mirror the Z-Alizadeh Sani extension dataset columns.
LAD, LCX, RCA, and Cath are intentionally EXCLUDED from inputs (leakage guard).
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Clinical / demographic / ECG / echo features used as model input.

    Field names should match the dataset columns after preprocessing.
    Adjust types/defaults once the dataset schema is finalized in Day 1 EDA.
    """
    age: int = Field(..., ge=0, le=120)
    sex: int = Field(..., description="1 = male, 0 = female")
    bp: float = Field(..., description="Blood pressure (mmHg)")
    pulse: float = Field(..., description="Pulse rate (bpm)")
    cholesterol: float = Field(..., description="Serum cholesterol (mg/dL)")
    # ECG features
    st_elevation: int = Field(0, description="ST elevation 0/1")
    t_inversion: int = Field(0, description="T inversion 0/1")
    lvh: int = Field(0, description="Left ventricular hypertrophy 0/1")
    # Echo
    region_rwma: int = Field(0, description="Region with RWMA 0/1")
    # TODO: add remaining dataset features during EDA (Day 1)


class VesselProbability(BaseModel):
    lad: float = Field(..., ge=0.0, le=1.0)
    lcx: float = Field(..., ge=0.0, le=1.0)
    rca: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    cad_probability: float = Field(..., ge=0.0, le=1.0)
    vessel_probabilities: VesselProbability
    cad_prediction: int = Field(..., description="1 = CAD, 0 = no CAD")
    vessel_predictions: VesselProbability


class ShapExplanation(BaseModel):
    base_value: float
    feature_names: list[str]
    shap_values: list[float]
    feature_values: list[float]


class ExplainResponse(PredictionResponse):
    explanation: ShapExplanation
