"""Pydantic schemas for the prediction API.

Field names mirror the Extension of Z-Alizadeh Sani dataset columns (see SCHEMA_TO_COLUMN for
the ones that differ). LAD, LCX, RCA and Cath are intentionally EXCLUDED (leakage guard).

Categorical fields use the raw dataset values:
  Sex: "Male" | "Fmale"   (the dataset's spelling)
  Y/N columns: "Y" | "N"
  BBB: "N" | "LBBB" | "RBBB"
  VHD: "N" | "mild" | "Moderate" | "Severe"

Only Age and Sex are required. Any other field may be omitted (or null): the API fills it with
the training-cohort median / most common value and lists it in the response's `imputed` field.
BMI and Obesity are derived from Weight and Height when those are given (the dataset defines
Obesity as BMI >= 25).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

YN = Optional[Literal["Y", "N"]]
Flag = Optional[int]


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Demographic
    Age: float = Field(..., ge=18, le=110, description="Age (years)")
    Sex: Literal["Male", "Fmale"]
    Weight: Optional[float] = Field(None, ge=20, le=250, description="Weight (kg)")
    Length: Optional[float] = Field(None, ge=100, le=220, description="Height (cm)")
    BMI: Optional[float] = Field(None, ge=10, le=70, description="Body mass index; computed from Weight/Height")
    # History / risk factors
    DM: Flag = Field(None, ge=0, le=1, description="Diabetes mellitus")
    HTN: Flag = Field(None, ge=0, le=1, description="Hypertension")
    CurrentSmoker: Flag = Field(None, ge=0, le=1)
    EX_Smoker: Flag = Field(None, ge=0, le=1)
    FH: Flag = Field(None, ge=0, le=1, description="Family history of premature CAD")
    Obesity: YN = Field(None, description="Derived: BMI >= 25")
    CRF: YN = Field(None, description="Chronic renal failure")
    CVA: YN = Field(None, description="Prior stroke")
    AirwayDisease: YN = None
    ThyroidDisease: YN = None
    CHF: YN = Field(None, description="Congestive heart failure")
    DLP: YN = Field(None, description="Dyslipidemia")
    # Examination / symptoms
    BP: Optional[float] = Field(None, ge=60, le=260, description="Systolic blood pressure (mmHg)")
    PR: Optional[float] = Field(None, ge=30, le=220, description="Pulse rate (bpm)")
    Edema: Flag = Field(None, ge=0, le=1)
    WeakPeripheralPulse: YN = None
    LungRales: YN = None
    SystolicMurmur: YN = None
    DiastolicMurmur: YN = None
    TypicalChestPain: Flag = Field(None, ge=0, le=1, description="Typical angina")
    Dyspnea: YN = None
    FunctionClass: Optional[int] = Field(None, ge=0, le=4, description="NYHA class (0 = none)")
    Atypical: YN = Field(None, description="Atypical chest pain")
    Nonanginal: YN = Field(None, description="Non-anginal chest pain")
    LowTHAng: YN = Field(None, description="Low-threshold angina")
    # ECG
    QWave: Flag = Field(None, ge=0, le=1)
    StElevation: Flag = Field(None, ge=0, le=1)
    StDepression: Flag = Field(None, ge=0, le=1)
    Tinversion: Flag = Field(None, ge=0, le=1)
    LVH: YN = Field(None, description="Left ventricular hypertrophy")
    PoorRProgression: YN = None
    BBB: Optional[Literal["N", "LBBB", "RBBB"]] = Field(None, description="Bundle branch block")
    # Laboratory
    FBS: Optional[float] = Field(None, ge=40, le=600, description="Fasting blood sugar (mg/dL)")
    CR: Optional[float] = Field(None, ge=0.1, le=20, description="Creatinine (mg/dL)")
    TG: Optional[float] = Field(None, ge=20, le=2000, description="Triglycerides (mg/dL)")
    LDL: Optional[float] = Field(None, ge=10, le=500, description="LDL cholesterol (mg/dL)")
    HDL: Optional[float] = Field(None, ge=5, le=200, description="HDL cholesterol (mg/dL)")
    BUN: Optional[float] = Field(None, ge=1, le=200, description="Blood urea nitrogen (mg/dL)")
    ESR: Optional[float] = Field(None, ge=1, le=200, description="ESR (mm/h)")
    HB: Optional[float] = Field(None, ge=3, le=25, description="Hemoglobin (g/dL)")
    K: Optional[float] = Field(None, ge=1, le=10, description="Potassium (mEq/L)")
    Na: Optional[float] = Field(None, ge=100, le=180, description="Sodium (mEq/L)")
    WBC: Optional[float] = Field(None, ge=500, le=60000, description="White blood cells (cells/uL)")
    Lymph: Optional[float] = Field(None, ge=1, le=100, description="Lymphocytes (%)")
    Neut: Optional[float] = Field(None, ge=1, le=100, description="Neutrophils (%)")
    PLT: Optional[float] = Field(None, ge=10, le=1500, description="Platelets (x1000/uL)")
    # Echo
    EF_TTE: Optional[float] = Field(None, ge=5, le=90, description="Ejection fraction, transthoracic echo (%)")
    RegionRWMA: Optional[int] = Field(None, ge=0, le=4, description="Regional wall motion abnormality extent score (0 = none)")
    VHD: Optional[Literal["N", "mild", "Moderate", "Severe"]] = Field(None, description="Valvular heart disease")

    @model_validator(mode="after")
    def _consistency(self):
        chest_pain = [self.TypicalChestPain == 1, self.Atypical == "Y", self.Nonanginal == "Y"]
        if sum(chest_pain) > 1:
            raise ValueError("Choose at most one chest-pain type (typical, atypical or non-anginal).")
        if self.CurrentSmoker == 1 and self.EX_Smoker == 1:
            raise ValueError("A patient cannot be both a current smoker and an ex-smoker.")
        if self.Weight is not None and self.Length is not None:
            bmi = self.Weight / (self.Length / 100) ** 2
            if self.BMI is not None and abs(self.BMI - bmi) > 0.5:
                raise ValueError(f"BMI {self.BMI:.1f} does not match weight and height (computed {bmi:.1f}).")
            self.BMI = bmi
        if self.BMI is not None:
            derived = "Y" if self.BMI >= 25 else "N"
            if self.Obesity is not None and self.Obesity != derived:
                raise ValueError(f"Obesity must be {derived!r} for BMI {self.BMI:.1f} (defined as BMI >= 25).")
            self.Obesity = derived
        return self


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
    "LowTHAng": "LowTH Ang",
    "QWave": "Q Wave",
    "StElevation": "St Elevation",
    "StDepression": "St Depression",
    "PoorRProgression": "Poor R Progression",
    "EF_TTE": "EF-TTE",
    "RegionRWMA": "Region RWMA",
}
COLUMN_TO_SCHEMA = {v: k for k, v in SCHEMA_TO_COLUMN.items()}

# Dataset columns with no schema field: always set to their only observed value.
FIXED_COLUMNS = {"Exertional CP": "N"}

# The most informative inputs; the UI's quick mode asks only for these (others are imputed).
QUICK_FIELDS = [
    "Age", "Sex", "Weight", "Length", "TypicalChestPain", "Atypical", "Nonanginal",
    "DM", "HTN", "CurrentSmoker", "BP", "EF_TTE", "RegionRWMA", "Tinversion", "StDepression",
    "FBS", "TG", "LDL", "HDL", "CR", "ESR",
]


class VesselProbability(BaseModel):
    lad: float = Field(..., ge=0.0, le=1.0)
    lcx: float = Field(..., ge=0.0, le=1.0)
    rca: float = Field(..., ge=0.0, le=1.0)


class TargetResult(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., description="1 if probability >= threshold")
    threshold: float = Field(..., description="F1-optimal decision threshold learned in training")
    band: str = Field(..., description="Lower / Intermediate / Higher")
    band_observed_rate: Optional[float] = Field(
        None, description="Share of cross-validation patients in this band who had the condition")


class RangeWarning(BaseModel):
    field: str
    column: str
    value: float
    train_min: float
    train_max: float


class PredictionResponse(BaseModel):
    cad_probability: float = Field(..., ge=0.0, le=1.0)
    vessel_probabilities: VesselProbability
    cad_prediction: int
    vessel_predictions: VesselProbability
    results: dict[str, TargetResult]
    imputed: list[str] = Field(default_factory=list, description="Schema fields filled with cohort defaults")
    warnings: list[RangeWarning] = Field(default_factory=list)
    model_version: str
    safety_disclaimer: str


class ShapExplanation(BaseModel):
    base_value: float = Field(..., description="Model starting point (log-odds)")
    feature_names: list[str]
    shap_values: list[float] = Field(..., description="Exact additive contributions (log-odds)")
    feature_values: list[float] = Field(..., description="Input value (encoded for categories, raw units)")
    derived: list[bool] = Field(..., description="True for computed clinical scores")


class ExplainResponse(PredictionResponse):
    explanation: ShapExplanation  # overall CAD model (kept for compatibility)
    explanations: dict[str, ShapExplanation]  # per model: cad, lad, lcx, rca
