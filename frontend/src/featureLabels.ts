// Human-readable labels and units for the dataset's feature columns.
export const FEATURE_LABELS: Record<string, { label: string; unit?: string }> = {
  Age: { label: "Age", unit: "yr" },
  Weight: { label: "Weight", unit: "kg" },
  Length: { label: "Height", unit: "cm" },
  Sex: { label: "Sex" },
  BMI: { label: "Body mass index", unit: "kg/m²" },
  DM: { label: "Diabetes mellitus" },
  HTN: { label: "Hypertension" },
  "Current Smoker": { label: "Current smoker" },
  "EX-Smoker": { label: "Ex-smoker" },
  FH: { label: "Family history of CAD" },
  Obesity: { label: "Obesity" },
  CRF: { label: "Chronic renal failure" },
  CVA: { label: "Prior stroke (CVA)" },
  "Airway disease": { label: "Airway disease" },
  "Thyroid Disease": { label: "Thyroid disease" },
  CHF: { label: "Congestive heart failure" },
  DLP: { label: "Dyslipidemia" },
  BP: { label: "Systolic blood pressure", unit: "mmHg" },
  PR: { label: "Pulse rate", unit: "bpm" },
  Edema: { label: "Edema" },
  "Weak Peripheral Pulse": { label: "Weak peripheral pulse" },
  "Lung rales": { label: "Lung rales" },
  "Systolic Murmur": { label: "Systolic murmur" },
  "Diastolic Murmur": { label: "Diastolic murmur" },
  "Typical Chest Pain": { label: "Typical chest pain (angina)" },
  Dyspnea: { label: "Dyspnea" },
  "Function Class": { label: "NYHA class" },
  Atypical: { label: "Atypical chest pain" },
  Nonanginal: { label: "Non-anginal chest pain" },
  "Exertional CP": { label: "Exertional chest pain" },
  "LowTH Ang": { label: "Low-threshold angina" },
  "Q Wave": { label: "ECG: pathological Q wave" },
  "St Elevation": { label: "ECG: ST elevation" },
  "St Depression": { label: "ECG: ST depression" },
  Tinversion: { label: "ECG: T-wave inversion" },
  LVH: { label: "ECG: LV hypertrophy" },
  "Poor R Progression": { label: "ECG: poor R-wave progression" },
  BBB: { label: "Bundle branch block" },
  FBS: { label: "Fasting blood sugar", unit: "mg/dL" },
  CR: { label: "Creatinine", unit: "mg/dL" },
  TG: { label: "Triglycerides", unit: "mg/dL" },
  LDL: { label: "LDL cholesterol", unit: "mg/dL" },
  HDL: { label: "HDL cholesterol", unit: "mg/dL" },
  BUN: { label: "Blood urea nitrogen", unit: "mg/dL" },
  ESR: { label: "ESR", unit: "mm/h" },
  HB: { label: "Hemoglobin", unit: "g/dL" },
  K: { label: "Potassium", unit: "mEq/L" },
  Na: { label: "Sodium", unit: "mEq/L" },
  WBC: { label: "White blood cells", unit: "/µL" },
  Lymph: { label: "Lymphocytes", unit: "%" },
  Neut: { label: "Neutrophils", unit: "%" },
  PLT: { label: "Platelets", unit: "×10³/µL" },
  "EF-TTE": { label: "Ejection fraction (echo)", unit: "%" },
  "Region RWMA": { label: "Wall-motion abnormality score" },
  VHD: { label: "Valvular heart disease" },
  // Derived clinical scores computed by the model from the inputs above.
  "DF pretest prob": { label: "Pre-test probability (Diamond-Forrester)" },
  "Risk factor count": { label: "Number of cardiac risk factors" },
  "ECG abnormality count": { label: "Number of ECG abnormalities" },
  "Chest pain class": { label: "Chest pain type" },
  "RWMA present": { label: "Any wall-motion abnormality" },
  "TG/HDL": { label: "Triglyceride / HDL ratio" },
  "LDL/HDL": { label: "LDL / HDL ratio" },
  "AIP log(TG/HDL)": { label: "Atherogenic index of plasma" },
  "Age x Male": { label: "Age (male)" },
  eGFR: { label: "Kidney function (eGFR)", unit: "mL/min/1.73m²" },
  NLR: { label: "Neutrophil / lymphocyte ratio" },
  "EF below 50": { label: "Reduced ejection fraction (<50%)" },
  "FBS 126+": { label: "Diabetic-range glucose (≥126)" },
  "Systolic BP 140+": { label: "Systolic BP ≥ 140" },
};

const CHEST_PAIN_CLASS = ["None", "Non-anginal", "Atypical", "Typical angina"];
const DERIVED_FLAGS = new Set(["RWMA present", "EF below 50", "FBS 126+", "Systolic BP 140+"]);

// Encoded categorical values -> display text (codes are sorted category order from train.py).
const CATEGORY_DISPLAY: Record<string, string[]> = {
  Sex: ["Female", "Male"],
  BBB: ["LBBB", "None", "RBBB"],
  VHD: ["Moderate", "None", "Severe", "Mild"],
};

// 0/1 flags and Y/N columns (Y/N encodes as N=0, Y=1).
const BINARY = new Set([
  "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Edema", "Typical Chest Pain",
  "Q Wave", "St Elevation", "St Depression", "Tinversion",
  "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP",
  "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
  "Dyspnea", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "LVH", "Poor R Progression",
]);

export function formatFeatureValue(name: string, value: number): string {
  if (CATEGORY_DISPLAY[name]) return CATEGORY_DISPLAY[name][value] ?? String(value);
  if (BINARY.has(name) || DERIVED_FLAGS.has(name)) return value >= 0.5 ? "Yes" : "No";
  if (name === "Region RWMA") return value === 0 ? "None" : `Score ${value}`;
  if (name === "DF pretest prob") return `${Math.round(value * 100)}%`;
  if (name === "Chest pain class") return CHEST_PAIN_CLASS[value] ?? String(value);
  if (name === "Age x Male") return value === 0 ? "Female" : `${value} yr`;
  if (name === "Risk factor count" || name === "ECG abnormality count") return String(Math.round(value));
  if (name === "AIP log(TG/HDL)") return value.toFixed(2);
  if (name === "TG/HDL" || name === "LDL/HDL" || name === "NLR") return value.toFixed(1);
  if (name === "eGFR") return `${Math.round(value)}`;
  const unit = FEATURE_LABELS[name]?.unit;
  const v = Number.isInteger(value) ? String(value) : value.toFixed(1);
  return unit ? `${v} ${unit}` : v;
}
