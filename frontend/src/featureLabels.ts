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
  BP: { label: "Blood pressure", unit: "mmHg" },
  PR: { label: "Pulse rate", unit: "bpm" },
  Edema: { label: "Edema" },
  "Weak Peripheral Pulse": { label: "Weak peripheral pulse" },
  "Lung rales": { label: "Lung rales" },
  "Systolic Murmur": { label: "Systolic murmur" },
  "Diastolic Murmur": { label: "Diastolic murmur" },
  "Typical Chest Pain": { label: "Typical chest pain (angina)" },
  Dyspnea: { label: "Dyspnea" },
  "Function Class": { label: "NYHA function class" },
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
  "Region RWMA": { label: "Regional wall motion abnormality" },
  VHD: { label: "Valvular heart disease" },
};

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
  if (BINARY.has(name)) return value >= 0.5 ? "Yes" : "No";
  if (name === "Region RWMA") return value === 0 ? "None" : `Region ${value}`;
  const unit = FEATURE_LABELS[name]?.unit;
  const v = Number.isInteger(value) ? String(value) : value.toFixed(1);
  return unit ? `${v} ${unit}` : v;
}
