// Single source of truth for the clinical input form: labels, units, ranges, grouping.
// Keys are the backend schema field names (backend/app/schemas.py). Ranges mirror the schema.

export type FieldKind = "number" | "flag" | "yn" | "select";

export interface FieldSpec {
  key: string;
  label: string;
  kind: FieldKind;
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: [string, string][]; // [value, label] for selects
  hint?: string;
  required?: boolean;
  quick?: boolean; // asked in quick mode (the most informative inputs)
}

export interface Section {
  id: string;
  title: string;
  fields: FieldSpec[];
}

const num = (key: string, label: string, unit: string | undefined, min: number, max: number, step = 1,
  extra: Partial<FieldSpec> = {}): FieldSpec => ({ key, label, kind: "number", unit, min, max, step, ...extra });
const flag = (key: string, label: string, extra: Partial<FieldSpec> = {}): FieldSpec => ({ key, label, kind: "flag", ...extra });
const yn = (key: string, label: string, extra: Partial<FieldSpec> = {}): FieldSpec => ({ key, label, kind: "yn", ...extra });

// Chest pain and smoking are entered as one choice each (they are mutually exclusive in the data).
export const CHEST_PAIN_KEYS = ["TypicalChestPain", "Atypical", "Nonanginal"] as const;
export const SMOKING_KEYS = ["CurrentSmoker", "EX_Smoker"] as const;

export const SECTIONS: Section[] = [
  {
    id: "patient", title: "Patient",
    fields: [
      num("Age", "Age", "years", 18, 110, 1, { required: true, quick: true }),
      { key: "Sex", label: "Sex", kind: "select", options: [["Male", "Male"], ["Fmale", "Female"]], required: true, quick: true },
      num("Weight", "Weight", "kg", 20, 250, 0.1, { quick: true }),
      num("Length", "Height", "cm", 100, 220, 1, { quick: true }),
    ],
  },
  {
    id: "symptoms", title: "Symptoms",
    fields: [
      yn("LowTHAng", "Low-threshold angina"),
      yn("Dyspnea", "Dyspnea"),
      {
        key: "FunctionClass", label: "Functional class", kind: "select",
        options: [["0", "0 – not limited / not recorded"], ["1", "I"], ["2", "II"], ["3", "III"], ["4", "IV (not in training data)"]],
        hint: "The dataset's 'Function Class' (coded 0–3). It is not defined further; NYHA-style grading is assumed.",
      },
    ],
  },
  {
    id: "risk", title: "Risk factors",
    fields: [
      flag("DM", "Diabetes", { quick: true }),
      flag("HTN", "Hypertension", { quick: true }),
      flag("FH", "Family history of CAD"),
      yn("DLP", "Dyslipidemia"),
    ],
  },
  {
    id: "vitals", title: "Vitals & examination",
    fields: [
      num("BP", "Systolic BP", "mmHg", 60, 260, 1, {
        quick: true, hint: "The dataset records one blood-pressure value (90–190 mmHg, i.e. the systolic range)",
      }),
      num("PR", "Pulse rate", "bpm", 30, 220),
      flag("Edema", "Edema"),
      yn("WeakPeripheralPulse", "Weak peripheral pulse"),
      yn("LungRales", "Lung rales"),
      yn("SystolicMurmur", "Systolic murmur"),
      yn("DiastolicMurmur", "Diastolic murmur"),
    ],
  },
  {
    id: "echo", title: "Echocardiography",
    fields: [
      num("EF_TTE", "Ejection fraction", "%", 5, 90, 1, { quick: true }),
      {
        key: "RegionRWMA", label: "Regional wall-motion abnormality", kind: "select", quick: true,
        options: [["0", "0 – none"], ["1", "1"], ["2", "2"], ["3", "3"], ["4", "4 – most extensive"]],
        hint: "The dataset's 'Region RWMA' grade (0–4). Higher means more extensive (ejection fraction falls as it rises); where the abnormality is is not recorded.",
      },
      {
        key: "VHD", label: "Valvular heart disease", kind: "select",
        options: [["N", "None"], ["mild", "Mild"], ["Moderate", "Moderate"], ["Severe", "Severe"]],
      },
    ],
  },
  {
    id: "ecg", title: "ECG",
    fields: [
      flag("Tinversion", "T-wave inversion", { quick: true }),
      flag("StDepression", "ST depression", { quick: true }),
      flag("StElevation", "ST elevation"),
      flag("QWave", "Pathological Q wave"),
      yn("LVH", "LV hypertrophy"),
      yn("PoorRProgression", "Poor R-wave progression"),
      {
        key: "BBB", label: "Bundle branch block", kind: "select",
        options: [["N", "None"], ["LBBB", "LBBB"], ["RBBB", "RBBB"]],
      },
    ],
  },
  {
    id: "labs", title: "Laboratory",
    fields: [
      num("FBS", "Fasting glucose", "mg/dL", 40, 600, 1, { quick: true }),
      num("TG", "Triglycerides", "mg/dL", 20, 2000, 1, { quick: true }),
      num("LDL", "LDL cholesterol", "mg/dL", 10, 500, 1, { quick: true }),
      num("HDL", "HDL cholesterol", "mg/dL", 5, 200, 0.1, { quick: true }),
      num("CR", "Creatinine", "mg/dL", 0.1, 20, 0.01, { quick: true }),
      num("ESR", "ESR", "mm/h", 1, 200, 1, { quick: true }),
      num("BUN", "Urea nitrogen (BUN)", "mg/dL", 1, 200),
      num("HB", "Hemoglobin", "g/dL", 3, 25, 0.1),
      num("K", "Potassium", "mEq/L", 1, 10, 0.1),
      num("Na", "Sodium", "mEq/L", 100, 180),
      num("WBC", "White cells", "/µL", 500, 60000, 100),
      num("Lymph", "Lymphocytes", "%", 1, 100),
      num("Neut", "Neutrophils", "%", 1, 100),
      num("PLT", "Platelets", "×10³/µL", 10, 1500),
    ],
  },
  {
    id: "history", title: "Other history",
    fields: [
      yn("CRF", "Chronic renal failure"),
      yn("CVA", "Prior stroke (CVA)"),
      yn("CHF", "Heart failure (CHF)"),
      yn("AirwayDisease", "Airway disease"),
      yn("ThyroidDisease", "Thyroid disease"),
    ],
  },
];

export const ALL_FIELDS: FieldSpec[] = SECTIONS.flatMap((s) => s.fields);
export const FIELD_BY_KEY: Record<string, FieldSpec> = Object.fromEntries(ALL_FIELDS.map((f) => [f.key, f]));

/** Every schema key the form controls (including the composite chest-pain / smoking keys). */
export const FORM_KEYS: string[] = [...ALL_FIELDS.map((f) => f.key), ...CHEST_PAIN_KEYS, ...SMOKING_KEYS];

/** Keys sent in quick mode (others are filled with cohort defaults by the API). */
export const QUICK_KEYS = new Set<string>([
  ...ALL_FIELDS.filter((f) => f.quick).map((f) => f.key),
  ...CHEST_PAIN_KEYS,
  "CurrentSmoker",
]);

export const FIELD_LABEL: Record<string, string> = {
  ...Object.fromEntries(ALL_FIELDS.map((f) => [f.key, f.label])),
  TypicalChestPain: "Typical angina",
  Atypical: "Atypical chest pain",
  Nonanginal: "Non-anginal chest pain",
  CurrentSmoker: "Current smoker",
  EX_Smoker: "Ex-smoker",
  BMI: "BMI",
  Obesity: "BMI ≥ 25 (dataset 'Obesity' flag)",
};

export type FormValues = Record<string, string>; // "" = not entered

export const BLANK: FormValues = Object.fromEntries(FORM_KEYS.map((k) => [k, ""]));

/** Synthetic demonstration patient (not from the dataset). */
export const DEMO_PATIENT: FormValues = {
  ...BLANK,
  Age: "60", Sex: "Male", Weight: "78", Length: "170",
  TypicalChestPain: "1", Atypical: "N", Nonanginal: "N", LowTHAng: "Y", Dyspnea: "Y", FunctionClass: "2",
  DM: "1", HTN: "1", CurrentSmoker: "1", EX_Smoker: "0", FH: "1", DLP: "Y",
  BP: "145", PR: "80", Edema: "0", WeakPeripheralPulse: "N", LungRales: "N", SystolicMurmur: "N", DiastolicMurmur: "N",
  EF_TTE: "50", RegionRWMA: "1", VHD: "N",
  Tinversion: "1", StDepression: "1", StElevation: "1", QWave: "0", LVH: "N", PoorRProgression: "N", BBB: "N",
  FBS: "120", TG: "180", LDL: "150", HDL: "38", CR: "1.1", ESR: "12", BUN: "16", HB: "13.5", K: "4.2",
  Na: "140", WBC: "7500", Lymph: "30", Neut: "65", PLT: "250",
  CRF: "N", CVA: "N", CHF: "N", AirwayDisease: "N", ThyroidDisease: "N",
};

/** Dataset-row features (numbers / strings) -> form values. */
export function fromFeatures(features: Record<string, unknown>): FormValues {
  const out: FormValues = { ...BLANK };
  for (const k of FORM_KEYS) {
    const v = features[k];
    if (v !== undefined && v !== null) out[k] = String(v);
  }
  return out;
}

export function bmiOf(v: FormValues): number | null {
  const w = parseFloat(v.Weight);
  const h = parseFloat(v.Length);
  if (!isFinite(w) || !isFinite(h) || h <= 0) return null;
  return w / (h / 100) ** 2;
}

/** Client-side validation; returns field -> message. */
export function validate(v: FormValues, quick: boolean): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const f of ALL_FIELDS) {
    if (quick && !QUICK_KEYS.has(f.key)) continue;
    const raw = v[f.key]?.trim() ?? "";
    if (raw === "") {
      if (f.required) errors[f.key] = "Required";
      continue;
    }
    if (f.kind === "number") {
      const n = Number(raw);
      if (!isFinite(n)) errors[f.key] = "Enter a number";
      else if (f.min !== undefined && n < f.min) errors[f.key] = `Must be ≥ ${f.min}${f.unit ? " " + f.unit : ""}`;
      else if (f.max !== undefined && n > f.max) errors[f.key] = `Must be ≤ ${f.max}${f.unit ? " " + f.unit : ""}`;
    }
  }
  return errors;
}

/** Form values -> API payload (only entered values; omitted ones are imputed by the API). */
export function toPayload(v: FormValues, quick: boolean): Record<string, string | number> {
  const out: Record<string, string | number> = {};
  for (const k of FORM_KEYS) {
    if (quick && !QUICK_KEYS.has(k)) continue;
    const raw = v[k]?.trim() ?? "";
    if (raw === "") continue;
    const spec = FIELD_BY_KEY[k];
    if (spec?.kind === "number") out[k] = Number(raw);
    else if (spec?.kind === "flag" || k === "TypicalChestPain" || k === "CurrentSmoker" || k === "EX_Smoker"
      || k === "FunctionClass" || k === "RegionRWMA") out[k] = parseInt(raw, 10);
    else out[k] = raw;
  }
  return out;
}
