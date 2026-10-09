// What-if scenarios: model inputs changed on top of the patient's form, kept separate from the form.
// Pure logic (levers, contrasts, reducer, deep link, throttle) so it can be unit-tested.

import type { ExplainResponse, FeatureInfo, PredictionResponse, Target } from "../api/client";
import { MODEL_TABS, VESSEL } from "../anatomy/registry";

export type Payload = Record<string, string | number>;

/** A modifiable risk factor shown as a slider. `column` is the dataset column (feature_info / explanation key). */
export interface Lever {
  key: "BP" | "FBS" | "TG" | "HDL";
  column: string;
  label: string;
  unit: string;
  step: number;
  param: string; // deep-link query parameter
  note?: string;
}

export const LEVERS: readonly Lever[] = [
  { key: "BP", column: "BP", label: "Systolic BP", unit: "mmHg", step: 1, param: "bp" },
  { key: "FBS", column: "FBS", label: "Fasting glucose", unit: "mg/dL", step: 1, param: "fbs" },
  { key: "TG", column: "TG", label: "Triglycerides", unit: "mg/dL", step: 1, param: "tg" },
  {
    key: "HDL", column: "HDL", label: "HDL cholesterol", unit: "mg/dL", step: 1, param: "hdl",
    note: "Weak, non-monotonic association in this cohort.", // docs/model-verification.md §7
  },
];

/** Shown but not adjustable: the verification found no learned benefit (docs/model-verification.md §7). */
export const LOCKED: readonly { key: string; column: string; label: string; unit: string }[] = [
  { key: "LDL", column: "LDL", label: "LDL cholesterol", unit: "mg/dL" },
  { key: "Weight", column: "Weight", label: "Weight", unit: "kg" },
];
export const LOCKED_NOTE =
  "No learned benefit in this referral cohort (likely treatment/selection effects), so not adjustable.";

export type ContrastId = "angina" | "st" | "rwma" | "ef";

/** A diagnostic finding switched to its contrast value (absent/normal, or present when already absent). */
export interface Contrast {
  id: ContrastId;
  key: string; // schema key whose recorded value decides the direction
  column: string;
  /** Recorded value counts as abnormal (finding present). */
  abnormal: (v: number) => boolean;
  /** Payload patch when the finding is present (make it absent/normal) or absent (make it present). */
  toNormal: Payload;
  toAbnormal: Payload;
  normalText: string;
  abnormalText: string;
  /** How a recorded absent finding reads, when it differs from `normalText`. */
  absentText?: string;
  label: string;
}

// Contrast values follow the clinical-direction table in docs/model-verification.md §7
// (RWMA 2 vs 0, EF 35% vs 60%, typical angina vs no chest pain, ST depression vs none).
export const CONTRASTS: readonly Contrast[] = [
  {
    id: "angina", key: "TypicalChestPain", column: "Typical Chest Pain", label: "Chest pain",
    abnormal: (v) => v >= 0.5,
    toNormal: { TypicalChestPain: 0, Atypical: "N", Nonanginal: "N" },
    toAbnormal: { TypicalChestPain: 1, Atypical: "N", Nonanginal: "N" },
    normalText: "no chest pain", abnormalText: "typical angina", absentText: "not typical angina",
  },
  {
    id: "st", key: "StDepression", column: "St Depression", label: "ST depression",
    abnormal: (v) => v >= 0.5,
    toNormal: { StDepression: 0 }, toAbnormal: { StDepression: 1 },
    normalText: "absent", abnormalText: "present",
  },
  {
    id: "rwma", key: "RegionRWMA", column: "Region RWMA", label: "Regional wall-motion abnormality",
    abnormal: (v) => v > 0,
    toNormal: { RegionRWMA: 0 }, toAbnormal: { RegionRWMA: 2 },
    normalText: "grade 0", abnormalText: "grade 2",
  },
  {
    id: "ef", key: "EF_TTE", column: "EF-TTE", label: "Ejection fraction",
    abnormal: (v) => v < 50,
    toNormal: { EF_TTE: 60 }, toAbnormal: { EF_TTE: 35 },
    normalText: "60%", abnormalText: "35%",
  },
];

export const SMOKING = { key: "CurrentSmoker", column: "Current Smoker", param: "quit" } as const;
const QUIT_PATCH: Payload = { CurrentSmoker: 0, EX_Smoker: 1 };

// ---------------------------------------------------------------------------------------------------
// Scenario state
// ---------------------------------------------------------------------------------------------------

export interface Scenario {
  levers: Partial<Record<Lever["key"], number>>;
  quitSmoking: boolean;
  contrasts: Partial<Record<ContrastId, boolean>>;
}

export const EMPTY_SCENARIO: Scenario = { levers: {}, quitSmoking: false, contrasts: {} };

export type ScenarioAction =
  | { type: "lever"; key: Lever["key"]; value: number; baseline: number | undefined }
  | { type: "quit"; on: boolean }
  | { type: "contrast"; id: ContrastId; on: boolean }
  | { type: "reset" }
  | { type: "load"; scenario: Scenario };

export function scenarioReducer(s: Scenario, a: ScenarioAction): Scenario {
  switch (a.type) {
    case "lever": {
      const levers = { ...s.levers };
      // Back at the recorded value: no override (so the scenario can become inactive again).
      if (a.baseline !== undefined && a.value === a.baseline) delete levers[a.key];
      else levers[a.key] = a.value;
      return { ...s, levers };
    }
    case "quit":
      return { ...s, quitSmoking: a.on };
    case "contrast": {
      const contrasts = { ...s.contrasts };
      if (a.on) contrasts[a.id] = true;
      else delete contrasts[a.id];
      return { ...s, contrasts };
    }
    case "reset":
      return EMPTY_SCENARIO;
    case "load":
      return a.scenario;
  }
}

export function isActive(s: Scenario): boolean {
  return Object.keys(s.levers).length > 0 || s.quitSmoking || Object.values(s.contrasts).some(Boolean);
}

// ---------------------------------------------------------------------------------------------------
// Baselines (the value the model saw for this patient) and the scenario payload
// ---------------------------------------------------------------------------------------------------

/** The patient's value for `key`: the entered value, else the value the model used (cohort typical value). */
export function baselineOf(key: string, column: string, payload: Payload, result: ExplainResponse | null): number | undefined {
  const v = payload[key];
  if (typeof v === "number" && isFinite(v)) return v;
  const exp = result?.explanations.cad;
  if (!exp) return undefined;
  const i = exp.feature_names.indexOf(column);
  return i >= 0 ? exp.feature_values[i] : undefined;
}

/** Direction of a contrast for this patient: "normal" = switch a present finding off. */
export function contrastDirection(c: Contrast, baseline: number | undefined): "normal" | "abnormal" | null {
  if (baseline === undefined) return null;
  return c.abnormal(baseline) ? "normal" : "abnormal";
}

/**
 * Form payload + scenario -> request body. Returns null while a needed baseline is still unknown
 * (e.g. a contrast on a value the form left blank, before the first estimate arrives).
 */
export function applyScenario(payload: Payload, s: Scenario, result: ExplainResponse | null): Payload | null {
  const out: Payload = { ...payload };
  for (const [k, v] of Object.entries(s.levers)) if (v !== undefined) out[k] = v;
  if (s.quitSmoking) {
    if (baselineOf(SMOKING.key, SMOKING.column, payload, result) !== 1) return null;
    Object.assign(out, QUIT_PATCH);
  }
  for (const c of CONTRASTS) {
    if (!s.contrasts[c.id]) continue;
    const dir = contrastDirection(c, baselineOf(c.key, c.column, payload, result));
    if (!dir) return null;
    Object.assign(out, dir === "normal" ? c.toNormal : c.toAbnormal);
  }
  return out;
}

/** Slider range: cohort 5th-95th percentile, widened to include the patient's own value. */
export function sliderRange(info: FeatureInfo | undefined, baseline: number | undefined, step: number) {
  const p5 = info?.p5 ?? info?.min ?? baseline ?? 0;
  const p95 = info?.p95 ?? info?.max ?? baseline ?? 1;
  const snapDown = (v: number) => Math.floor(v / step) * step;
  const snapUp = (v: number) => Math.ceil(v / step) * step;
  const lo = snapDown(Math.min(p5, baseline ?? p5));
  const hi = snapUp(Math.max(p95, baseline ?? p95));
  return { lo, hi, p5, p95 };
}

export interface Delta {
  target: Target;
  label: string;
  from: number;
  to: number;
  fromBand: string;
  toBand: string;
}

/** Patient vs scenario, for the overall CAD model and each artery model. */
export function scenarioDeltas(base: PredictionResponse, scen: PredictionResponse): Delta[] {
  return MODEL_TABS.map((t) => ({
    target: t,
    label: t === "cad" ? "CAD" : VESSEL[t].label,
    from: base.results[t].probability,
    to: scen.results[t].probability,
    fromBand: base.results[t].band,
    toBand: scen.results[t].band,
  }));
}

// ---------------------------------------------------------------------------------------------------
// Deep link: ?whatif=1&bp=120&fbs=100&tg=150&hdl=45&quit=1&contrast=rwma,ef
// ---------------------------------------------------------------------------------------------------

export function scenarioFromQuery(search: string): { open: boolean; scenario: Scenario } {
  const q = new URLSearchParams(search);
  const open = q.get("whatif") === "1";
  if (!open) return { open, scenario: EMPTY_SCENARIO };
  const levers: Scenario["levers"] = {};
  for (const l of LEVERS) {
    const raw = q.get(l.param);
    const v = raw === null ? NaN : Number(raw);
    if (raw !== null && raw.trim() !== "" && isFinite(v)) levers[l.key] = v;
  }
  const contrasts: Scenario["contrasts"] = {};
  for (const id of (q.get("contrast") ?? "").split(",")) {
    if (CONTRASTS.some((c) => c.id === id)) contrasts[id as ContrastId] = true;
  }
  return { open, scenario: { levers, quitSmoking: q.get(SMOKING.param) === "1", contrasts } };
}

// ---------------------------------------------------------------------------------------------------
// Throttle (leading + trailing): at most one call per `ms`, always ending on the latest arguments
// ---------------------------------------------------------------------------------------------------

export interface Throttled<A extends unknown[]> {
  call: (...args: A) => void;
  /** Run a pending trailing call now. */
  flush: () => void;
  cancel: () => void;
}

export function throttle<A extends unknown[]>(fn: (...args: A) => void, ms: number): Throttled<A> {
  let last = -Infinity;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: A | null = null;
  const fire = () => {
    timer = null;
    if (!pending) return;
    const args = pending;
    pending = null;
    last = Date.now();
    fn(...args);
  };
  return {
    call: (...args: A) => {
      pending = args;
      const wait = last + ms - Date.now();
      if (wait <= 0 && !timer) fire();
      else if (!timer) timer = setTimeout(fire, Math.max(0, wait));
    },
    flush: () => {
      if (timer) clearTimeout(timer);
      fire();
    },
    cancel: () => {
      if (timer) clearTimeout(timer);
      timer = null;
      pending = null;
    },
  };
}
