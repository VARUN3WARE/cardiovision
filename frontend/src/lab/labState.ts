/**
 * Ischaemia lab settings: which artery the sandbox narrowing sits on, how narrow it is and how hard the
 * heart is working. The lab starts from the patient's recorded resting heart rate, systolic pressure and age
 * (so exercise is relative to them) but the narrowing is generic and chosen by the user: it is never this
 * patient's artery. Deep link: ?lab=lad|lcx|rca&narrowing=80&exercise=1 (opens the Atlas heart).
 */
import { VESSEL_IDS, type Vessel } from "../anatomy/registry";
import type { ExplainResponse } from "../api/client";
import {
  AGE_DEFAULT, HR_DEFAULT, MAX_NARROWING, SBP_DEFAULT, type LabPatient, type LabState,
} from "../twin/coronaryPhysiology";
import type { TwinParams } from "../twin/twin";

export interface LabSettings {
  vessel: Vessel;
  narrowing: number; // diameter stenosis, % (0..95)
  exercise: number; // 0 rest .. 1 peak
}

/** Opening state: a severe narrowing at rest, so raising the exercise level shows the cascade appear. */
export const LAB_DEFAULT: LabSettings = { vessel: VESSEL_IDS[0], narrowing: 80, exercise: 0 };

/** Lab settings from the URL, or null when the lab was not asked for. */
export function labFromQuery(search: string): LabSettings | null {
  const q = new URLSearchParams(search);
  const lab = q.get("lab");
  if (lab === null || lab === "0" || lab === "off") return null;
  const vessel = (VESSEL_IDS as readonly string[]).includes(lab) ? (lab as Vessel) : LAB_DEFAULT.vessel;
  const num = (key: string, lo: number, hi: number, fallback: number) => {
    const raw = q.get(key);
    const v = raw === null || raw.trim() === "" ? NaN : Number(raw);
    return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : fallback;
  };
  return {
    vessel,
    narrowing: Math.round(num("narrowing", 0, MAX_NARROWING * 100, LAB_DEFAULT.narrowing)),
    exercise: num("exercise", 0, 1, LAB_DEFAULT.exercise),
  };
}

// The model's columns for the lab's starting values (the API's schema keys are the same names).
const STARTS = [["PR", "pulse"], ["BP", "BP"], ["Age", "age"]] as const;

/**
 * The patient's recorded resting pulse, systolic pressure and age as the model saw them (inputs left empty
 * carry the cohort median the model used); the dataset medians before there is any estimate.
 */
export function labPatient(recorded: ExplainResponse | null): LabPatient {
  const exp = recorded?.explanations.cad;
  const value = (col: string, fallback: number) => {
    const i = exp ? exp.feature_names.indexOf(col) : -1;
    const v = exp && i >= 0 ? exp.feature_values[i] : NaN;
    return Number.isFinite(v) && v > 0 ? v : fallback;
  };
  return { hr: value("PR", HR_DEFAULT), sbp: value("BP", SBP_DEFAULT), age: value("Age", AGE_DEFAULT) };
}

/**
 * Where the lab's starting values come from, in words: the patient's recorded values (never a what-if
 * scenario's), with inputs that were not entered named, or the dataset medians before any estimate.
 */
export function baselineText(recorded: ExplainResponse | null, p: LabPatient, whatIf = false): string {
  const v = `pulse ${Math.round(p.hr)}, BP ${Math.round(p.sbp)}, age ${Math.round(p.age)}`;
  if (!recorded) return `Starts from the dataset medians (no estimate yet): ${v}`;
  const imputed = new Set(recorded.imputed);
  const missing = STARTS.filter(([col]) => imputed.has(col)).map(([, name]) => name);
  const names = missing.length > 1 ? `${missing.slice(0, -1).join(", ")} and ${missing[missing.length - 1]}` : missing[0];
  const note = missing.length ? ` (${names} not entered: cohort median)` : "";
  return `Starts from the patient's recorded ${v}${note}${whatIf ? ", not the what-if scenario's" : ""}`;
}

/**
 * The twin while the lab runs: the exercise heart rate and systolic pressure, and the lab's ST depression
 * on the monitor. Everything else (EF, recorded ECG findings) stays the patient's; switching the lab off
 * returns the twin to the patient's own values.
 */
export function labTwin(twin: TwinParams, s: LabState): TwinParams {
  return { ...twin, hr: s.hr, bp: s.sbp, lab: { stMv: s.stMv } };
}

/** Plain-language exercise level. */
export function exerciseLabel(x: number) {
  return x <= 0.02 ? "rest" : x >= 0.98 ? "peak" : x < 0.5 ? "light" : "moderate";
}
