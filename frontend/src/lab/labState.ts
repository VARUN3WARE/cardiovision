/**
 * Ischaemia lab settings: which artery the sandbox narrowing sits on, how narrow it is and how hard the
 * heart is working. The lab starts from the patient's resting heart rate, systolic pressure and age (so
 * exercise is relative to them) but the narrowing is generic and chosen by the user: it is never this
 * patient's artery. Deep link: ?lab=lad|lcx|rca&narrowing=80&exercise=1 (opens the Atlas heart).
 */
import { VESSEL_IDS, type Vessel } from "../anatomy/registry";
import type { ExplainResponse } from "../api/client";
import { AGE_DEFAULT, MAX_NARROWING, type LabPatient, type LabState } from "../twin/coronaryPhysiology";
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

/** The patient's resting values the lab starts from (age as the model saw it; dataset median if absent). */
export function labPatient(result: ExplainResponse | null, twin: TwinParams): LabPatient {
  const exp = result?.explanations.cad;
  const i = exp ? exp.feature_names.indexOf("Age") : -1;
  const age = exp && i >= 0 ? exp.feature_values[i] : NaN;
  return { hr: twin.hr, sbp: twin.bp, age: Number.isFinite(age) && age > 0 ? age : AGE_DEFAULT };
}

/**
 * Where the lab's starting values come from, in words: the twin's (the patient's, or a what-if scenario's while
 * one is shown), with typical fill-in values said to be typical.
 */
export function baselineText(result: ExplainResponse | null, twin: TwinParams, p: LabPatient): string {
  const v = `pulse ${Math.round(p.hr)}, BP ${Math.round(p.sbp)}, age ${Math.round(p.age)}`;
  if (!result) return `Starts from typical values (no estimate yet): ${v}`;
  const typical = [twin.notRecorded.has("hr") && "pulse", twin.notRecorded.has("bp") && "BP"].filter(Boolean);
  return `Starts from the twin's ${v}${typical.length ? ` (${typical.join(" and ")} typical: not entered)` : ""}`;
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
