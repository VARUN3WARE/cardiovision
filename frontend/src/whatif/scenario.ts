// What-if scenarios: model inputs changed on top of the patient's submitted values, kept separate from the form.
// Pure logic (levers, contrasts, reducer, normalisation, summaries, deep link, throttle and what every view shows)
// so it can be unit-tested; useScenario.ts wires it to the API.

import type { ExplainResponse, FeatureInfo, PredictionResponse, Target } from "../api/client";
import { MODEL_TABS, VESSEL } from "../anatomy/registry";
import { FIELD_BY_KEY } from "../fieldSpec";

export type Payload = Record<string, string | number>;

/** A modifiable risk factor shown as a slider. `column` is the dataset column (feature_info / explanation key). */
export interface Lever {
  key: "BP" | "FBS" | "TG";
  column: string;
  label: string;
  /** Lower-case name for scenario summaries. */
  short: string;
  unit: string;
  step: number;
  param: string; // deep-link query parameter
}

export const LEVERS: readonly Lever[] = [
  { key: "BP", column: "BP", label: "Systolic BP", short: "systolic BP", unit: "mmHg", step: 1, param: "bp" },
  { key: "FBS", column: "FBS", label: "Fasting glucose", short: "fasting glucose", unit: "mg/dL", step: 1, param: "fbs" },
  { key: "TG", column: "TG", label: "Triglycerides", short: "triglycerides", unit: "mg/dL", step: 1, param: "tg" },
];

/**
 * Shown but not adjustable: in this referral cohort the models learned no benefit from lowering them, so a slider
 * would point the wrong way. Each note quotes docs/model-verification.md §7 (backend/verify_models.py).
 */
export const LOCKED: readonly { key: string; column: string; label: string; unit: string; note: string }[] = [
  {
    key: "LDL", column: "LDL", label: "LDL cholesterol", unit: "mg/dL",
    note: "160 vs 90 mg/dL: −1.7 points. No signal in this cohort, most likely because patients with known disease were already on lipid-lowering treatment.",
  },
  {
    key: "HDL", column: "HDL", label: "HDL cholesterol", unit: "mg/dL",
    note: "30 vs 55 mg/dL: −0.2 points. No protective association learned in this referral cohort (CAD rates across HDL are weak and non-monotonic).",
  },
  {
    key: "Weight", column: "Weight", label: "Weight", unit: "kg",
    note: "94 vs 55 kg, BMI following: −3.8 points. Lighter patients more often had CAD in this referral cohort (76% at 65 kg or less, 68% above 80 kg).",
  },
];
export const LOCKED_NOTE =
  "Locked: the models learned no benefit from lowering these in this referral cohort (likely treatment and selection " +
  "effects). Mean change in the CAD estimate, from Model verification §7:";

export type ContrastId = "angina" | "st" | "rwma" | "ef";

/** One recorded finding flipped: present → absent/normal, or absent → present (a hypothetical finding). */
export interface Contrast {
  id: ContrastId;
  key: string; // schema key whose recorded value decides the direction
  column: string;
  label: string;
  /** Recorded value counts as abnormal (finding present). */
  abnormal: (v: number) => boolean;
  /** Payload patch when the finding is present (make it absent/normal) or absent (make it present). */
  toNormal: Payload;
  toAbnormal: Payload;
  normalText: string;
  abnormalText: string;
  /** Scenario summary, e.g. "wall-motion grade 0 → 2". */
  summary: (from: string, to: string) => string;
}

// Contrast values follow the clinical-direction table in docs/model-verification.md §7
// (RWMA 2 vs 0, EF 35% vs 60%, typical angina vs no chest pain, ST depression vs none).
export const CONTRASTS: readonly Contrast[] = [
  {
    id: "angina", key: "TypicalChestPain", column: "Typical Chest Pain", label: "Chest pain",
    abnormal: (v) => v >= 0.5,
    toNormal: { TypicalChestPain: 0, Atypical: "N", Nonanginal: "N" },
    toAbnormal: { TypicalChestPain: 1, Atypical: "N", Nonanginal: "N" },
    normalText: "no chest pain", abnormalText: "typical angina",
    summary: (f, t) => `${f} → ${t}`,
  },
  {
    id: "st", key: "StDepression", column: "St Depression", label: "ST depression",
    abnormal: (v) => v >= 0.5,
    toNormal: { StDepression: 0 }, toAbnormal: { StDepression: 1 },
    normalText: "absent", abnormalText: "present",
    summary: (f, t) => `ST depression ${f} → ${t}`,
  },
  {
    id: "rwma", key: "RegionRWMA", column: "Region RWMA", label: "Regional wall-motion abnormality",
    abnormal: (v) => v > 0,
    toNormal: { RegionRWMA: 0 }, toAbnormal: { RegionRWMA: 2 },
    normalText: "grade 0", abnormalText: "grade 2",
    summary: (f, t) => `wall-motion ${f} → ${t.replace(/^grade /, "")}`,
  },
  {
    id: "ef", key: "EF_TTE", column: "EF-TTE", label: "Ejection fraction",
    abnormal: (v) => v < 50,
    toNormal: { EF_TTE: 60 }, toAbnormal: { EF_TTE: 35 },
    normalText: "60%", abnormalText: "35%",
    summary: (f, t) => `EF ${f} → ${t}`,
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
// Baselines (the value the model saw for this patient), normalisation and the request body
// ---------------------------------------------------------------------------------------------------

/** The patient's value for `key`: the submitted value, else the value the model used (cohort typical value). */
export function baselineOf(key: string, column: string, payload: Payload, result: ExplainResponse | null): number | undefined {
  const v = payload[key];
  if (typeof v === "number" && isFinite(v)) return v;
  const exp = result?.explanations.cad;
  if (!exp) return undefined;
  const i = exp.feature_names.indexOf(column);
  return i >= 0 ? exp.feature_values[i] : undefined;
}

/** A yes/no input as 1/0: the submitted "Y"/"N" (or 0/1), else the value the model used. */
function flagOf(key: string, column: string, payload: Payload, result: ExplainResponse | null): number | undefined {
  if (payload[key] === "Y") return 1;
  if (payload[key] === "N") return 0;
  return baselineOf(key, column, payload, result);
}

/** Direction of a contrast for this patient: "normal" = switch a present finding off. */
export function contrastDirection(c: Contrast, baseline: number | undefined): "normal" | "abnormal" | null {
  if (baseline === undefined) return null;
  return c.abnormal(baseline) ? "normal" : "abnormal";
}

/** A contrast for this patient: which way it flips and the recorded and contrast values in words. */
export function contrastView(c: Contrast, payload: Payload, result: ExplainResponse | null) {
  const v = baselineOf(c.key, c.column, payload, result);
  const dir = contrastDirection(c, v);
  let from = "not known yet";
  if (v !== undefined) {
    if (c.id === "angina") {
      from = v >= 0.5 ? "typical angina"
        : flagOf("Atypical", "Atypical", payload, result) === 1 ? "atypical chest pain"
        : flagOf("Nonanginal", "Nonanginal", payload, result) === 1 ? "non-anginal chest pain"
        : "no chest pain recorded";
    } else if (c.id === "rwma") from = `grade ${round1(v)}`;
    else if (c.id === "ef") from = `${round1(v)}%`;
    else from = dir === "normal" ? c.abnormalText : c.normalText;
  }
  return { dir, from, to: dir === "normal" ? c.normalText : c.abnormalText };
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

/**
 * Drops what no longer applies to the patient run and keeps lever values usable: quitting smoking only for a
 * current smoker; levers snapped and clamped to the slider range (the form's validation range until the cohort
 * ranges load), and dropped when back at the patient's value. Returns `s` itself when nothing changes; leaves it
 * alone while the patient's values are not known yet.
 */
export function normaliseScenario(
  s: Scenario, payload: Payload | null, result: ExplainResponse | null, info: Record<string, FeatureInfo> | null,
): Scenario {
  if (!payload || !result) return s;
  let levers = s.levers;
  for (const l of LEVERS) {
    const v = s.levers[l.key];
    if (v === undefined) continue;
    const baseline = baselineOf(l.key, l.column, payload, result);
    const field = FIELD_BY_KEY[l.key];
    const { lo, hi } = info ? sliderRange(info[l.column], baseline, l.step) : { lo: field?.min ?? -Infinity, hi: field?.max ?? Infinity };
    const c = Math.min(hi, Math.max(lo, Math.round(v / l.step) * l.step));
    if (c === v && c !== baseline) continue;
    if (levers === s.levers) levers = { ...s.levers };
    if (c === baseline) delete levers[l.key];
    else levers[l.key] = c;
  }
  const quitSmoking = s.quitSmoking && baselineOf(SMOKING.key, SMOKING.column, payload, result) === 1;
  return levers === s.levers && quitSmoking === s.quitSmoking ? s : { ...s, levers, quitSmoking };
}

/**
 * Submitted values + scenario -> request body. Returns null while a needed baseline is still unknown
 * (a contrast on a value the form left blank, before the first estimate arrives).
 */
export function applyScenario(payload: Payload, s: Scenario, result: ExplainResponse | null): Payload | null {
  const out: Payload = { ...payload };
  for (const [k, v] of Object.entries(s.levers)) if (v !== undefined) out[k] = v;
  if (s.quitSmoking) {
    const smoker = baselineOf(SMOKING.key, SMOKING.column, payload, result);
    if (smoker === undefined) return null;
    if (smoker === 1) Object.assign(out, QUIT_PATCH); // nothing to quit otherwise (normaliseScenario drops it)
  }
  for (const c of CONTRASTS) {
    if (!s.contrasts[c.id]) continue;
    const dir = contrastDirection(c, baselineOf(c.key, c.column, payload, result));
    if (!dir) return null;
    Object.assign(out, dir === "normal" ? c.toNormal : c.toAbnormal);
  }
  return out;
}

/** The patient run a scenario builds on: the request body that produced `result` (never the live form). */
export interface ScenarioBase {
  body: Payload;
  key: string;
  result: ExplainResponse;
}

export interface ScenarioRequest {
  body: Payload;
  key: string; // JSON of the body
  baseKey: string; // the patient run it builds on
}

/** The request for a scenario on the submitted patient run (null: no run yet, or nothing changed). */
export function scenarioRequest(base: ScenarioBase | null, s: Scenario): ScenarioRequest | null {
  if (!base || !isActive(s)) return null;
  const body = applyScenario(base.body, s, base.result);
  return body ? { body, key: JSON.stringify(body), baseKey: base.key } : null;
}

/** What a scenario changes, in words, e.g. "wall-motion grade 0 → 2 (hypothetical)". */
export function describeScenario(s: Scenario, payload: Payload, result: ExplainResponse | null): string[] {
  const out: string[] = [];
  const imputed = new Set(result?.imputed ?? []);
  for (const l of LEVERS) {
    const v = s.levers[l.key];
    if (v === undefined) continue;
    const b = baselineOf(l.key, l.column, payload, result);
    const from = b === undefined ? "" : `${imputed.has(l.key) ? "typical " : ""}${round1(b)} → `;
    out.push(`${l.short} ${from}${round1(v)} ${l.unit}`);
  }
  if (s.quitSmoking) out.push("current smoker → ex-smoker");
  for (const c of CONTRASTS) {
    if (!s.contrasts[c.id]) continue;
    const { dir, from, to } = contrastView(c, payload, result);
    if (dir) out.push(c.summary(from, to) + (dir === "abnormal" ? " (hypothetical)" : ""));
  }
  return out;
}

const round1 = (v: number) => Math.round(v * 10) / 10;

// ---------------------------------------------------------------------------------------------------
// What every view shows
// ---------------------------------------------------------------------------------------------------

export interface ScenarioReply {
  key: string; // request body it belongs to
  baseKey: string; // the patient run it was built on
  scenario: Scenario; // what it was calculated for (labels describe what is on screen)
  data: ExplainResponse; // during a drag: fresh probabilities over the last explanations
  explained: boolean; // explanations match the probabilities
}

export interface ScenarioError {
  key: string;
  kind: "predict" | "explain";
  message: string;
}

export type ScenarioStatus = "off" | "calculating" | "ready" | "updating" | "error";

export interface ScenarioView {
  status: ScenarioStatus;
  /** What every view shows: the scenario's reply, else the patient's own result. */
  shown: ExplainResponse | null;
  /** The reply on screen (null: the patient's own result is shown, with no what-if labels). */
  reply: ScenarioReply | null;
  /** The explanations on screen are older than the probabilities (during a drag). */
  explanationsLag: boolean;
}

/**
 * Decides what the dashboard, 3D view and explanations show. The hook keeps a request for `req` scheduled or in
 * flight until a reply or an error for it arrives, so an older reply on the same patient run may stand in meanwhile
 * (mid-drag, between a release and its explanation); it is never kept after the request failed, never shown over
 * another patient run, and nothing is labelled what-if before the first reply ("calculating") or after a failure.
 */
export function scenarioView(
  req: ScenarioRequest | null, reply: ScenarioReply | null, error: ScenarioError | null, base: ExplainResponse | null,
): ScenarioView {
  const patient = { shown: base, reply: null, explanationsLag: false };
  if (!req) return { status: "off", ...patient };
  if (error && error.key === req.key) return { status: "error", ...patient };
  if (!reply || reply.baseKey !== req.baseKey) return { status: "calculating", ...patient };
  const current = reply.key === req.key && reply.explained;
  return { status: current ? "ready" : "updating", shown: reply.data, reply, explanationsLag: !reply.explained };
}

// ---------------------------------------------------------------------------------------------------
// Deltas
// ---------------------------------------------------------------------------------------------------

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

/** Change in whole points between the percentages as shown (both ends clamped to 5-95%, like formatPct). */
export function deltaPts(from: number, to: number): number {
  const shown = (p: number) => Math.round(Math.min(0.95, Math.max(0.05, p)) * 100);
  return shown(to) - shown(from);
}

// ---------------------------------------------------------------------------------------------------
// Deep link: ?whatif=1&bp=120&fbs=100&tg=150&quit=1&contrast=angina,st (levers clamped once the patient loads)
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

// ---------------------------------------------------------------------------------------------------
// One request at a time: a newer job waits for the running one (only the newest waiting job is kept)
// ---------------------------------------------------------------------------------------------------

export interface QueueJob {
  key: string;
  kind: "predict" | "explain";
}

/**
 * Single-flight queue: never more than one request in flight and none aborted mid-drag (so a slow API cannot
 * starve a drag: every reply shows progress). `run(job, done)` must call `done()` once the request settles.
 */
export function singleFlight<J extends QueueJob>(run: (job: J, done: () => void) => void) {
  let running: J | null = null;
  let waiting: J | null = null;
  const start = (job: J) => {
    running = job;
    let finished = false;
    run(job, () => {
      if (finished || running !== job) return;
      finished = true;
      running = null;
      const next = waiting;
      waiting = null;
      if (next) start(next);
    });
  };
  return {
    submit(job: J) {
      if (!running) return start(job);
      // The running request already answers this job (an explanation also answers a fast estimate).
      if (running.key === job.key && (running.kind === "explain" || running.kind === job.kind)) {
        waiting = null;
        return;
      }
      waiting = job;
    },
    /** Forget the running and waiting jobs (the caller aborts the running request). */
    clear() {
      running = null;
      waiting = null;
    },
    get running() {
      return running;
    },
    get waiting() {
      return waiting;
    },
  };
}
