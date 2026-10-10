/**
 * Function twin: the heart's rhythm, timing, pumping and an illustrative ECG, driven by this patient's
 * recorded values (pulse rate, ejection fraction, blood pressure, ECG findings). The anatomy stays a
 * reference model; nothing here is the patient's own tracing or imaging.
 *
 * One shared clock keeps electrical and mechanical events in their physiological order within each beat:
 * P wave -> atrial contraction -> QRS -> ventricular contraction (ejection) -> T wave -> relaxation.
 *
 * ECG: sum-of-Gaussians waves in the spirit of McSharry et al., IEEE Trans Biomed Eng 2003;50:289-94,
 * with normal intervals PR 0.16 s, QRS 0.09 s (0.14 s in bundle branch block) and QT from Bazett's
 * formula (QTc 0.40 s).
 */
import type { ExplainResponse } from "../api/client";

export interface TwinParams {
  hr: number; // beats per minute (pulse rate)
  ef: number; // ejection fraction, %
  bp: number; // systolic blood pressure, mmHg
  stDepression: boolean;
  stElevation: boolean;
  tInversion: boolean;
  qWave: boolean;
  lvh: boolean;
  poorR: boolean;
  bbb: "none" | "LBBB" | "RBBB";
  /** Which of the above were not entered (filled with typical values by the model). */
  notRecorded: Set<keyof Omit<TwinParams, "notRecorded" | "lab">>;
  /**
   * Set only while the ischaemia lab (a teaching sandbox) drives the twin: hr and bp are then the lab's
   * exercise values and stMv the lab's illustrative ST depression (mV, 0 = none).
   */
  lab?: { stMv: number };
}

export interface BeatTiming {
  rr: number; // s per beat
  pr: number; // P onset -> QRS onset
  qrs: number; // QRS duration
  qt: number; // QRS onset -> end of T
}

export const DEFAULT_TWIN: TwinParams = {
  hr: 72, ef: 55, bp: 125, stDepression: false, stElevation: false, tInversion: false, qWave: false,
  lvh: false, poorR: false, bbb: "none", notRecorded: new Set(),
};

// Dataset column -> (twin field, API schema key) for reading the model's inputs.
const SOURCES: [string, keyof Omit<TwinParams, "notRecorded" | "lab">, string][] = [
  ["EF-TTE", "ef", "EF_TTE"], ["BP", "bp", "BP"], ["PR", "hr", "PR"],
  ["St Depression", "stDepression", "StDepression"], ["St Elevation", "stElevation", "StElevation"],
  ["Tinversion", "tInversion", "Tinversion"], ["Q Wave", "qWave", "QWave"], ["LVH", "lvh", "LVH"],
  ["Poor R Progression", "poorR", "PoorRProgression"], ["BBB", "bbb", "BBB"],
];
const BBB_CODES: TwinParams["bbb"][] = ["LBBB", "none", "RBBB"]; // label-encoded alphabetically: LBBB, N, RBBB

/**
 * The patient's recorded values, as the model saw them (the cohort median fills a measurement that was not
 * entered, and a finding that was not entered counts as absent). `bpm` is the pulse before any estimate.
 */
export function twinFromResult(result: ExplainResponse | null, bpm: number): TwinParams {
  if (!result) return { ...DEFAULT_TWIN, hr: bpm, notRecorded: new Set() };
  const exp = result.explanations.cad;
  const value = (col: string) => {
    const i = exp.feature_names.indexOf(col);
    return i >= 0 ? exp.feature_values[i] : undefined;
  };
  const imputed = new Set(result.imputed);
  const out: TwinParams = { ...DEFAULT_TWIN, hr: bpm, notRecorded: new Set() };
  for (const [col, field, schemaKey] of SOURCES) {
    const v = value(col);
    if (v === undefined) continue;
    if (field === "bbb") out.bbb = BBB_CODES[Math.round(v)] ?? "none";
    else if (field === "ef" || field === "bp" || field === "hr") out[field] = v > 0 ? v : out[field];
    else out[field] = v >= 0.5;
    if (imputed.has(schemaKey)) out.notRecorded.add(field);
  }
  return out;
}

export function beatTiming(p: Pick<TwinParams, "hr" | "bbb">): BeatTiming {
  const rr = 60 / Math.max(30, Math.min(220, p.hr || 72));
  const qrs = p.bbb === "none" ? 0.09 : 0.14;
  // Bazett: QT = QTc * sqrt(RR); wide QRS lengthens QT by about the extra QRS width. Keep diastole > 0.
  const qt = Math.min(0.4 * Math.sqrt(rr) + (qrs - 0.09), rr - 0.2);
  return { rr, pr: 0.16, qrs, qt: Math.max(0.2, qt) };
}

// ---------------------------------------------------------------------------------------------------
// Shared clock (advanced once per frame inside the 3D canvas; read by the ECG monitor and the hearts)
// ---------------------------------------------------------------------------------------------------

export const TICK = 0.002; // s: the ECG monitor reads beat time at 500 Hz
const RING = 4096;

export const heartClock = {
  t: 0, // time within the current beat, s (0 = P-wave onset)
  ticks: new Float32Array(RING), // beat time at each 2 ms tick
  count: 0, // ticks written so far
  carry: 0,
};

export function advanceClock(dt: number, timing: BeatTiming, running: boolean) {
  if (!running) return;
  let left = Math.min(dt, 0.1) + heartClock.carry; // avoid a burst after a stalled frame
  while (left >= TICK) {
    heartClock.t += TICK;
    if (heartClock.t >= timing.rr) heartClock.t -= timing.rr;
    heartClock.ticks[heartClock.count % RING] = heartClock.t;
    heartClock.count++;
    left -= TICK;
  }
  heartClock.carry = left;
}

export function readTick(k: number) {
  return heartClock.ticks[k % RING];
}
export const RING_SIZE = RING;

// ---------------------------------------------------------------------------------------------------
// Mechanics
// ---------------------------------------------------------------------------------------------------

const gauss = (t: number, c: number, s: number) => Math.exp(-0.5 * ((t - c) / s) ** 2);

export interface Mechanics {
  atrial: number; // 0..1 atrial contraction
  ventricular: number; // 0..1 ventricular contraction (ejection)
  rock: number; // -1..1 apical rocking in left bundle branch block (septal early, lateral late)
  squeeze: number; // contraction amplitude from the ejection fraction (fraction of size)
  aortic: number; // arterial distension amplitude from blood pressure
}

/** Ventricular contraction 0..1 at beat time t: electromechanical delay ~0.04 s after QRS onset, relaxation by the end of T. */
export function ventricularContraction(t: number, { pr, qt }: BeatTiming): number {
  const t0 = pr + 0.04;
  const t1 = pr + qt + 0.02;
  const s = (t - t0) / (t1 - t0);
  return s > 0 && s < 1 ? Math.sin(Math.PI * s) ** 1.4 : 0;
}

export function mechanics(t: number, timing: BeatTiming, p: TwinParams): Mechanics {
  const { pr } = timing;
  // Atrial systole follows the P wave by about 0.08 s.
  const atrial = gauss(t, 0.1, 0.035);
  const ventricular = ventricularContraction(t, timing);
  // LBBB: the septum contracts early, the lateral wall late (apical rocking on echo).
  const rock = p.bbb === "LBBB" ? -gauss(t, pr + 0.07, 0.03) + 0.9 * gauss(t, pr + 0.22, 0.06) : 0;
  const ef = Math.max(10, Math.min(75, p.ef || 55));
  return { atrial, ventricular, rock, squeeze: 0.055 * (ef / 60), aortic: 0.012 * Math.max(0.6, Math.min(1.6, (p.bp || 125) / 120)) };
}

/** Plain-language pump category (2021 ESC heart-failure ranges). */
export function efCategory(ef: number) {
  return ef <= 40 ? "reduced" : ef < 50 ? "mildly reduced" : "preserved";
}

// ---------------------------------------------------------------------------------------------------
// Illustrative ECG (mV), leads II and V1
// ---------------------------------------------------------------------------------------------------

export type Lead = "II" | "V1";

export function ecgSample(t: number, timing: BeatTiming, p: TwinParams, lead: Lead): number {
  const q0 = timing.pr; // QRS onset
  const w = timing.qrs / 0.09; // QRS widening factor
  const j = q0 + timing.qrs; // J point (end of QRS)
  const tc = q0 + timing.qt - 0.11; // T-wave peak
  const ts = 0.04 * Math.sqrt(timing.qt / 0.38); // T-wave width
  const g = (c: number, s: number, a: number) => a * gauss(t, c, s);
  // ST segment shift between the J point and the T wave (horizontal).
  const st = (a: number, tPeak = tc) => a * smooth(j - 0.004, j + 0.012, t) * (1 - smooth(tPeak - 0.05, tPeak + 0.01, t));
  let v = 0;

  if (lead === "II") {
    v += g(0.07, 0.022, 0.12); // P
    if (p.bbb === "LBBB") {
      v += g(q0 + 0.04, 0.018, 0.65) + g(q0 + 0.095, 0.018, 0.6); // broad, notched R
      v += g(tc, ts, -0.12); // secondary (discordant) T
    } else {
      const qAmp = p.qWave ? -0.35 : -0.06;
      const qWidth = p.qWave ? 0.012 : 0.006;
      v += g(q0 + 0.012, qWidth, qAmp);
      v += g(q0 + 0.035 * w, 0.009 * w, (p.lvh ? 1.7 : 1.0) * (p.qWave ? 0.7 : 1)); // R
      v += g(q0 + 0.06 * w, 0.009, -0.2); // S
      if (p.bbb === "RBBB") v += g(q0 + 0.1, 0.02, -0.28); // wide, slurred terminal S
      // Recorded ST depression (0.18 mV) or the ischaemia lab's (0.1-0.2 mV, while its narrowing causes
      // ischaemia): the deeper of the two is drawn.
      const stDepth = Math.max(p.stDepression ? 0.18 : 0, p.lab?.stMv ?? 0);
      // ST depression is read 60-80 ms after the J point (ACC/AHA 2002): keep that stretch ahead of the T wave
      // even at exercise rates, as long as the T wave still ends within the beat (no change at resting rates).
      const tcSt = stDepth > 0 ? Math.max(tc, Math.min(j + 0.08 + ts, timing.rr - 2 * ts - 0.01)) : tc;
      const tAmp = p.tInversion ? -0.25 : stDepth > 0 ? 0.15 : 0.28;
      v += g(tcSt, ts, tAmp);
      if (stDepth > 0) v += st(-stDepth, tcSt);
      if (p.stElevation) v += st(0.2);
    }
  } else {
    v += g(0.055, 0.018, 0.06) + g(0.1, 0.018, -0.05); // biphasic P
    if (p.bbb === "LBBB") {
      v += g(q0 + 0.02, 0.008, 0.03) + g(q0 + 0.075, 0.03, -1.25); // QS: deep, broad
      v += st(0.15) + g(tc, ts, 0.25); // discordant ST elevation, upright T
    } else if (p.bbb === "RBBB") {
      v += g(q0 + 0.02, 0.008, 0.3) + g(q0 + 0.05, 0.01, -0.45) + g(q0 + 0.1, 0.016, 0.85); // rSR'
      v += g(tc, ts, -0.2); // secondary T inversion
    } else {
      v += g(q0 + 0.025, 0.008, p.poorR ? 0.08 : 0.25); // small r
      v += g(q0 + 0.055, 0.012, p.lvh ? -2.0 : -0.9); // deep S (LVH: Sokolow-Lyon voltage)
      v += g(tc, ts, 0.06);
    }
  }
  return v;
}

function smooth(a: number, b: number, x: number) {
  const k = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return k * k * (3 - 2 * k);
}

/** Recorded ECG findings in words, for the monitor's annotations (plus the lab's, marked `lab`). */
export function ecgFindings(p: TwinParams): { text: string; recorded: boolean; lab?: boolean }[] {
  const rec = (k: keyof Omit<TwinParams, "notRecorded" | "lab">) => !p.notRecorded.has(k);
  const out: { text: string; recorded: boolean; lab?: boolean }[] = [];
  // The lab's ST depression is a whole-heart sign: it does not show where the ischaemia is. Not readable in LBBB.
  if (p.lab && p.lab.stMv > 0 && p.bbb !== "LBBB") out.push({ text: "ST depression", recorded: false, lab: true });
  if (p.stDepression) out.push({ text: "ST depression", recorded: rec("stDepression") });
  if (p.stElevation) out.push({ text: "ST elevation", recorded: rec("stElevation") });
  if (p.tInversion) out.push({ text: "T-wave inversion", recorded: rec("tInversion") });
  if (p.qWave) out.push({ text: "Pathological Q wave", recorded: rec("qWave") });
  if (p.lvh) out.push({ text: "LVH voltage", recorded: rec("lvh") });
  if (p.poorR) out.push({ text: "Poor R-wave progression", recorded: rec("poorR") });
  if (p.bbb !== "none") out.push({ text: p.bbb, recorded: rec("bbb") });
  return out;
}
