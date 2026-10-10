/**
 * Ischaemia lab: textbook coronary physiology for ONE generic narrowing chosen by the user. A teaching
 * sandbox, never this patient's arteries (we have no imaging). Pure functions, unit-tested against the
 * published pressure-flow curve (coronaryPhysiology.test.ts).
 *
 * Model (CGS units inside; outputs are fractions of normal):
 *  - Pressure drop across the narrowing, Gould, Cleve Clin Q 1980;47:140 (p.141):
 *      ΔP(Q) = F·Q + S·Q²,  F = 8πμL/As² (viscous),  S = (ρk/2)(1/As − 1/An)² (separation at the exit),
 *      An = πDn²/4, As = An(1 − ds)², ds = diameter stenosis.
 *  - One microvascular resistance in series. At maximal vasodilation it is Rmin = (Pa − Pv)/(CFR_N·Qref),
 *    so a normal artery has a flow reserve of CFR_N; venous pressure Pv = 0 (Pijls et al., Circulation
 *    1993;87:1354), which makes hyperaemic Pd/Pa exactly the maximal-flow fraction.
 *  - Autoregulation: the microvessels deliver whatever the territory demands until they reach Rmin, so
 *    flow = min(demand, Qmax). Resting flow therefore stays flat until about 85% narrowing while maximal
 *    flow starts to fall at about 50% (Gould, Lipscomb, Hamilton, Am J Cardiol 1974;33:87; Uren et al.,
 *    N Engl J Med 1994;330:1782).
 *  - Demand follows the rate-pressure product (Gobel et al., Circulation 1978;57:549): exercise raises
 *    heart rate to 85% of the age-predicted maximum (ACC/AHA 2002 exercise-testing guideline) and systolic
 *    pressure by 50 mmHg, capped at 210 (Fletcher et al., Circulation 2013;128:873).
 *  - Ischaemia fraction f = unmet share of the territory's demand. The cascade ORDER is sourced (Nesto &
 *    Kowalchuk, Am J Cardiol 1987;59:23C); its thresholds are illustrative.
 *  - Wall thickening vs flow: Gallagher et al., Am J Physiol 1984;247:H727 (akinesis at ~80% flow loss,
 *    Vatner, Circ Res 1980;47:201).
 *
 * Documented choices (not from a single paper): lesion length 5 mm, resting mean velocity 8 cm/s,
 * mean aortic pressure fixed at 100 mmHg, the +50 mmHg exercise rise, Pv = 0.
 */
import type { Vessel } from "../anatomy/registry";
import { TICK, ventricularContraction, type BeatTiming } from "./twin";

export const MMHG = 1333.22; // dyn/cm² per mmHg
const MU = 0.035; // blood viscosity, g/(cm·s): 3.5 cP, whole blood at high shear
const RHO = 1.05; // blood density, g/cm³
const K_EXIT = 1.0; // exit coefficient: Borda–Carnot abrupt expansion (Young & Tsai 1973 measured 1.52)
const LESION_CM = 0.5; // focal lesion length
const V_REST_CM_S = 8; // cross-sectional mean resting velocity (half the average peak velocity of ~16 cm/s)
export const PA_MMHG = 100; // mean aortic pressure (fixed)
const PV_MMHG = 0; // venous pressure (FFR simplification)
export const CFR_NORMAL = 4; // normal coronary flow reserve at the reference demand (3–4 is normal)
export const RPP_REF = 70 * 130; // reference rate-pressure product: dataset medians, pulse 70 × systolic BP 130
export const AGE_DEFAULT = 58; // dataset median age, used when age is missing
export const MAX_NARROWING = 0.95;

/** Per-artery constants, looked up by registry id (adding an artery to the registry requires an entry). */
export const ARTERY_PHYSIOLOGY: Record<Vessel, { diameterCm: number; dsvr: number; source: string }> = {
  // Proximal lumen diameters, right-dominant circulation: Dodge et al., Circulation 1992;86:232.
  // Resting diastolic-to-systolic velocity ratio: Seligman et al., EuroIntervention 2022;17:1260.
  lad: { diameterCm: 0.37, dsvr: 1.85, source: "Dodge 1992 (3.7 mm); Seligman 2022 (DSVR 1.85)" },
  lcx: { diameterCm: 0.34, dsvr: 1.85, source: "Dodge 1992 (3.4 mm); Seligman 2022 (DSVR 1.85)" },
  rca: { diameterCm: 0.39, dsvr: 1.53, source: "Dodge 1992 (3.9 mm); Seligman 2022 (DSVR 1.53)" },
};
/** The left main feeds the LAD and LCX: left-coronary phasic pattern. */
export const LEFT_MAIN_DSVR = 1.85;

/** Ischaemic cascade, in order (Nesto & Kowalchuk 1987). Thresholds on f are illustrative. */
export const CASCADE = [
  { key: "perfusion", label: "Perfusion defect", threshold: 0.05, input: null },
  { key: "wallMotion", label: "Wall-motion abnormality", threshold: 0.15, input: "Region RWMA" },
  { key: "st", label: "ST depression", threshold: 0.25, input: "St Depression" },
  { key: "angina", label: "Angina", threshold: 0.4, input: "Typical Chest Pain" },
] as const;

/** 0 = no ischaemia; 1..4 = the last cascade step reached. */
export type CascadeStage = 0 | 1 | 2 | 3 | 4;

export interface LabPatient {
  hr: number; // resting heart rate, bpm
  sbp: number; // resting systolic blood pressure, mmHg
  age: number; // years
}

export interface LabState {
  vessel: Vessel;
  narrowing: number; // diameter stenosis, 0..0.95
  exercise: number; // 0 rest .. 1 peak
  hr: number; // bpm at this exercise level
  sbp: number; // mmHg at this exercise level
  rpp: number; // rate-pressure product (demand), mmHg·bpm
  demand: number; // coronary flow demand in units of the reference resting flow
  qmaxFrac: number; // maximal (hyperaemic) flow as a fraction of normal maximal flow
  restFrac: number; // resting flow as a fraction of normal resting flow
  cfr: number; // coronary flow reserve: maximal / resting flow
  ffr: number; // hyperaemic distal-to-aortic pressure ratio (Pd/Pa at maximal flow)
  pdpa: number; // Pd/Pa at the current flow
  supplyRpp: number; // highest rate-pressure product this artery can supply (ischaemic threshold)
  ischaemia: number; // f, unmet share of the territory's demand, 0..1
  relFlow: number; // 1 − f: flow relative to a normal territory at the same demand
  stage: CascadeStage;
  wallThickening: number; // systolic wall thickening as a fraction of normal (1 = normal, 0 = akinetic)
  stMv: number; // illustrative ST depression, mV (0 below the ST step)
}

interface Coefficients {
  F: number; // viscous coefficient, dyn·s/cm⁵
  S: number; // separation coefficient, g/cm⁷
  qref: number; // reference resting flow, mL/s
  rmin: number; // microvascular resistance at maximal vasodilation, dyn·s/cm⁵
}

function coefficients(diameterCm: number, ds: number): Coefficients {
  const an = (Math.PI * diameterCm * diameterCm) / 4;
  const as = an * (1 - ds) ** 2;
  const qref = V_REST_CM_S * an;
  return {
    F: ds > 0 ? (8 * Math.PI * MU * LESION_CM) / (as * as) : 0,
    S: ((RHO * K_EXIT) / 2) * (1 / as - 1 / an) ** 2,
    qref,
    rmin: ((PA_MMHG - PV_MMHG) * MMHG) / (CFR_NORMAL * qref),
  };
}

/** Maximal flow (mL/s): stable root of S·Q² + (F + Rmin)·Q − (Pa − Pv) = 0. */
function maxFlow({ F, S, rmin }: Coefficients): number {
  const c = (PA_MMHG - PV_MMHG) * MMHG;
  const b = F + rmin;
  return S === 0 ? c / b : (2 * c) / (b + Math.sqrt(b * b + 4 * S * c));
}

/** Pressure drop across the narrowing at flow q (mL/s), mmHg. */
function pressureDrop({ F, S }: Coefficients, q: number): number {
  return (F * q + S * q * q) / MMHG;
}

/** Heart rate and systolic pressure at exercise level x (0 rest .. 1 peak). */
export function exerciseVitals(x: number, p: LabPatient) {
  const age = Number.isFinite(p.age) && p.age > 0 ? p.age : AGE_DEFAULT;
  const hrPeak = Math.max(p.hr, 0.85 * (220 - age));
  // +50 mmHg (about +10 per MET, ~5 METs), capped at 210; never below the resting value.
  const sbpPeak = Math.max(p.sbp, Math.min(p.sbp + 50, 210));
  const k = clamp01(x);
  return { hr: p.hr + k * (hrPeak - p.hr), sbp: p.sbp + k * (sbpPeak - p.sbp), hrPeak, sbpPeak };
}

/** Last cascade step reached at ischaemia fraction f. */
export function cascadeStage(f: number): CascadeStage {
  let s = 0;
  for (const step of CASCADE) if (f >= step.threshold) s++;
  return s as CascadeStage;
}

/** Systolic wall thickening vs relative subendocardial flow r (Gallagher et al. 1984), clamped to 0..1. */
export function wallThickening(r: number): number {
  return clamp01(-0.75 * r * r + 2.15 * r - 0.39);
}

/** Illustrative ST depression (mV): 0.1 mV (the 1 mm diagnostic threshold, ACC/AHA 2002) at the ST step, up to 0.2. */
export function stDepressionMv(f: number): number {
  return f >= CASCADE[2].threshold ? 0.1 + 0.1 * clamp01((f - 0.25) / 0.35) : 0;
}

/**
 * The lab's state for one artery, narrowing ds (0..0.95) and exercise level x (0..1), starting from the
 * patient's resting heart rate, systolic pressure and age.
 */
export function labState(vessel: Vessel, ds: number, x: number, patient: LabPatient): LabState {
  const narrowing = Math.min(MAX_NARROWING, Math.max(0, ds));
  const co = coefficients(ARTERY_PHYSIOLOGY[vessel].diameterCm, narrowing);
  const qmax = maxFlow(co);
  const { hr, sbp } = exerciseVitals(x, patient);
  const d0 = (patient.hr * patient.sbp) / RPP_REF;
  const d = (hr * sbp) / RPP_REF;
  const flow = Math.min(d * co.qref, qmax);
  const f = Math.max(0, 1 - qmax / (d * co.qref));
  return {
    vessel,
    narrowing,
    exercise: clamp01(x),
    hr,
    sbp,
    rpp: hr * sbp,
    demand: d,
    qmaxFrac: qmax / (CFR_NORMAL * co.qref),
    restFrac: Math.min(1, qmax / (d0 * co.qref)),
    cfr: qmax / Math.min(d0 * co.qref, qmax),
    ffr: 1 - pressureDrop(co, qmax) / PA_MMHG,
    pdpa: 1 - pressureDrop(co, flow) / PA_MMHG,
    supplyRpp: (qmax / co.qref) * RPP_REF,
    ischaemia: f,
    relFlow: 1 - f,
    stage: cascadeStage(f),
    wallThickening: wallThickening(1 - f),
    stMv: stDepressionMv(f),
  };
}

/**
 * Flow-reserve curve for the artery: resting and maximal flow (both in units of the patient's resting
 * flow through a normal artery) against narrowing, from 0 to 95% in `steps` steps.
 */
export function flowReserveCurve(vessel: Vessel, patient: LabPatient, steps = 96) {
  const d0 = (patient.hr * patient.sbp) / RPP_REF;
  const out: { narrowing: number; rest: number; max: number }[] = [];
  for (let i = 0; i < steps; i++) {
    const ds = (MAX_NARROWING * i) / (steps - 1);
    const co = coefficients(ARTERY_PHYSIOLOGY[vessel].diameterCm, ds);
    const qmax = maxFlow(co) / co.qref; // in reference resting flows
    out.push({ narrowing: ds, rest: Math.min(d0, qmax) / d0, max: qmax / d0 });
  }
  return out;
}

/**
 * Phasic coronary flow: diastole dominates because the contracting ventricle squeezes its own arteries.
 * Weight w(t) = (DSVR − (DSVR − 1)·v(t)) / mean, where v is the ventricular contraction (0..1) of the
 * shared twin mechanics; the mean over one beat is 1. Returns the normalising mean (computed at the
 * 2 ms clock ticks) so callers can evaluate w(t) cheaply every frame.
 */
export function phasicMean(dsvr: number, timing: BeatTiming): number {
  let sum = 0;
  let n = 0;
  for (let t = 0; t < timing.rr; t += TICK) {
    sum += dsvr - (dsvr - 1) * ventricularContraction(t, timing);
    n++;
  }
  return n ? sum / n : dsvr;
}

/** Share of each beat's flow that arrives while the ventricle is relaxed (outside the contraction window). */
export function diastolicShare(dsvr: number, timing: BeatTiming): number {
  let all = 0;
  let dia = 0;
  for (let t = 0; t < timing.rr; t += TICK) {
    const v = ventricularContraction(t, timing);
    const w = dsvr - (dsvr - 1) * v;
    all += w;
    if (v === 0) dia += w;
  }
  return all ? dia / all : 0;
}

function clamp01(x: number) {
  return Math.min(1, Math.max(0, x));
}
