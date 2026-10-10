import { describe, expect, it } from "vitest";
import { VESSEL_IDS } from "../anatomy/registry";
import {
  ARTERY_PHYSIOLOGY, CASCADE, cascadeStage, diastolicShare, exerciseVitals, flowReserveCurve, labState,
  phasicMean, stDepressionMv, wallThickening,
} from "./coronaryPhysiology";
import { DEFAULT_TWIN, beatTiming, mechanics, TICK } from "./twin";

// Reference patient: dataset medians (pulse 70, systolic BP 130, age 58), so the resting demand is 1.
const REF = { hr: 70, sbp: 130, age: 58 };
const TOL = 0.02;
const at = (v: "lad" | "lcx" | "rca", pct: number, x = 0, p = REF) => labState(v, pct / 100, x, p);
/** First narrowing (0.1% steps) at which `cond` holds. */
const firstWhere = (cond: (pct: number) => boolean, from = 0, to = 95) => {
  for (let pct = from; pct <= to + 1e-9; pct += 0.1) if (cond(pct)) return pct;
  return NaN;
};

describe("coronary physiology: Gould pressure-flow curve (LAD, reference demand)", () => {
  // narrowing %, max flow / normal, flow reserve, resting flow, hyperaemic Pd/Pa, resting Pd/Pa
  const table: [number, number, number, number, number, number][] = [
    [0, 1.0, 4.0, 1.0, 1.0, 1.0],
    [30, 0.992, 3.97, 1.0, 0.992, 0.999],
    [40, 0.98, 3.92, 1.0, 0.98, 0.997],
    [50, 0.952, 3.81, 1.0, 0.952, 0.994],
    [60, 0.88, 3.52, 1.0, 0.88, 0.983],
    [70, 0.708, 2.83, 1.0, 0.708, 0.944],
    [75, 0.566, 2.27, 1.0, 0.566, 0.88],
    [80, 0.395, 1.58, 1.0, 0.395, 0.701],
    [85, 0.218, 1.0, 0.87, 0.218, 0.218],
    [88, 0.124, 1.0, 0.5, 0.124, 0.124],
    [90, 0.073, 1.0, 0.29, 0.073, 0.073],
    [95, 0.006, 1.0, 0.025, 0.006, 0.006],
  ];
  it.each(table)("%i%% narrowing", (pct, qmax, cfr, rest, ffr, pdpa) => {
    const s = at("lad", pct);
    expect(s.qmaxFrac).toBeCloseTo(qmax, 1);
    expect(Math.abs(s.qmaxFrac - qmax)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(s.cfr - cfr)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(s.restFrac - rest)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(s.ffr - ffr)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(s.pdpa - pdpa)).toBeLessThanOrEqual(TOL);
  });

  it("normal flow reserve is about 4 (3–4 is normal)", () => {
    expect(at("lad", 0).cfr).toBeCloseTo(4, 6);
  });

  it("maximal flow holds to 50% and first falls below 0.95 between 45% and 55%", () => {
    for (let pct = 0; pct <= 50; pct += 0.5) expect(at("lad", pct).qmaxFrac).toBeGreaterThanOrEqual(0.95);
    const onset = firstWhere((pct) => at("lad", pct).qmaxFrac < 0.95);
    expect(onset).toBeGreaterThan(45);
    expect(onset).toBeLessThan(55);
  });

  it("resting flow stays flat (≥ 0.98) to 80% and starts to fall between 80% and 88%", () => {
    for (let pct = 0; pct <= 80; pct += 0.5) expect(at("lad", pct).restFrac).toBeGreaterThanOrEqual(0.98);
    const onset = firstWhere((pct) => at("lad", pct).restFrac < 0.98);
    expect(onset).toBeGreaterThan(80);
    expect(onset).toBeLessThan(88);
  });

  it("hyperaemic Pd/Pa first reaches 0.80 between 60% and 70%", () => {
    const onset = firstWhere((pct) => at("lad", pct).ffr <= 0.8);
    expect(onset).toBeGreaterThan(60);
    expect(onset).toBeLessThan(70);
  });

  it("with venous pressure 0, hyperaemic Pd/Pa equals the maximal-flow fraction", () => {
    for (const v of VESSEL_IDS) {
      for (let pct = 0; pct <= 95; pct += 5) {
        const s = labState(v, pct / 100, 0, REF);
        expect(Math.abs(s.ffr - s.qmaxFrac)).toBeLessThan(1e-9);
      }
    }
  });

  it("every output changes monotonically with narrowing", () => {
    for (const v of VESSEL_IDS) {
      for (const x of [0, 0.5, 1]) {
        let prev = labState(v, 0, x, REF);
        for (let pct = 0.5; pct <= 95; pct += 0.5) {
          const s = labState(v, pct / 100, x, REF);
          for (const k of ["qmaxFrac", "restFrac", "cfr", "ffr", "pdpa", "supplyRpp", "relFlow", "wallThickening"] as const) {
            expect(s[k]).toBeLessThanOrEqual(prev[k] + 1e-12);
          }
          expect(s.ischaemia).toBeGreaterThanOrEqual(prev.ischaemia - 1e-12);
          expect(s.stage).toBeGreaterThanOrEqual(prev.stage);
          prev = s;
        }
      }
    }
  });
});

describe("coronary physiology: per-artery rows", () => {
  it("LCX at 50% / 80% / 90%", () => {
    expect(Math.abs(at("lcx", 50).qmaxFrac - 0.95)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(at("lcx", 80).qmaxFrac - 0.382)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(at("lcx", 90).qmaxFrac - 0.066)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(at("lcx", 90).restFrac - 0.263)).toBeLessThanOrEqual(TOL);
  });
  it("RCA at 50% / 80% / 90%", () => {
    expect(Math.abs(at("rca", 50).qmaxFrac - 0.954)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(at("rca", 80).qmaxFrac - 0.402)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(at("rca", 90).qmaxFrac - 0.078)).toBeLessThanOrEqual(TOL);
    expect(Math.abs(at("rca", 90).restFrac - 0.31)).toBeLessThanOrEqual(TOL);
  });
  it("has constants for every registry artery", () => {
    for (const v of VESSEL_IDS) expect(ARTERY_PHYSIOLOGY[v].diameterCm).toBeGreaterThan(0);
  });
});

describe("coronary physiology: exercise and the ischaemic cascade (LAD, age 58, HR 70, SBP 130)", () => {
  it("peak exercise: 85% of age-predicted maximum heart rate, systolic +50 mmHg", () => {
    const peak = exerciseVitals(1, REF);
    expect(peak.hr).toBeCloseTo(137.7, 1);
    expect(peak.sbp).toBe(180);
    expect(at("lad", 0, 1).demand).toBeCloseTo(2.724, 2);
    expect(at("lad", 0, 0.5).demand).toBeCloseTo(1.769, 2);
  });

  it("never lowers systolic pressure with exercise, even when it starts above the cap", () => {
    expect(exerciseVitals(1, { hr: 70, sbp: 220, age: 58 }).sbp).toBe(220);
    expect(exerciseVitals(1, { hr: 170, sbp: 130, age: 80 }).hr).toBe(170);
  });

  // narrowing %, [f, stage] at rest, mid (x = 0.5) and peak (x = 1)
  const rows: [number, [number, number], [number, number], [number, number]][] = [
    [0, [0, 0], [0, 0], [0, 0]],
    [60, [0, 0], [0, 0], [0, 0]],
    [70, [0, 0], [0, 0], [0, 0]],
    [75, [0, 0], [0, 0], [0.168, 2]],
    [80, [0, 0], [0.107, 1], [0.42, 4]],
    [85, [0.13, 1], [0.508, 4], [0.68, 4]],
    [90, [0.708, 4], [0.835, 4], [0.893, 4]],
  ];
  it.each(rows)("%i%% narrowing", (pct, rest, mid, peak) => {
    [rest, mid, peak].forEach(([f, stage], i) => {
      const s = at("lad", pct, i / 2);
      expect(Math.abs(s.ischaemia - f)).toBeLessThanOrEqual(TOL);
      expect(s.stage).toBe(stage);
    });
  });

  it("at peak, ischaemia first appears at about 71% and the ST step falls between 75% and 80%", () => {
    const onset = firstWhere((pct) => at("lad", pct, 1).ischaemia > 0);
    expect(onset).toBeGreaterThan(69);
    expect(onset).toBeLessThan(73);
    const st = firstWhere((pct) => at("lad", pct, 1).stage >= 3);
    expect(st).toBeGreaterThan(75);
    expect(st).toBeLessThan(80);
  });

  it("supports a rate-pressure product of about 14,380 mmHg·bpm at 80%", () => {
    expect(Math.abs(at("lad", 80).supplyRpp - 14380) / 14380).toBeLessThan(0.01);
  });

  it("lights the cascade in the fixed order: perfusion, wall motion, ST depression, angina", () => {
    expect(CASCADE.map((c) => c.key)).toEqual(["perfusion", "wallMotion", "st", "angina"]);
    for (let i = 1; i < CASCADE.length; i++) expect(CASCADE[i].threshold).toBeGreaterThan(CASCADE[i - 1].threshold);
    expect([0, 0.05, 0.15, 0.25, 0.4, 1].map(cascadeStage)).toEqual([0, 1, 2, 3, 4, 4]);
  });

  it("wall thickening: normal at full flow, akinetic at about 80% flow loss (Gallagher 1984)", () => {
    expect(wallThickening(1)).toBe(1);
    expect(wallThickening(0.2)).toBeLessThan(0.02);
    expect(wallThickening(0)).toBe(0);
  });

  it("ST depression: 0.1 mV at the ST step, at most 0.2 mV, none below it", () => {
    expect(stDepressionMv(0.2)).toBe(0);
    expect(stDepressionMv(0.25)).toBeCloseTo(0.1, 9);
    expect(stDepressionMv(1)).toBeCloseTo(0.2, 9);
  });

  it("the flow-reserve curve meets resting flow at about 84% narrowing", () => {
    const curve = flowReserveCurve("lad", REF);
    const meet = curve.find((c) => c.max <= c.rest + 1e-9)!;
    expect(meet.narrowing).toBeGreaterThan(0.8);
    expect(meet.narrowing).toBeLessThan(0.88);
    expect(curve[0].max).toBeCloseTo(4, 6);
  });
});

describe("coronary physiology: phasic (mostly diastolic) flow", () => {
  const p = { ...DEFAULT_TWIN, hr: 70, notRecorded: new Set<never>() };
  const tm = beatTiming(p);
  it("normalised weight averages 1 over a beat, highest in diastole", () => {
    const left = ARTERY_PHYSIOLOGY.lad.dsvr;
    const mean = phasicMean(left, tm);
    let sum = 0, n = 0, lo = Infinity, hi = -Infinity;
    for (let t = 0; t < tm.rr; t += TICK) {
      const w = (left - (left - 1) * mechanics(t, tm, p).ventricular) / mean;
      sum += w; n++;
      lo = Math.min(lo, w); hi = Math.max(hi, w);
    }
    expect(sum / n).toBeCloseTo(1, 6);
    expect(hi / lo).toBeCloseTo(left, 2); // diastolic-to-systolic velocity ratio
  });
  it("most flow arrives in diastole, more so in the left coronary arteries than the RCA", () => {
    const left = diastolicShare(ARTERY_PHYSIOLOGY.lad.dsvr, tm);
    const right = diastolicShare(ARTERY_PHYSIOLOGY.rca.dsvr, tm);
    expect(left).toBeGreaterThan(0.6);
    expect(right).toBeGreaterThan(0.6);
    expect(left).toBeGreaterThan(right);
  });
});
