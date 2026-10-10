import { describe, expect, it } from "vitest";
import { VESSEL_IDS } from "../anatomy/registry";
import type { ExplainResponse } from "../api/client";
import { labState } from "../twin/coronaryPhysiology";
import { DEFAULT_TWIN, beatTiming, ecgFindings, ecgSample, twinFromResult, type TwinParams } from "../twin/twin";
import { LAB_DEFAULT, baselineText, labFromQuery, labPatient, labTwin } from "./labState";

describe("ischaemia lab deep link", () => {
  it("is off without ?lab", () => {
    expect(labFromQuery("")).toBeNull();
    expect(labFromQuery("?patient=lad&heart=atlas")).toBeNull();
    expect(labFromQuery("?lab=0")).toBeNull();
    expect(labFromQuery("?lab=off")).toBeNull();
  });
  it("reads artery, narrowing and exercise", () => {
    expect(labFromQuery("?lab=lad&narrowing=80&exercise=1")).toEqual({ vessel: "lad", narrowing: 80, exercise: 1 });
    expect(labFromQuery("?lab=rca&narrowing=60&exercise=0.5")).toEqual({ vessel: "rca", narrowing: 60, exercise: 0.5 });
    for (const v of VESSEL_IDS) expect(labFromQuery(`?lab=${v}`)?.vessel).toBe(v);
  });
  it("clamps out-of-range values and falls back to defaults", () => {
    expect(labFromQuery("?lab=lcx&narrowing=150&exercise=7")).toEqual({ vessel: "lcx", narrowing: 95, exercise: 1 });
    expect(labFromQuery("?lab=lcx&narrowing=-5&exercise=-1")).toEqual({ vessel: "lcx", narrowing: 0, exercise: 0 });
    expect(labFromQuery("?lab=1&narrowing=abc")).toEqual({ ...LAB_DEFAULT, vessel: VESSEL_IDS[0] });
  });
});

describe("ischaemia lab and the twin", () => {
  const twin: TwinParams = { ...DEFAULT_TWIN, hr: 70, bp: 130, notRecorded: new Set() };
  const recorded = (values: Record<string, number>, imputed: string[] = []) => ({
    imputed,
    explanations: {
      cad: {
        feature_names: Object.keys(values), feature_values: Object.values(values),
        shap_values: Object.keys(values).map(() => 0), base_value: 0, derived: Object.keys(values).map(() => false),
      },
    },
  }) as unknown as ExplainResponse;
  const result = recorded({ Age: 66, BP: 130, PR: 70 });

  it("starts from the patient's recorded pulse, BP and age, and names their source", () => {
    expect(labPatient(result)).toEqual({ hr: 70, sbp: 130, age: 66 });
    const p = recorded({ Age: 50, BP: 180, PR: 100 });
    expect(labPatient(p)).toEqual({ hr: 100, sbp: 180, age: 50 });
    expect(baselineText(p, labPatient(p))).toBe("Starts from the patient's recorded pulse 100, BP 180, age 50");
    // A what-if scenario never changes the lab's starting values; the text says so.
    expect(baselineText(p, labPatient(p), true)).toBe("Starts from the patient's recorded pulse 100, BP 180, age 50, not the what-if scenario's");
  });

  it("uses the dataset medians before any estimate, and names inputs that were not entered", () => {
    expect(labPatient(null)).toEqual({ hr: 70, sbp: 130, age: 58 });
    expect(baselineText(null, labPatient(null))).toBe("Starts from the dataset medians (no estimate yet): pulse 70, BP 130, age 58");
    const imputed = recorded({ Age: 66, BP: 130, PR: 70 }, ["BP"]);
    expect(baselineText(imputed, labPatient(imputed))).toBe("Starts from the patient's recorded pulse 70, BP 130, age 66 (BP not entered: cohort median)");
    const both = recorded({ Age: 66, BP: 130, PR: 70 }, ["PR", "BP"]);
    expect(baselineText(both, labPatient(both))).toMatch(/\(pulse and BP not entered: cohort median\)$/);
  });

  it("drives the twin with the exercise rate and pressure, and marks the lab's ST depression", () => {
    const s = labState("lad", 0.9, 1, labPatient(result));
    const t = labTwin(twin, s);
    expect(t.hr).toBeCloseTo(s.hr, 9);
    expect(t.bp).toBeCloseTo(s.sbp, 9);
    expect(t.ef).toBe(twin.ef);
    expect(t.lab?.stMv).toBeGreaterThanOrEqual(0.1);
    expect(ecgFindings(t).some((f) => f.lab && f.text === "ST depression")).toBe(true);
    // At the exercise rate, the ST segment 60 ms after the J point (lead II) is depressed by at least 0.1 mV
    // compared with the same rate without the lab's ischaemia.
    const tm = beatTiming(t);
    const at = (p: TwinParams) => ecgSample(tm.pr + tm.qrs + 0.06, tm, p, "II");
    expect(at(t)).toBeLessThan(at({ ...t, lab: { stMv: 0 } }) - 0.1);
    expect(at(t)).toBeLessThan(-0.05);
  });

  it("leaves the twin unmarked without ischaemia, and returns to the patient's own values when off", () => {
    const s = labState("lad", 0.6, 1, labPatient(result));
    expect(s.ischaemia).toBe(0);
    const t = labTwin(twin, s);
    expect(t.lab?.stMv).toBe(0);
    expect(ecgFindings(t).some((f) => f.lab)).toBe(false);
    expect(twin.lab).toBeUndefined(); // the patient's twin itself is never modified
  });
});

describe("function twin from the recorded result", () => {
  it("beats at the recorded pulse (the cohort median when it was not entered), else the given pulse", () => {
    const res = (pr: number, imputed: string[] = []) => ({
      imputed,
      explanations: { cad: { feature_names: ["PR", "BP", "EF-TTE"], feature_values: [pr, 120, 55], shap_values: [0, 0, 0], base_value: 0, derived: [false, false, false] } },
    }) as unknown as ExplainResponse;
    expect(twinFromResult(res(88), 72).hr).toBe(88);
    const median = twinFromResult(res(70, ["PR"]), 72);
    expect(median.hr).toBe(70);
    expect(median.notRecorded.has("hr")).toBe(true);
    expect(twinFromResult(null, 64).hr).toBe(64);
  });
});
