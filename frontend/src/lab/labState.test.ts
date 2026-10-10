import { describe, expect, it } from "vitest";
import { VESSEL_IDS } from "../anatomy/registry";
import type { ExplainResponse } from "../api/client";
import { labState } from "../twin/coronaryPhysiology";
import { DEFAULT_TWIN, beatTiming, ecgFindings, ecgSample, type TwinParams } from "../twin/twin";
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
  const result = {
    explanations: { cad: { feature_names: ["Age", "BP"], feature_values: [66, 130], shap_values: [0, 0], base_value: 0, derived: [false, false] } },
  } as unknown as ExplainResponse;

  it("starts from the patient's resting values and age, and says when they are typical values", () => {
    expect(labPatient(result, twin)).toEqual({ hr: 70, sbp: 130, age: 66 });
    expect(labPatient(null, twin).age).toBe(58); // dataset median when age is unknown
    expect(baselineText(result, twin, labPatient(result, twin))).toBe("Starts from this patient's pulse 70, BP 130, age 66");
    expect(baselineText(null, twin, labPatient(null, twin))).toMatch(/^Starts from typical values/);
    const imputed: TwinParams = { ...twin, notRecorded: new Set(["bp"]) };
    expect(baselineText(result, imputed, labPatient(result, imputed))).toMatch(/BP typical: not entered/);
  });

  it("drives the twin with the exercise rate and pressure, and marks the lab's ST depression", () => {
    const s = labState("lad", 0.9, 1, labPatient(result, twin));
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
    const s = labState("lad", 0.6, 1, labPatient(result, twin));
    expect(s.ischaemia).toBe(0);
    const t = labTwin(twin, s);
    expect(t.lab?.stMv).toBe(0);
    expect(ecgFindings(t).some((f) => f.lab)).toBe(false);
    expect(twin.lab).toBeUndefined(); // the patient's twin itself is never modified
  });
});
