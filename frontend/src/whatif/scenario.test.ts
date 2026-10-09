import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExplainResponse, PredictionResponse } from "../api/client";
import {
  EMPTY_SCENARIO, applyScenario, isActive, scenarioDeltas, scenarioFromQuery, scenarioReducer, sliderRange, throttle,
  type Scenario,
} from "./scenario";

const patient = { Age: 60, BP: 180, FBS: 104, TypicalChestPain: 1, Atypical: "N", Nonanginal: "N", CurrentSmoker: 1, EX_Smoker: 0, RegionRWMA: 0, EF_TTE: 55, StDepression: 0 };

describe("scenario reducer", () => {
  it("sets overrides, drops them when back at the recorded value, and resets to empty", () => {
    let s = scenarioReducer(EMPTY_SCENARIO, { type: "lever", key: "BP", value: 130, baseline: 180 });
    expect(s.levers.BP).toBe(130);
    expect(isActive(s)).toBe(true);
    s = scenarioReducer(s, { type: "lever", key: "BP", value: 180, baseline: 180 });
    expect(s.levers.BP).toBeUndefined();
    expect(isActive(s)).toBe(false);
    s = scenarioReducer(s, { type: "contrast", id: "rwma", on: true });
    s = scenarioReducer(s, { type: "quit", on: true });
    expect(isActive(s)).toBe(true);
    s = scenarioReducer(s, { type: "reset" });
    expect(s).toEqual(EMPTY_SCENARIO);
    expect(isActive(s)).toBe(false);
  });

  it("never edits the form payload", () => {
    const form = { ...patient };
    const s: Scenario = { levers: { BP: 120 }, quitSmoking: true, contrasts: { angina: true } };
    const body = applyScenario(form, s, null)!;
    expect(form).toEqual(patient);
    expect(body).toMatchObject({ BP: 120, CurrentSmoker: 0, EX_Smoker: 1, TypicalChestPain: 0, Atypical: "N", Nonanginal: "N" });
    expect(applyScenario(form, EMPTY_SCENARIO, null)).toEqual(patient);
  });

  it("switches a present finding off and an absent one on (verification contrasts)", () => {
    const s: Scenario = { levers: {}, quitSmoking: false, contrasts: { rwma: true, ef: true, st: true } };
    expect(applyScenario(patient, s, null)).toMatchObject({ RegionRWMA: 2, EF_TTE: 35, StDepression: 1 });
    expect(applyScenario({ ...patient, RegionRWMA: 3, EF_TTE: 30, StDepression: 1 }, s, null))
      .toMatchObject({ RegionRWMA: 0, EF_TTE: 60, StDepression: 0 });
  });

  it("uses the value the model saw when the form left a field blank, and waits when it is unknown", () => {
    const { RegionRWMA: _, ...blank } = patient;
    const s: Scenario = { levers: {}, quitSmoking: false, contrasts: { rwma: true } };
    expect(applyScenario(blank, s, null)).toBeNull();
    const result = { explanations: { cad: { feature_names: ["Region RWMA"], feature_values: [1] } } } as unknown as ExplainResponse;
    expect(applyScenario(blank, s, result)).toMatchObject({ RegionRWMA: 0 });
  });

  it("only lets a current smoker quit", () => {
    const s: Scenario = { levers: {}, quitSmoking: true, contrasts: {} };
    expect(applyScenario({ ...patient, CurrentSmoker: 0 }, s, null)).toBeNull();
  });
});

describe("slider range and deltas", () => {
  it("spans the cohort 5th-95th percentile, widened to include the patient", () => {
    const info = { type: "numeric" as const, default: 130, p5: 100, p95: 160 };
    expect(sliderRange(info, 130, 1)).toMatchObject({ lo: 100, hi: 160 });
    expect(sliderRange(info, 180, 1)).toMatchObject({ lo: 100, hi: 180, p95: 160 });
  });

  it("reports CAD then every artery", () => {
    const r = (p: number) => ({ probability: p, band: "Lower" });
    const mk = (p: number) => ({ results: { cad: r(p), lad: r(p), lcx: r(p), rca: r(p) } }) as unknown as PredictionResponse;
    const d = scenarioDeltas(mk(0.3), mk(0.6));
    expect(d.map((x) => x.label)).toEqual(["CAD", "LAD", "LCX", "RCA"]);
    expect(d[1]).toMatchObject({ from: 0.3, to: 0.6 });
  });
});

describe("deep link", () => {
  it("parses ?whatif=1 with levers, quit and contrasts", () => {
    const { open, scenario } = scenarioFromQuery("?patient=rca&whatif=1&bp=120&fbs=abc&quit=1&contrast=rwma,bogus");
    expect(open).toBe(true);
    expect(scenario).toEqual({ levers: { BP: 120 }, quitSmoking: true, contrasts: { rwma: true } });
    expect(scenarioFromQuery("?bp=120").open).toBe(false);
  });
});

describe("throttle", () => {
  afterEach(() => vi.useRealTimers());

  it("calls at most once per interval and always ends on the latest value", () => {
    vi.useFakeTimers();
    const calls: number[] = [];
    const t = throttle((v: number) => calls.push(v), 120);
    t.call(1); // leading edge: immediate
    for (let v = 2; v <= 10; v++) {
      vi.advanceTimersByTime(10);
      t.call(v);
    }
    expect(calls).toEqual([1]);
    vi.advanceTimersByTime(120);
    expect(calls).toEqual([1, 10]); // trailing edge with the newest value only
    // A 1-second drag with an input every 16 ms sends about 1000/120 requests, not 60.
    calls.length = 0;
    for (let i = 0; i < 60; i++) {
      t.call(i);
      vi.advanceTimersByTime(16);
    }
    vi.advanceTimersByTime(200);
    expect(calls.length).toBeLessThanOrEqual(10);
    expect(calls[calls.length - 1]).toBe(59);
  });

  it("cancel drops the pending call (reset / release)", () => {
    vi.useFakeTimers();
    const calls: number[] = [];
    const t = throttle((v: number) => calls.push(v), 120);
    t.call(1);
    t.call(2);
    t.cancel();
    vi.advanceTimersByTime(500);
    expect(calls).toEqual([1]);
  });
});
