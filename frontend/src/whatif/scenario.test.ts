import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExplainResponse, FeatureInfo, PredictionResponse } from "../api/client";
import {
  EMPTY_SCENARIO, applyScenario, contrastView, CONTRASTS, deltaPts, describeScenario, isActive, normaliseScenario,
  scenarioDeltas, scenarioFromQuery, scenarioReducer, scenarioRequest, scenarioView, singleFlight, sliderRange, throttle,
  type Payload, type Scenario, type ScenarioBase, type ScenarioReply,
} from "./scenario";

const patient = { Age: 60, BP: 180, FBS: 104, TypicalChestPain: 1, Atypical: "N", Nonanginal: "N", CurrentSmoker: 1, EX_Smoker: 0, RegionRWMA: 0, EF_TTE: 55, StDepression: 0 };

/** A result whose CAD explanation lists the values the model used (as the API does for every input). */
function resultFor(values: Record<string, number>, imputed: string[] = [], cad = 0.9): ExplainResponse {
  const r = (p: number) => ({ probability: p, prediction: p > 0.5 ? 1 : 0, threshold: 0.5, band: "Higher", band_observed_rate: null });
  return {
    results: { cad: r(cad), lad: r(0.3), lcx: r(0.2), rca: r(0.5) },
    vessel_probabilities: { lad: 0.3, lcx: 0.2, rca: 0.5 },
    imputed,
    explanations: { cad: { feature_names: Object.keys(values), feature_values: Object.values(values) } },
  } as unknown as ExplainResponse;
}
const COLUMNS = { Age: 60, BP: 180, FBS: 104, TG: 122, "Typical Chest Pain": 1, Atypical: 0, Nonanginal: 0, "Current Smoker": 1, "Region RWMA": 0, "EF-TTE": 55, "St Depression": 0 };
const baseOf = (body: Payload, result = resultFor(COLUMNS)): ScenarioBase => ({ body, key: JSON.stringify(body), result });
const info: Record<string, FeatureInfo> = {
  BP: { type: "numeric", default: 130, p5: 100, p95: 160 },
  FBS: { type: "numeric", default: 98, p5: 77, p95: 218.9 },
  TG: { type: "numeric", default: 122, p5: 69, p95: 308.6 },
};
const rwma: Scenario = { levers: {}, quitSmoking: false, contrasts: { rwma: true } };

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
    expect(applyScenario(blank, rwma, null)).toBeNull();
    const result = { explanations: { cad: { feature_names: ["Region RWMA"], feature_values: [1] } } } as unknown as ExplainResponse;
    expect(applyScenario(blank, rwma, result)).toMatchObject({ RegionRWMA: 0 });
  });

  it("has nothing to quit for a patient who is not a current smoker", () => {
    const s: Scenario = { levers: {}, quitSmoking: true, contrasts: {} };
    const never = { ...patient, CurrentSmoker: 0 };
    expect(applyScenario(never, s, null)).toEqual(never);
    expect(applyScenario({ ...patient, CurrentSmoker: undefined as unknown as number }, s, null)).toBeNull(); // not known yet
  });
});

describe("M3: scenarios build on the submitted patient run, never on unsubmitted form edits", () => {
  it("takes every value from the request body behind the patient's result", () => {
    const submitted = { ...patient };
    const req = scenarioRequest(baseOf(submitted), rwma)!;
    expect(req.body).toEqual({ ...submitted, RegionRWMA: 2 });
    expect(req.baseKey).toBe(JSON.stringify(submitted));
    // Editing the form afterwards (Age 60 -> 80, not yet re-run) cannot reach the request: it is not an input.
    const edited = { ...submitted, Age: 80 };
    expect(scenarioRequest(baseOf(submitted), rwma)!.body.Age).toBe(60);
    expect(scenarioRequest(baseOf(edited), rwma)!.key).not.toBe(req.key); // only a new run moves the scenario
  });

  it("needs a patient run first, and drops replies built on an earlier run", () => {
    expect(scenarioRequest(null, rwma)).toBeNull();
    const run1 = baseOf(patient);
    const run2 = baseOf({ ...patient, Age: 80 }, resultFor({ ...COLUMNS, Age: 80 }));
    const reply: ScenarioReply = { ...scenarioRequest(run1, rwma)!, scenario: rwma, data: resultFor(COLUMNS, [], 0.97), explained: true };
    const v = scenarioView(scenarioRequest(run2, rwma), reply, null, run2.result);
    expect(v).toMatchObject({ status: "calculating", reply: null });
    expect(v.shown).toBe(run2.result);
  });
});

describe("M4: no stuck scenarios", () => {
  it("drops quitting for a patient who is not a current smoker (deep link ?quit=1 on a non-smoker)", () => {
    const quit: Scenario = { levers: {}, quitSmoking: true, contrasts: {} };
    const never = { ...patient, CurrentSmoker: 0 };
    const s = normaliseScenario(quit, never, resultFor({ ...COLUMNS, "Current Smoker": 0 }), info);
    expect(s.quitSmoking).toBe(false);
    expect(isActive(s)).toBe(false);
    const base = baseOf(never, resultFor({ ...COLUMNS, "Current Smoker": 0 }));
    expect(scenarioRequest(base, s)).toBeNull();
    expect(scenarioView(null, null, null, base.result)).toMatchObject({ status: "off", reply: null });
    // A current smoker keeps it, and an unchanged scenario is returned as is (no re-render loop).
    expect(normaliseScenario(quit, patient, resultFor(COLUMNS), info)).toBe(quit);
    // Before the patient's values are known nothing is dropped.
    expect(normaliseScenario(quit, null, null, null)).toBe(quit);
  });

  it("clamps deep-link lever values to the slider range and drops them when they land on the patient's value", () => {
    const deep = scenarioFromQuery("?whatif=1&bp=999&fbs=10&tg=150.4").scenario;
    const s = normaliseScenario(deep, patient, resultFor(COLUMNS), info);
    // BP range 100..180 (p95 160 widened to the patient's 180): 999 -> 180 = the patient's value -> no override.
    expect(s.levers).toEqual({ FBS: 77, TG: 150 });
    expect(normaliseScenario(s, patient, resultFor(COLUMNS), info)).toBe(s); // idempotent
    // Before the cohort ranges load: the form's validation range (BP 60-260), so the request is still valid.
    expect(normaliseScenario(deep, patient, resultFor(COLUMNS), null).levers).toMatchObject({ BP: 260, FBS: 40 });
  });

  it("exposes a reply only for the current request, or while that request is pending on the same run", () => {
    const base = baseOf(patient);
    const reqA = scenarioRequest(base, rwma)!;
    const both: Scenario = { ...rwma, contrasts: { rwma: true, angina: true } };
    const reqB = scenarioRequest(base, both)!;
    const replyA: ScenarioReply = { ...reqA, scenario: rwma, data: resultFor(COLUMNS, [], 0.97), explained: true };
    expect(scenarioView(reqA, replyA, null, base.result)).toMatchObject({ status: "ready", reply: replyA });
    // B requested, A still on screen while B is calculated (mid-drag / just after a toggle): flagged as updating.
    expect(scenarioView(reqB, replyA, null, base.result)).toMatchObject({ status: "updating", reply: replyA });
    // B failed: back to the patient, never the stale A under B's switches.
    const failed = scenarioView(reqB, replyA, { key: reqB.key, kind: "explain", message: "API offline" }, base.result);
    expect(failed).toMatchObject({ status: "error", reply: null });
    expect(failed.shown).toBe(base.result);
    // An old failure does not hide a newer request.
    expect(scenarioView(reqB, replyA, { key: reqA.key, kind: "predict", message: "x" }, base.result).status).toBe("updating");
  });

  it("Reset returns the patient's own result exactly", () => {
    const base = baseOf(patient);
    const req = scenarioRequest(base, rwma)!;
    const reply: ScenarioReply = { ...req, scenario: rwma, data: resultFor(COLUMNS, [], 0.97), explained: true };
    const s = scenarioReducer(rwma, { type: "reset" });
    const v = scenarioView(scenarioRequest(base, s), reply, null, base.result);
    expect(v.status).toBe("off");
    expect(v.shown).toBe(base.result);
  });
});

describe("m1: no what-if labels on the patient's own or a stale result", () => {
  it("shows the patient, unlabelled, before the first reply and after a failed first request", () => {
    const base = baseOf(patient);
    const req = scenarioRequest(base, rwma)!;
    const calculating = scenarioView(req, null, null, base.result);
    expect(calculating).toMatchObject({ status: "calculating", reply: null, explanationsLag: false });
    expect(calculating.shown).toBe(base.result);
    const failed = scenarioView(req, null, { key: req.key, kind: "explain", message: "timeout" }, base.result);
    expect(failed).toMatchObject({ status: "error", reply: null });
    expect(failed.shown).toBe(base.result);
  });

  it("flags explanations that trail a fast estimate", () => {
    const base = baseOf(patient);
    const req = scenarioRequest(base, rwma)!;
    const fast: ScenarioReply = { ...req, scenario: rwma, data: resultFor(COLUMNS, [], 0.97), explained: false };
    expect(scenarioView(req, fast, null, base.result)).toMatchObject({ status: "updating", explanationsLag: true });
  });
});

describe("summaries and contrast directions", () => {
  it("says what changed, marking findings added to the patient as hypothetical", () => {
    const s: Scenario = { levers: { BP: 120, TG: 100 }, quitSmoking: true, contrasts: { angina: true, rwma: true, ef: true } };
    const { TG: _, ...body } = { ...patient, TG: 0 };
    expect(describeScenario(s, body, resultFor(COLUMNS, ["TG"]))).toEqual([
      "systolic BP 180 → 120 mmHg",
      "triglycerides typical 122 → 100 mg/dL",
      "current smoker → ex-smoker",
      "typical angina → no chest pain",
      "wall-motion grade 0 → 2 (hypothetical)",
      "EF 55% → 35% (hypothetical)",
    ]);
  });

  it("names the recorded chest-pain type when typical angina is added", () => {
    const angina = CONTRASTS.find((c) => c.id === "angina")!;
    const atypical = { ...patient, TypicalChestPain: 0, Atypical: "Y" };
    expect(contrastView(angina, atypical, null)).toEqual({ dir: "abnormal", from: "atypical chest pain", to: "typical angina" });
    expect(contrastView(angina, patient, null)).toEqual({ dir: "normal", from: "typical angina", to: "no chest pain" });
  });
});

describe("slider range and deltas", () => {
  it("spans the cohort 5th-95th percentile, widened to include the patient", () => {
    const i = { type: "numeric" as const, default: 130, p5: 100, p95: 160 };
    expect(sliderRange(i, 130, 1)).toMatchObject({ lo: 100, hi: 160 });
    expect(sliderRange(i, 180, 1)).toMatchObject({ lo: 100, hi: 180, p95: 160 });
  });

  it("reports CAD then every artery", () => {
    const r = (p: number) => ({ probability: p, band: "Lower" });
    const mk = (p: number) => ({ results: { cad: r(p), lad: r(p), lcx: r(p), rca: r(p) } }) as unknown as PredictionResponse;
    const d = scenarioDeltas(mk(0.3), mk(0.6));
    expect(d.map((x) => x.label)).toEqual(["CAD", "LAD", "LCX", "RCA"]);
    expect(d[1]).toMatchObject({ from: 0.3, to: 0.6 });
  });

  it("m3: points match the percentages shown, so '>95%' never leaks the hidden value", () => {
    expect(deltaPts(0.9, 0.98)).toBe(5); // "90% → >95%", not +8
    expect(deltaPts(0.03, 0.3)).toBe(25); // "<5% → 30%"
    expect(deltaPts(0.33, 0.09)).toBe(-24);
    expect(deltaPts(0.5, 0.5)).toBe(0);
  });
});

describe("deep link", () => {
  it("parses ?whatif=1 with levers, quit and contrasts", () => {
    const { open, scenario } = scenarioFromQuery("?patient=rca&whatif=1&bp=120&fbs=abc&hdl=45&quit=1&contrast=angina,bogus");
    expect(open).toBe(true);
    // HDL is locked (Model verification §7), so ?hdl= is ignored.
    expect(scenario).toEqual({ levers: { BP: 120 }, quitSmoking: true, contrasts: { angina: true } });
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

describe("m8: one request at a time, newest waiting, none aborted", () => {
  type J = { key: string; kind: "predict" | "explain" };
  const setup = () => {
    const started: J[] = [];
    const dones: (() => void)[] = [];
    const q = singleFlight<J>((job, done) => {
      started.push(job);
      dones.push(done);
    });
    return { q, started, finish: () => dones.shift()!() };
  };

  it("lets the running request finish and then sends only the newest waiting one", () => {
    const { q, started, finish } = setup();
    q.submit({ key: "a", kind: "predict" });
    q.submit({ key: "b", kind: "predict" });
    q.submit({ key: "c", kind: "predict" });
    expect(started.map((j) => j.key)).toEqual(["a"]); // b superseded while waiting, a not aborted
    finish();
    expect(started.map((j) => j.key)).toEqual(["a", "c"]);
    finish();
    expect(q.running).toBeNull();
  });

  it("does not repeat a request already running (an explanation also answers a fast estimate)", () => {
    const { q, started, finish } = setup();
    q.submit({ key: "a", kind: "explain" });
    q.submit({ key: "b", kind: "predict" });
    q.submit({ key: "a", kind: "predict" }); // back at a: the running explanation answers it
    finish();
    expect(started).toEqual([{ key: "a", kind: "explain" }]);
    q.submit({ key: "a", kind: "predict" });
    q.submit({ key: "a", kind: "explain" }); // a fast estimate does not answer an explanation
    finish();
    expect(started.slice(1)).toEqual([{ key: "a", kind: "predict" }, { key: "a", kind: "explain" }]);
  });

  it("clear forgets everything (scenario off or a new patient run)", () => {
    const { q, started, finish } = setup();
    q.submit({ key: "a", kind: "predict" });
    q.submit({ key: "b", kind: "predict" });
    q.clear();
    finish(); // the aborted request settling late starts nothing
    expect(started.map((j) => j.key)).toEqual(["a"]);
    q.submit({ key: "c", kind: "explain" });
    expect(started.map((j) => j.key)).toEqual(["a", "c"]);
  });
});
