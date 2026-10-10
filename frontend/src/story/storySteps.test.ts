import { describe, expect, it } from "vitest";
import { VESSEL_IDS } from "../anatomy/registry";
import type { ExplainResponse, MetricsResponse } from "../api/client";
import { labState } from "../twin/coronaryPhysiology";
import { DEFAULT_TWIN, type TwinParams } from "../twin/twin";
import {
  HONESTY, LAB_DEMO, STORY_STEPS, honestyAucs, labDemo, storyCaption, storyFromQuery, topArtery, type StoryContext,
} from "./storySteps";

const result = (p: Record<string, number>) => ({
  results: Object.fromEntries(Object.entries(p).map(([k, v]) => [k, { probability: v, prediction: v >= 0.5 ? 1 : 0, threshold: 0.5, band: "", band_observed_rate: null }])),
  imputed: [],
}) as unknown as ExplainResponse;

const twin: TwinParams = { ...DEFAULT_TWIN, hr: 80, ef: 55, bp: 130, stDepression: true, tInversion: true, notRecorded: new Set() };
const ctx = (over: Partial<StoryContext> = {}): StoryContext => ({
  result: result({ cad: 0.97, lad: 0.93, lcx: 0.88, rca: 0.83 }), twin, whatIf: false, stale: false, top: "lad", lab: null, t: 0, ...over,
});

describe("story script", () => {
  it("has six steps in the planned order, about 90 s in all", () => {
    expect(STORY_STEPS.map((s) => s.id)).toEqual(["chest", "twin", "reasoning", "arteries", "lab", "honesty"]);
    const total = STORY_STEPS.reduce((a, s) => a + s.seconds, 0);
    expect(total).toBeGreaterThanOrEqual(80);
    expect(total).toBeLessThanOrEqual(100);
    expect(LAB_DEMO.exercise.to).toBeLessThan(STORY_STEPS[4].seconds);
  });

  it("reads ?story=1 and ?story=1&step=N (shown paused at the step's end; &play=1 plays on)", () => {
    expect(storyFromQuery("")).toBeNull();
    expect(storyFromQuery("?patient=lad")).toBeNull();
    expect(storyFromQuery("?story=0")).toBeNull();
    expect(storyFromQuery("?story=1")).toEqual({ step: 0, playing: true, atEnd: false });
    expect(storyFromQuery("?story=1&step=5")).toEqual({ step: 4, playing: false, atEnd: true });
    expect(storyFromQuery("?story=1&step=3&play=1")).toEqual({ step: 2, playing: true, atEnd: false });
    expect(storyFromQuery("?story=1&step=9")?.step).toBe(5);
    expect(storyFromQuery("?story=1&step=0")?.step).toBe(0);
    expect(storyFromQuery("?story=1&step=abc")).toEqual({ step: 0, playing: true, atEnd: false });
  });

  it("inspects the artery in the highest band, ties broken by probability", () => {
    expect(topArtery(null)).toBeNull();
    expect(topArtery(result({ cad: 0.9, lad: 0.93, lcx: 0.88, rca: 0.83 }))).toBe("lad");
    expect(topArtery(result({ cad: 0.9, lad: 0.7, lcx: 0.95, rca: 0.83 }))).toBe("lcx");
    expect(topArtery(result({ cad: 0.6, lad: 0.2, lcx: 0.1, rca: 0.5 }))).toBe("rca"); // undecided beats lower
    expect(topArtery(result({ cad: 0.2, lad: 0.1, lcx: 0.12, rca: 0.05 }))).toBe("lcx"); // all lower: highest chance
    // Equal probabilities: registry order.
    expect(topArtery(result({ cad: 0.5, lad: 0.4, lcx: 0.4, rca: 0.4 }))).toBe(VESSEL_IDS[0]);
  });

  it("demonstrates the lab: the narrowing grows to 85% first, then rest to peak exercise", () => {
    expect(labDemo(0)).toEqual({ narrowing: 0, exercise: 0 });
    expect(labDemo(LAB_DEMO.narrowing.to)).toEqual({ narrowing: 85, exercise: 0 });
    expect(labDemo(LAB_DEMO.exercise.to)).toEqual({ narrowing: 85, exercise: 1 });
    let prev = labDemo(0);
    for (let t = 0; t <= 18; t += 0.25) {
      const d = labDemo(t);
      expect(d.narrowing).toBeGreaterThanOrEqual(prev.narrowing);
      expect(d.exercise).toBeGreaterThanOrEqual(prev.exercise);
      if (d.exercise > 0) expect(d.narrowing).toBe(85);
      prev = d;
    }
  });
});

describe("story captions keep the honesty rules", () => {
  it("names the twin's recorded inputs, marks values not entered, and says the ECG is illustrative", () => {
    const text = storyCaption("twin", ctx());
    expect(text).toMatch(/recorded pulse \(80 bpm\)/);
    expect(text).toMatch(/EF \(55%, preserved\)/);
    expect(text).toMatch(/blood pressure \(130 mmHg\)/);
    expect(text).toMatch(/ST depression and T-wave inversion/);
    expect(text).toMatch(/illustrative, not the patient's tracing/);
    const median = storyCaption("twin", ctx({ twin: { ...twin, notRecorded: new Set(["ef"]) } }));
    expect(median).toMatch(/EF \(55%, preserved, not entered: cohort median\)/);
    expect(storyCaption("twin", ctx({ result: null }))).toMatch(/typical values/);
  });

  it("calls the anatomy a reference, the streams data not anatomy, and the attributions associations", () => {
    expect(storyCaption("chest", ctx())).toMatch(/reference heart.*not this patient's imaging/);
    const r = storyCaption("reasoning", ctx());
    expect(r).toMatch(/Overall CAD estimate >95%/);
    expect(r).toMatch(/data, not anatomy/);
    expect(r).toMatch(/not causes/);
    expect(storyCaption("reasoning", ctx({ whatIf: true }))).toMatch(/what-if scenario/);
  });

  it("says the three artery estimates are independent, not a culprit ranking", () => {
    const a = storyCaption("arteries", ctx());
    expect(a).toMatch(/LAD 93%, LCX 88%, RCA 83%/);
    expect(a).toMatch(/Shown: the LAD \(highest band\)/);
    expect(a).toMatch(/independent, not a culprit ranking/);
    expect(storyCaption("arteries", ctx({ result: null, top: null }))).toMatch(/Run an estimate/);
  });

  it("labels the lab a teaching sandbox on a generic narrowing, with the model's numbers", () => {
    const patient = { hr: 80, sbp: 130, age: 68 };
    const rest = storyCaption("lab", ctx({ lab: labState("lad", 0.5, 0, patient) }));
    expect(rest).toMatch(/Teaching sandbox: textbook physiology on a generic LAD narrowing, not this patient's artery/);
    const peak = labState("lad", 0.85, 1, patient);
    const text = storyCaption("lab", ctx({ lab: peak }));
    expect(text).toContain(`${Math.round(peak.ischaemia * 100)}% of its territory's need is unmet`);
  });

  it("closes with reference anatomy, the referral cohort, AUCs with CIs and decision support only", () => {
    expect(HONESTY.anatomy).toMatch(/BodyParts3D/);
    expect(HONESTY.cohort).toMatch(/single-centre referral cohort: 303 patients/);
    expect(HONESTY.use).toMatch(/Decision support/);
    const metrics = { metrics: { cad: { cv: { roc_auc: { mean: 0.943 } } }, lad: { cv: { roc_auc: { mean: 0.856 } } }, lcx: { cv: { roc_auc: { mean: 0.761 } } }, rca: { cv: { roc_auc: { mean: 0.739 } } } } } as unknown as MetricsResponse;
    const rows = honestyAucs(metrics);
    expect(rows.map((r) => [r.label, r.auc?.toFixed(2), r.ci.map((x) => x.toFixed(2)).join("–")])).toEqual([
      ["CAD", "0.94", "0.91–0.97"], ["LAD", "0.86", "0.82–0.91"], ["LCX", "0.76", "0.71–0.81"], ["RCA", "0.74", "0.70–0.81"],
    ]);
    expect(honestyAucs(null).every((r) => r.auc === null)).toBe(true);
  });
});
