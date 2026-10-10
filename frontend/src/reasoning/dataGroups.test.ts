import { describe, expect, it } from "vitest";
import type { ShapExplanation } from "../api/client";
import { FEATURE_GROUPS } from "../featureLabels";
import { DATA_GROUPS, DERIVED_GROUP, dataGroupOf, groupAttributions, groupWeight } from "./dataGroups";
import presets from "./__fixtures__/presetExplanations.json";

type Fixture = Record<string, { probabilities: Record<string, number>; explanations: Record<string, ShapExplanation> }>;
const PRESETS = presets as unknown as Fixture;

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

/** Every column lands in exactly one group, and the group sums add back to the exact log-odds. */
function checkExact(exp: ShapExplanation) {
  const g = groupAttributions(exp);
  const members = g.groups.flatMap((x) => x.findings.map((f) => f.name));
  expect(members.length).toBe(exp.feature_names.length);
  expect(new Set(members)).toEqual(new Set(exp.feature_names));
  const direct = exp.base_value + sum(exp.shap_values);
  expect(Math.abs(g.base + sum(g.groups.map((x) => x.sum)) - direct)).toBeLessThan(1e-9);
  expect(Math.abs(g.logOdds - direct)).toBeLessThan(1e-9);
  return g;
}

// Deterministic pseudo-random numbers (no flaky tests).
function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

describe("data-type groups", () => {
  it("assigns every known column to exactly one named group", () => {
    const raw = FEATURE_GROUPS.flatMap(([, cols]) => cols);
    const all = [...raw, ...Object.keys(DERIVED_GROUP)];
    expect(new Set(all).size).toBe(all.length); // no column listed twice
    for (const c of all) expect(DATA_GROUPS.map((d) => d.id)).toContain(dataGroupOf(c));
    expect(dataGroupOf("Age")).toBe("history");
    expect(dataGroupOf("BP")).toBe("symptoms");
    expect(dataGroupOf("Systolic BP 140+")).toBe("symptoms");
    expect(dataGroupOf("ECG abnormality count")).toBe("ecg");
    expect(dataGroupOf("RWMA present")).toBe("echo");
    expect(dataGroupOf("AIP log(TG/HDL)")).toBe("labs");
    expect(dataGroupOf("DF pretest prob")).toBe("pretest");
  });

  it("sums exactly for synthetic attributions, including unknown columns", () => {
    const r = rng(7);
    const pool = [...FEATURE_GROUPS.flatMap(([, c]) => c), ...Object.keys(DERIVED_GROUP), "Some future column"];
    for (let trial = 0; trial < 200; trial++) {
      const names = pool.filter(() => r() < 0.7);
      const exp: ShapExplanation = {
        base_value: (r() - 0.5) * 6,
        feature_names: names,
        shap_values: names.map(() => (r() - 0.5) * (r() < 0.1 ? 40 : 2)),
        feature_values: names.map(() => r() * 100),
        derived: names.map((n) => n in DERIVED_GROUP),
      };
      const g = checkExact(exp);
      const other = g.groups.find((x) => x.id === "other");
      expect(!!other).toBe(names.includes("Some future column"));
    }
  });

  it("sums exactly for every dataset preset and model, and matches the model's probability", () => {
    let n = 0;
    for (const [id, p] of Object.entries(PRESETS)) {
      for (const [target, exp] of Object.entries(p.explanations)) {
        const g = checkExact(exp);
        expect(g.groups.some((x) => x.id === "other"), `${id}/${target} has unmapped columns`).toBe(false);
        expect(Math.abs(1 / (1 + Math.exp(-g.logOdds)) - p.probabilities[target])).toBeLessThan(1e-6);
        // Findings inside a group are ranked by magnitude.
        for (const x of g.groups) {
          for (let i = 1; i < x.findings.length; i++) {
            expect(Math.abs(x.findings[i - 1].shap)).toBeGreaterThanOrEqual(Math.abs(x.findings[i].shap));
          }
        }
        n++;
      }
    }
    expect(n).toBe(20);
  });

  it("scales bars and streams in proportion to |log-odds|", () => {
    expect(groupWeight(1)).toBeCloseTo(2 * groupWeight(0.5), 12);
    expect(groupWeight(-1)).toBe(groupWeight(1));
    expect(groupWeight(100)).toBe(1);
  });
});
