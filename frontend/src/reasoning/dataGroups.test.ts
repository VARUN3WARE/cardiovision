import { describe, expect, it } from "vitest";
import type { ShapExplanation } from "../api/client";
import { FEATURE_GROUPS } from "../featureLabels";
import {
  DATA_GROUPS, DERIVED_GROUP, dataGroupOf, filledNote, groupAttributions, groupWeight, typicalColumns,
} from "./dataGroups";
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

  it("marks inputs the form left empty (mapping schema keys to columns) with how the API filled them", () => {
    const exp: ShapExplanation = {
      base_value: 0.5,
      feature_names: ["EF-TTE", "BP", "Age", "EF below 50", "Systolic BP 140+", "eGFR", "Region RWMA", "Q Wave"],
      shap_values: [0.4, -0.1, 0.2, 0.3, -0.05, 0.1, 0.6, -0.2],
      feature_values: [55, 125, 60, 0, 0, 80, 0, 0],
      derived: [false, false, false, true, true, true, false, false],
    };
    const info = {
      schema_to_column: { EF_TTE: "EF-TTE", RegionRWMA: "Region RWMA", QWave: "Q Wave" },
      derived_sources: { "EF below 50": ["EF-TTE"], "Systolic BP 140+": ["BP"], eGFR: ["CR", "Age", "Sex"] },
      feature_info: {
        "EF-TTE": { type: "numeric", default: 55 }, BP: { type: "numeric", default: 130 },
        "Q Wave": { type: "binary", default: 0 }, "Region RWMA": { type: "numeric", default: 0 },
      },
    } as unknown as Parameters<typeof typicalColumns>[2];
    const filled = typicalColumns(exp, ["EF_TTE", "BP", "CR", "QWave"], info);
    // Schema keys map to columns; a calculated score counts only if all its inputs were filled in.
    expect([...filled.keys()].sort()).toEqual(["BP", "EF below 50", "EF-TTE", "Q Wave", "Systolic BP 140+"]);
    // Measurements take the cohort median, findings count as absent (backend/app/schemas.py).
    expect(filled.get("EF-TTE")).toBe("median");
    expect(filled.get("Q Wave")).toBe("absent");
    expect(filled.get("EF below 50")).toBe("derived");
    // Without /features, keys that differ from their column are not recognised (so ExplanationPanel and the
    // 3D view pass it, or fetch it).
    expect(typicalColumns(exp, ["EF_TTE"], null).has("EF-TTE")).toBe(false);

    const g = groupAttributions(exp, filled);
    expect(g.inputs).toBe(5); // raw columns only
    expect(g.filled).toBe(3); // EF-TTE, BP and Q Wave
    const echo = g.groups.find((x) => x.id === "echo")!;
    expect(echo.inputs).toBe(2);
    expect(echo.filled).toBe(1);
    expect(echo.findings.find((f) => f.name === "EF-TTE")?.fill).toBe("median");
    expect(echo.findings.find((f) => f.name === "Region RWMA")?.fill).toBeNull();
    expect(filledNote(g.filled, g.inputs)).toBe("3 of 5 inputs not entered");
    expect(filledNote(0, 4)).toBe("");
    // Flags never change the sums.
    checkExact(exp);
    expect(groupAttributions(exp).groups.every((x) => x.filled === 0)).toBe(true);
  });

  it("scales bars and streams in proportion to |log-odds|", () => {
    expect(groupWeight(1)).toBeCloseTo(2 * groupWeight(0.5), 12);
    expect(groupWeight(-1)).toBe(groupWeight(1));
    expect(groupWeight(100)).toBe(1);
  });
});
