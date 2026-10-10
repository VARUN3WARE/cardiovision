import { describe, expect, it } from "vitest";
import type { ShapExplanation } from "../api/client";
import raw from "./cohort.json";
import fixtures from "./__fixtures__/cadExplanations.json";
import {
  buildSpace, cohortRates, findSimilar, inputsFromExplanation, placeOnMap, sameInputs, scaleInputs, summarise,
  type CohortData,
} from "./knn";

// CAD explanations returned by the live /explain endpoint for the presets and the demo patient.
type Fixture = Record<string, { patient_index: number | null; imputed: string[]; cad: ShapExplanation }>;
const FIX = fixtures as unknown as Fixture;
const DATA = raw as unknown as CohortData;
const SPACE = buildSpace(DATA);
const col = (name: string) => DATA.columns.indexOf(name);

/** Textbook form of the distance, written independently of knn.ts. */
function naiveDistance(a: number[], b: number[]): number {
  const vhd = col("VHD");
  const sev = (v: number, j: number) => (j === vhd ? DATA.recode.VHD[String(v)] : v);
  let s = 0;
  DATA.columns.forEach((_, j) => {
    if (DATA.std[j] === 0) return;
    const za = (sev(a[j], j) - DATA.mean[j]) / DATA.std[j];
    const zb = (sev(b[j], j) - DATA.mean[j]) / DATA.std[j];
    s += (DATA.weight[j] * (za - zb)) ** 2;
  });
  return Math.sqrt(s);
}

describe("cohort data", () => {
  it("has 303 patients with consistent shapes", () => {
    expect(DATA.patients).toHaveLength(303);
    expect(DATA.columns).toHaveLength(55);
    for (const arr of [DATA.mean, DATA.std, DATA.weight]) expect(arr).toHaveLength(55);
    DATA.patients.forEach((p, i) => {
      expect(p.i).toBe(i);
      expect(p.x).toHaveLength(55);
      expect(p.angio).toHaveLength(DATA.targets.length);
      expect(p.p.every((v) => v >= 0 && v <= 1)).toBe(true);
    });
    expect(DATA.weight[col("Exertional CP")]).toBe(0); // constant column
    expect(Math.round(cohortRates(DATA).cad * 303)).toBe(216); // 216 of 303 had CAD
  });
});

describe("nearest neighbours", () => {
  it("finds the same neighbours as the Python script", () => {
    expect(DATA.check.length).toBeGreaterThanOrEqual(5);
    for (const { i, neighbours } of DATA.check) {
      const s = findSimilar(SPACE, DATA.patients[i].x);
      expect(s.self).toBe(i);
      expect(s.neighbours.map((n) => n.index)).toEqual(neighbours);
    }
  });

  it("recognises preset patients from a real /explain response and leaves them out", () => {
    for (const [id, f] of Object.entries(FIX)) {
      const x = inputsFromExplanation(f.cad, DATA.columns);
      expect(x, id).not.toBeNull();
      const s = findSimilar(SPACE, x!);
      expect(s.self, id).toBe(f.patient_index);
      expect(s.neighbours).toHaveLength(10);
      if (f.patient_index !== null) {
        expect(s.neighbours.map((n) => n.index)).toEqual(DATA.check.find((c) => c.i === f.patient_index)!.neighbours);
        expect(placeOnMap(SPACE, s)).toEqual(DATA.patients[f.patient_index].xyz);
      }
    }
  });

  it("sorts by the textbook weighted distance for a new patient", () => {
    const x = inputsFromExplanation(FIX.demo.cad, DATA.columns)!;
    const s = findSimilar(SPACE, x, 303);
    expect(s.self).toBeNull();
    expect(s.neighbours).toHaveLength(303);
    s.neighbours.forEach((n, r) => {
      expect(n.distance).toBeCloseTo(naiveDistance(x, DATA.patients[n.index].x), 9);
      if (r > 0) expect(n.distance).toBeGreaterThanOrEqual(s.neighbours[r - 1].distance);
    });
  });

  it("treats a slightly changed dataset patient as a new patient whose nearest is the original", () => {
    const x = [...DATA.patients[171].x];
    x[col("Age")] += 3;
    const s = findSimilar(SPACE, x);
    expect(s.self).toBeNull();
    expect(s.neighbours[0].index).toBe(171);
    expect(s.neighbours[0].distance).toBeCloseTo(naiveDistance(x, DATA.patients[171].x), 9);
  });

  it("uses the VHD severity order, not the alphabetical codes", () => {
    const base = [...DATA.patients[0].x];
    const withVhd = (code: number) => {
      const x = [...base];
      x[col("VHD")] = code;
      return scaleInputs(SPACE, x)[col("VHD")];
    };
    const [moderate, none, severe, mild] = [0, 1, 2, 3].map(withVhd); // alphabetical codes
    expect(none).toBeLessThan(mild);
    expect(mild).toBeLessThan(moderate);
    expect(moderate).toBeLessThan(severe);
  });

  it("rejects explanations that lack an input", () => {
    const e = FIX.low.cad;
    const drop = e.feature_names.indexOf("Age");
    const partial: ShapExplanation = {
      ...e,
      feature_names: e.feature_names.filter((_, i) => i !== drop),
      feature_values: e.feature_values.filter((_, i) => i !== drop),
      shap_values: e.shap_values.filter((_, i) => i !== drop),
      derived: e.derived.filter((_, i) => i !== drop),
    };
    expect(inputsFromExplanation(partial, DATA.columns)).toBeNull();
    expect(sameInputs(SPACE, DATA.patients[5].x, DATA.patients[5].x)).toBe(true);
  });
});

describe("placement and summary", () => {
  it("places a new patient at the inverse-distance-weighted mean of its neighbours", () => {
    const s = findSimilar(SPACE, inputsFromExplanation(FIX.demo.cad, DATA.columns)!);
    const at = placeOnMap(SPACE, s);
    const w = s.neighbours.map((n) => 1 / n.distance);
    const total = w.reduce((a, b) => a + b, 0);
    for (let a = 0; a < 3; a++) {
      const expected = s.neighbours.reduce((acc, n, r) => acc + w[r] * DATA.patients[n.index].xyz[a], 0) / total;
      expect(at[a]).toBeCloseTo(expected, 12);
      const vals = s.neighbours.map((n) => DATA.patients[n.index].xyz[a]);
      expect(at[a]).toBeGreaterThanOrEqual(Math.min(...vals));
      expect(at[a]).toBeLessThanOrEqual(Math.max(...vals));
    }
  });

  it("counts outcomes and flags the dataset's inconsistent label", () => {
    const nb = [93, 5, 3].map((index) => ({ index, distance: 1 }));
    const s = summarise(SPACE, nb);
    const t = (name: string) => DATA.targets.indexOf(name);
    for (const name of DATA.targets) {
      expect(s.counts[name]).toBe(nb.reduce((acc, n) => acc + DATA.patients[n.index].angio[t(name)], 0));
    }
    // Row 93 is labelled Normal overall but has >=50% LAD narrowing in the source data.
    expect(DATA.patients[93].angio[t("cad")]).toBe(0);
    expect(DATA.patients[93].angio[t("lad")]).toBe(1);
    expect(s.inconsistent).toEqual([93]);
  });

  it("gives sensible neighbourhoods for the low-risk and three-vessel presets", () => {
    const low = summarise(SPACE, findSimilar(SPACE, DATA.patients[FIX.low.patient_index!].x).neighbours);
    const triple = summarise(SPACE, findSimilar(SPACE, DATA.patients[FIX.triple.patient_index!].x).neighbours);
    expect(low.counts.cad).toBeLessThanOrEqual(3);
    expect(triple.counts.cad).toBeGreaterThanOrEqual(8);
    expect(triple.meanHeldOut.cad).toBeGreaterThan(low.meanHeldOut.cad);
  });
});
