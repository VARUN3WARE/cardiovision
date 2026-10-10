import { describe, expect, it } from "vitest";
import type { ShapExplanation } from "../api/client";
import raw from "./cohort.json";
import fixtures from "./__fixtures__/cadExplanations.json";
import {
  buildSpace, closeLimit, cohortRates, comparedInputs, findSimilar, inputsFromExplanation, kthDistances, matchRecord,
  placeOnMap, quantile, scaleInputs, summarise, type CohortData,
} from "./knn";

// CAD explanations returned by the live /explain endpoint for the presets and the demo patient, with every
// field sent (`low`, ...) and in Quick mode (`low_quick`, ...: only the quick fields; the API fills in the rest).
type Fixture = Record<string, { patient_index: number | null; imputed: string[]; cad: ShapExplanation }>;
const FIX = fixtures as unknown as Fixture;
const DATA = raw as unknown as CohortData;
const SPACE = buildSpace(DATA);
const col = (name: string) => DATA.columns.indexOf(name);
const inputs = (id: string) => inputsFromExplanation(FIX[id].cad, DATA.columns)!;
const compared = (id: string) => comparedInputs(SPACE, FIX[id].imputed);
const PRESETS = ["low", "lad", "rca", "triple", "uncertain"];

/** Textbook form of the distance over the compared inputs, written independently of knn.ts. */
function naiveDistance(a: number[], b: number[], use?: Uint8Array): number {
  const vhd = col("VHD");
  const sev = (v: number, j: number) => (j === vhd ? DATA.recode.VHD[String(v)] : v);
  let s = 0;
  DATA.columns.forEach((_, j) => {
    if (DATA.std[j] === 0 || (use && !use[j])) return;
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
    for (const arr of [DATA.mean, DATA.std, DATA.weight, DATA.identity.bits]) expect(arr).toHaveLength(55);
    DATA.patients.forEach((p, i) => {
      expect(p.i).toBe(i);
      expect(p.x).toHaveLength(55);
      expect(p.angio).toHaveLength(DATA.targets.length);
      expect(p.p.every((v) => v >= 0 && v <= 1)).toBe(true);
    });
    expect(DATA.weight[col("Exertional CP")]).toBe(0); // constant column
    expect(Math.round(cohortRates(DATA).cad * 303)).toBe(216); // 216 of 303 had CAD
    for (const c of Object.values(DATA.schema_to_column)) expect(DATA.columns).toContain(c);
  });

  it("describes the map's two groups with the dataset's own numbers", () => {
    const g = DATA.groups!;
    expect(g.column).toBe("Typical Chest Pain");
    expect(DATA.weight[col(g.column)]).toBe(Math.max(...DATA.weight)); // the input the CAD model weighs most
    expect(g.levels.reduce((s, l) => s + l.n, 0)).toBe(303);
    const cad = DATA.targets.indexOf("cad");
    for (const l of g.levels) {
      const members = DATA.patients.filter((p) => p.x[col(g.column)] === l.value);
      expect(members).toHaveLength(l.n);
      expect(l.cad).toBeCloseTo(members.reduce((s, p) => s + p.angio[cad], 0) / l.n, 3);
    }
    expect(g.map_neighbours_same).toBeGreaterThanOrEqual(0.95);
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
      const s = findSimilar(SPACE, x!, DATA.k, compared(id).use);
      expect(s.self, id).toBe(f.patient_index);
      expect(s.edited, id).toBeNull();
      expect(s.neighbours).toHaveLength(10);
      if (f.patient_index !== null) {
        expect(s.neighbours.map((n) => n.index)).not.toContain(f.patient_index);
        expect(placeOnMap(SPACE, s)).toEqual(DATA.patients[f.patient_index].xyz);
        if (f.imputed.length === 0) {
          expect(s.neighbours.map((n) => n.index)).toEqual(DATA.check.find((c) => c.i === f.patient_index)!.neighbours);
        }
      }
    }
  });

  it("preset fixture with non-quick inputs at defaults returns self = preset", () => {
    for (const id of PRESETS) {
      const f = FIX[`${id}_quick`];
      const cmp = compared(`${id}_quick`);
      expect(f.imputed).toHaveLength(31);
      expect([cmp.n, cmp.skipped]).toEqual([23, 31]); // 21 quick fields + BMI and its flag from weight and height
      const x = inputs(`${id}_quick`);
      const s = findSimilar(SPACE, x, DATA.k, cmp.use);
      expect(s.self, id).toBe(f.patient_index);
      expect(s.neighbours.map((n) => n.index), id).not.toContain(f.patient_index);
      // Before the fix (all 55 inputs, defaults included) the record was not recognised and was its own neighbour.
      const unmasked = findSimilar(SPACE, x);
      expect(unmasked.self, id).toBeNull();
      if (id === "triple" || id === "low") expect(unmasked.neighbours.map((n) => n.index), id).toContain(f.patient_index);
    }
  });

  it("measures distance over the entered inputs only", () => {
    for (const id of ["demo", "demo_quick"]) {
      const x = inputs(id);
      const { use } = compared(id);
      const s = findSimilar(SPACE, x, 303, use);
      expect(s.self).toBeNull();
      expect(s.neighbours).toHaveLength(303);
      s.neighbours.forEach((n, r) => {
        expect(n.distance).toBeCloseTo(naiveDistance(x, DATA.patients[n.index].x, use), 9);
        if (r > 0) expect(n.distance).toBeGreaterThanOrEqual(s.neighbours[r - 1].distance);
      });
    }
  });

  it("flags an edited dataset patient as its record with inputs changed and leaves it out", () => {
    for (const id of ["lad", "lad_quick"]) {
      const x = inputs(id);
      x[col("Age")] += 3;
      x[col("BP")] += 20;
      const s = findSimilar(SPACE, x, DATA.k, compared(id).use);
      expect(s.self, id).toBeNull();
      expect(s.edited, id).toEqual({ index: 171, changed: ["Age", "BP"] });
      expect(s.neighbours.map((n) => n.index), id).not.toContain(171);
      expect(summarise(SPACE, s.neighbours).k).toBe(10);
    }
  });

  it("counts a changed weight once (BMI and its flag follow weight and height)", () => {
    const x = inputs("lad");
    x[col("Weight")] += 12;
    x[col("BMI")] = x[col("Weight")] / (x[col("Length")] / 100) ** 2;
    x[col("Obesity")] = x[col("BMI")] >= 25 ? 1 : 0;
    expect(matchRecord(SPACE, x)).toEqual({ index: 171, changed: ["Weight"] });
  });

  it("recognises row 153 although the API recomputes its obesity flag from weight and height", () => {
    const x = [...DATA.patients[153].x];
    x[col("Obesity")] = 0; // 64 kg, 160 cm: BMI 24.999... in floating point, flag "N" (dataset: 25.0, "Y")
    expect(findSimilar(SPACE, x).self).toBe(153);
  });

  it("treats a patient with more than 3 changed inputs as a new patient", () => {
    const x = [...DATA.patients[171].x];
    for (const name of ["Age", "BP", "FBS", "TG"]) x[col(name)] += 0.5 * DATA.std[col(name)];
    const s = findSimilar(SPACE, x);
    expect(s.self).toBeNull();
    expect(s.edited).toBeNull();
    expect(s.neighbours[0].distance).toBeCloseTo(naiveDistance(x, DATA.patients[s.neighbours[0].index].x), 9);
  });

  it("never identifies anyone from a few entered inputs", () => {
    const use = new Uint8Array(DATA.columns.length);
    for (const name of ["Age", "Sex", "Typical Chest Pain", "DM"]) use[col(name)] = 1;
    for (const i of [143, 171, 173, 0, 302]) {
      const s = findSimilar(SPACE, DATA.patients[i].x, DATA.k, use);
      expect(s.self, String(i)).toBeNull();
      expect(s.edited, String(i)).toBeNull();
      expect(s.neighbours[0].distance).toBe(0); // the record is simply one of the patients with these values
    }
  });

  it("maps imputed schema fields to dataset columns", () => {
    const cmp = comparedInputs(SPACE, ["TypicalChestPain", "EF_TTE", "HB"]);
    for (const name of ["Typical Chest Pain", "EF-TTE", "HB"]) expect(cmp.use[col(name)]).toBe(0);
    expect(cmp.use[col("Age")]).toBe(1);
    expect([cmp.n, cmp.skipped]).toEqual([51, 3]);
    expect(comparedInputs(SPACE, []).n).toBe(54); // every informative input (Exertional CP is constant)
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
  });
});

describe("closeness", () => {
  it("reproduces the script's limit when every input is compared", () => {
    const kth = kthDistances(SPACE).sort();
    expect(quantile(kth, DATA.closeness.quantile)).toBeCloseTo(DATA.closeness.distance, 5);
    expect(closeLimit(SPACE)).toBe(DATA.closeness.distance);
    expect(closeLimit(SPACE, comparedInputs(SPACE, []).use)).toBe(DATA.closeness.distance);
  });

  it("recomputes it over the entered inputs only (shorter distances)", () => {
    const { use } = compared("triple_quick");
    const limit = closeLimit(SPACE, use);
    expect(limit).toBeLessThan(DATA.closeness.distance);
    expect(limit).toBeCloseTo(quantile(kthDistances(SPACE, 10, use).sort(), 0.95), 12);
    expect(closeLimit(SPACE, use)).toBe(limit); // cached
  });

  it("finds close neighbours for the presets but not for an extreme patient", () => {
    for (const id of [...PRESETS, ...PRESETS.map((p) => `${p}_quick`), "demo", "demo_quick"]) {
      const { use } = compared(id);
      const s = findSimilar(SPACE, inputs(id), DATA.k, use);
      expect(s.neighbours[9].distance, id).toBeLessThanOrEqual(closeLimit(SPACE, use));
    }
    const x = inputs("demo");
    const extreme = { Age: 86, "EF-TTE": 15, "Region RWMA": 4, FBS: 400, TG: 1000, CR: 6, ESR: 110, BP: 190, HB: 8, WBC: 18000 };
    for (const [name, v] of Object.entries(extreme)) x[col(name)] = v;
    const s = findSimilar(SPACE, x);
    expect(s.neighbours[9].distance).toBeGreaterThan(closeLimit(SPACE));
  });
});

describe("placement and summary", () => {
  it("places a new patient at the inverse-distance-weighted mean of its neighbours", () => {
    const s = findSimilar(SPACE, inputs("demo"));
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

  it("gives sensible neighbourhoods for the low-risk and three-vessel presets, full and quick", () => {
    for (const mode of ["", "_quick"]) {
      const summary = (id: string) => summarise(SPACE, findSimilar(SPACE, inputs(id + mode), DATA.k, compared(id + mode).use).neighbours);
      const low = summary("low");
      const triple = summary("triple");
      expect(low.counts.cad).toBeLessThanOrEqual(3);
      expect(triple.counts.cad).toBeGreaterThanOrEqual(8);
      expect(triple.meanHeldOut.cad).toBeGreaterThan(low.meanHeldOut.cad);
    }
  });
});
