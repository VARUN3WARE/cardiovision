/**
 * Twin cohort: the most similar real patients among the 303 in the dataset.
 *
 * Data: cohort.json, written offline by backend/make_cohort.py (see its docstring). Similarity is the
 * Euclidean distance between weighted, standardised model inputs: each of the 55 encoded inputs (the
 * raw `feature_values` of /explain; VHD re-coded to severity as the models do) is z-scored over the
 * cohort and multiplied by the CAD model's mean |attribution|, so an input counts as much as the model
 * uses it. The arithmetic matches the Python script operation for operation, so both find the same
 * neighbours (knn.test.ts checks this against the script's own answers).
 */
import type { ShapExplanation } from "../api/client";

export interface CohortPatient {
  /** Dataset row index (also the position in `patients`). */
  i: number;
  /** Position on the 3D map (t-SNE of the similarity space; ~unit radius). */
  xyz: [number, number, number];
  /** The 55 encoded inputs, in `columns` order. */
  x: number[];
  /** Angiography per target (1 = CAD / >=50% narrowing), in `targets` order. */
  angio: number[];
  /** Mean held-out (out-of-fold) probability per target, in `targets` order. */
  p: number[];
  age: number;
  sex: "M" | "F";
}

export interface CohortData {
  n: number;
  k: number;
  targets: string[];
  columns: string[];
  mean: number[];
  std: number[];
  weight: number[];
  /** Per column: encoded code -> value used for similarity (applied before standardising). */
  recode: Record<string, Record<string, number>>;
  embedding: { method: string; neighbour_retention: number; [key: string]: number | string };
  knn_loo_auc: Record<string, number>;
  /** Neighbours computed by the Python script (for the unit test). */
  check: { i: number; neighbours: number[] }[];
  patients: CohortPatient[];
}

/** The cohort pre-scaled into the similarity space (built once). */
export interface Space {
  data: CohortData;
  /** weight / sd per column (0 for constant columns). */
  scale: Float64Array;
  /** 1 / sd per column (0 for constant columns): unweighted z-units, for the identical-patient test. */
  inv: Float64Array;
  recodeAt: (Record<string, number> | undefined)[];
  /** n x d weighted vectors, row-major. */
  rows: Float64Array;
}

export interface Neighbour {
  /** Dataset row index. */
  index: number;
  distance: number;
}

export interface Similar {
  /** The k most similar other patients, most similar first. */
  neighbours: Neighbour[];
  /** Dataset row with the same recorded values as the query (left out of `neighbours`), if any. */
  self: number | null;
}

/** Inputs within this many standard deviations on every column count as the same recorded values. */
export const SAME_TOLERANCE = 0.02;

export function buildSpace(data: CohortData): Space {
  const d = data.columns.length;
  const scale = new Float64Array(d);
  const inv = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    const sd = data.std[j];
    scale[j] = sd > 0 ? data.weight[j] / sd : 0;
    inv[j] = sd > 0 ? 1 / sd : 0;
  }
  const recodeAt = data.columns.map((c) => data.recode[c]);
  const space: Space = { data, scale, inv, recodeAt, rows: new Float64Array(data.patients.length * d) };
  data.patients.forEach((p, i) => scaleInputs(space, p.x, space.rows.subarray(i * d, (i + 1) * d)));
  return space;
}

/** Encoded value -> similarity value (VHD severity order); unknown codes pass through. */
function recoded(space: Space, j: number, v: number): number {
  const map = space.recodeAt[j];
  return map && String(v) in map ? map[String(v)] : v;
}

/** Encoded inputs (in `columns` order) -> weighted standardised vector. */
export function scaleInputs(space: Space, raw: ArrayLike<number>, out: Float64Array = new Float64Array(space.scale.length)): Float64Array {
  const { mean } = space.data;
  for (let j = 0; j < out.length; j++) out[j] = (recoded(space, j, raw[j]) - mean[j]) * space.scale[j];
  return out;
}

/** The patient's encoded inputs as the CAD model saw them (raw columns of the explanation), or null. */
export function inputsFromExplanation(exp: ShapExplanation | undefined, columns: readonly string[]): number[] | null {
  if (!exp) return null;
  const byName = new Map<string, number>();
  exp.feature_names.forEach((n, i) => {
    if (!exp.derived[i]) byName.set(n, exp.feature_values[i]);
  });
  const out: number[] = [];
  for (const c of columns) {
    const v = byName.get(c);
    if (v === undefined || !Number.isFinite(v)) return null;
    out.push(v);
  }
  return out;
}

/** True when every input agrees within SAME_TOLERANCE standard deviations (unweighted). */
export function sameInputs(space: Space, a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  for (let j = 0; j < space.inv.length; j++) {
    if (Math.abs(recoded(space, j, a[j]) - recoded(space, j, b[j])) * space.inv[j] >= SAME_TOLERANCE) return false;
  }
  return true;
}

/** The k most similar dataset patients to these encoded inputs (ties: lower row index first). */
export function findSimilar(space: Space, raw: ArrayLike<number>, k = space.data.k): Similar {
  const d = space.scale.length;
  const n = space.data.patients.length;
  const u = scaleInputs(space, raw);
  const d2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * d;
    for (let j = 0; j < d; j++) {
      const diff = space.rows[o + j] - u[j];
      s += diff * diff;
    }
    d2[i] = s;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => d2[a] - d2[b] || a - b);
  const self = sameInputs(space, space.data.patients[order[0]].x, raw) ? order[0] : null;
  const neighbours = order
    .filter((i) => i !== self)
    .slice(0, k)
    .map((index) => ({ index, distance: Math.sqrt(d2[index]) }));
  return { neighbours, self };
}

/**
 * Where the patient sits on the map: its own point if it is a dataset patient, otherwise the
 * inverse-distance-weighted mean of its neighbours' positions.
 */
export function placeOnMap(space: Space, similar: Similar): [number, number, number] {
  const { patients } = space.data;
  if (similar.self !== null) return [...patients[similar.self].xyz];
  let wSum = 0;
  const out: [number, number, number] = [0, 0, 0];
  for (const { index, distance } of similar.neighbours) {
    const w = 1 / Math.max(distance, 1e-6);
    const xyz = patients[index].xyz;
    for (let a = 0; a < 3; a++) out[a] += w * xyz[a];
    wSum += w;
  }
  return wSum > 0 ? (out.map((v) => v / wSum) as [number, number, number]) : [0, 0, 0];
}

export interface CohortSummary {
  k: number;
  /** Neighbours positive on angiography, per target. */
  counts: Record<string, number>;
  /** Mean held-out probability over the neighbours, per target. */
  meanHeldOut: Record<string, number>;
  /** Neighbours labelled "no CAD" overall but with a >=50% narrowed artery (a source-label inconsistency). */
  inconsistent: number[];
}

export function summarise(space: Space, neighbours: readonly Neighbour[]): CohortSummary {
  const { targets, patients } = space.data;
  const counts: Record<string, number> = {};
  const meanHeldOut: Record<string, number> = {};
  targets.forEach((t, j) => {
    counts[t] = neighbours.reduce((s, nb) => s + patients[nb.index].angio[j], 0);
    meanHeldOut[t] = neighbours.length ? neighbours.reduce((s, nb) => s + patients[nb.index].p[j], 0) / neighbours.length : 0;
  });
  const cad = targets.indexOf("cad");
  const inconsistent = cad < 0 ? [] : neighbours
    .map((nb) => nb.index)
    .filter((i) => patients[i].angio[cad] === 0 && patients[i].angio.some((v, j) => j !== cad && v === 1));
  return { k: neighbours.length, counts, meanHeldOut, inconsistent };
}

/** Share of all cohort patients positive on angiography, per target. */
export function cohortRates(data: CohortData): Record<string, number> {
  const out: Record<string, number> = {};
  data.targets.forEach((t, j) => {
    out[t] = data.patients.reduce((s, p) => s + p.angio[j], 0) / data.patients.length;
  });
  return out;
}
