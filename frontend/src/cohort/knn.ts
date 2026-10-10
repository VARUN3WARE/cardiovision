/**
 * Twin cohort: the most similar real patients among the 303 in the dataset.
 *
 * Data: cohort.json, written offline by backend/make_cohort.py (see its docstring). Similarity is the
 * Euclidean distance between weighted, standardised model inputs: each of the 55 encoded inputs (the
 * raw `feature_values` of /explain; VHD re-coded to severity as the models do) is z-scored over the
 * cohort and multiplied by the CAD model's mean |attribution|, so an input counts as much as the model
 * uses it. Inputs that were not entered (the API fills them in and lists them in `imputed`) are left out
 * of the distance and of the own-record test: a patient is compared on what was actually recorded. The
 * arithmetic matches the Python script operation for operation, so both find the same neighbours
 * (knn.test.ts checks this against the script's own answers).
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
  /** API schema field -> dataset column, where they differ (to map the response's `imputed`). */
  schema_to_column: Record<string, string>;
  /** Recognising the patient's own record (see matchRecord). */
  identity: {
    /** Inputs closer than this many standard deviations count as the same value. */
    tolerance_sd: number;
    /** Minimum specificity of the agreeing inputs (bits) for an exact match, and for a changed record. */
    exact_bits: number;
    edited_bits: number;
    /** At most this many compared inputs may differ for "this patient's record with inputs changed". */
    max_changed: number;
    /** Inputs the API derives from others, compared through those when they are given (BMI: weight, height). */
    via: Record<string, string[]>;
    /** Per column: -log2 of the chance that two different cohort patients agree on it. */
    bits: number[];
  };
  /** "Close": the k-th most similar within `distance`, as for `quantile` of the dataset's own patients. */
  closeness: { k: number; quantile: number; distance: number };
  /** The binary input that splits the map (the one the CAD model weighs most), or null if none does. */
  groups: {
    column: string;
    levels: { value: number; n: number; cad: number }[];
    neighbours_same: number;
    map_neighbours_same: number;
  } | null;
  /** Neighbours computed by the Python script (for the unit test). */
  check: { i: number; neighbours: number[] }[];
  patients: CohortPatient[];
}

/** The cohort pre-scaled into the similarity space (built once). */
export interface Space {
  data: CohortData;
  /** weight / sd per column (0 for constant columns). */
  scale: Float64Array;
  /** 1 / sd per column (0 for constant columns): unweighted z-units, for the same-value test. */
  inv: Float64Array;
  recodeAt: (Record<string, number> | undefined)[];
  /** n x d weighted vectors, row-major. */
  rows: Float64Array;
  /** n x d re-coded input values, row-major (for the same-value test). */
  values: Float64Array;
  /** Per column: the columns it is compared through when they are compared, if any. */
  via: (number[] | undefined)[];
  /** Closeness limits already computed, by compared-inputs pattern. */
  limits: Map<string, number>;
}

export interface Neighbour {
  /** Dataset row index. */
  index: number;
  distance: number;
}

/** A dataset record that is this patient, possibly with a few inputs changed (e.g. an edited preset). */
export interface RecordMatch {
  index: number;
  /** Columns whose values differ from the record (empty: the same values for every compared input). */
  changed: string[];
}

export interface Similar {
  /** The k most similar other patients, most similar first. */
  neighbours: Neighbour[];
  /** Dataset row with the same values for every compared input (left out of `neighbours`), if any. */
  self: number | null;
  /** This patient's dataset record with a few inputs changed (also left out of `neighbours`), if any. */
  edited: RecordMatch | null;
}

/** Which inputs a comparison uses: the entered ones (inputs filled in by the API are left out). */
export interface Compared {
  /** Per column: 1 = compared, 0 = left out. */
  use: Uint8Array;
  /** Informative inputs compared (constant columns are not counted). */
  n: number;
  /** Informative inputs left out because they were not entered. */
  skipped: number;
}

export function buildSpace(data: CohortData): Space {
  const d = data.columns.length;
  const n = data.patients.length;
  const scale = new Float64Array(d);
  const inv = new Float64Array(d);
  for (let j = 0; j < d; j++) {
    const sd = data.std[j];
    scale[j] = sd > 0 ? data.weight[j] / sd : 0;
    inv[j] = sd > 0 ? 1 / sd : 0;
  }
  const recodeAt = data.columns.map((c) => data.recode[c]);
  const via = data.columns.map((c) => data.identity.via[c]?.map((s) => data.columns.indexOf(s)).filter((j) => j >= 0));
  const space: Space = {
    data, scale, inv, recodeAt, via, rows: new Float64Array(n * d), values: new Float64Array(n * d), limits: new Map(),
  };
  data.patients.forEach((p, i) => {
    scaleInputs(space, p.x, space.rows.subarray(i * d, (i + 1) * d));
    for (let j = 0; j < d; j++) space.values[i * d + j] = recoded(space, j, p.x[j]);
  });
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

/** The inputs to compare: all except those the API filled in (`imputed` holds schema field names). */
export function comparedInputs(space: Space, imputed: readonly string[] = []): Compared {
  const { columns, schema_to_column: toColumn } = space.data;
  const filled = new Set(imputed.map((k) => toColumn[k] ?? k));
  const use = new Uint8Array(columns.length);
  let n = 0;
  let skipped = 0;
  columns.forEach((c, j) => {
    const informative = space.inv[j] > 0;
    if (filled.has(c)) {
      if (informative) skipped++;
      return;
    }
    use[j] = 1;
    if (informative) n++;
  });
  return { use, n, skipped };
}

/**
 * The dataset record that is this patient, if one is: it agrees on every compared input (within the
 * tolerance), or on all but `max_changed` of them (this patient's record with those inputs changed).
 * The agreeing inputs must be specific enough (their summed `bits` at least `exact_bits`, or
 * `edited_bits` for a changed record) and no other record may qualify, so a few entered inputs (say
 * age and sex) never single anyone out. BMI and the obesity flag are compared through weight and
 * height when those are compared (the API derives them).
 */
export function matchRecord(space: Space, raw: ArrayLike<number>, use?: Uint8Array): RecordMatch | null {
  const { columns, patients, identity: id } = space.data;
  const d = columns.length;
  const compare = new Uint8Array(d);
  for (let j = 0; j < d; j++) {
    const src = space.via[j];
    const throughSources = !!src && src.length > 0 && src.every((s) => !use || use[s] === 1);
    compare[j] = space.inv[j] > 0 && (!use || use[j] === 1) && !throughSources ? 1 : 0;
  }
  const q = Float64Array.from({ length: d }, (_, j) => recoded(space, j, raw[j]));
  let found: RecordMatch | null = null;
  for (let i = 0; i < patients.length; i++) {
    const o = i * d;
    const changed: number[] = [];
    let bits = 0;
    for (let j = 0; j < d && changed.length <= id.max_changed; j++) {
      if (!compare[j]) continue;
      if (Math.abs(space.values[o + j] - q[j]) * space.inv[j] >= id.tolerance_sd) changed.push(j);
      else bits += id.bits[j];
    }
    if (changed.length > id.max_changed || bits < (changed.length ? id.edited_bits : id.exact_bits)) continue;
    if (found) return null; // two records qualify: not specific enough to say which
    found = { index: i, changed: changed.map((j) => columns[j]) };
  }
  return found;
}

/**
 * The k most similar dataset patients to these encoded inputs (ties: lower row index first), over the
 * compared inputs only (`use`; default all). This patient's own record, if recognised, is left out.
 */
export function findSimilar(space: Space, raw: ArrayLike<number>, k = space.data.k, use?: Uint8Array): Similar {
  const d = space.scale.length;
  const n = space.data.patients.length;
  const u = scaleInputs(space, raw);
  const d2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * d;
    for (let j = 0; j < d; j++) {
      if (use && !use[j]) continue;
      const diff = space.rows[o + j] - u[j];
      s += diff * diff;
    }
    d2[i] = s;
  }
  const match = matchRecord(space, raw, use);
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => d2[a] - d2[b] || a - b);
  const neighbours = order
    .filter((i) => i !== match?.index)
    .slice(0, k)
    .map((index) => ({ index, distance: Math.sqrt(d2[index]) }));
  const exact = match !== null && match.changed.length === 0;
  return { neighbours, self: exact ? match.index : null, edited: match && !exact ? match : null };
}

/** Each dataset patient's distance to its k-th most similar other patient, over the compared inputs. */
export function kthDistances(space: Space, k = space.data.closeness.k, use?: Uint8Array): Float64Array {
  const d = space.scale.length;
  const n = space.data.patients.length;
  const { rows } = space;
  const cols = Int32Array.from({ length: d }, (_, j) => j).filter((j) => (!use || use[j] === 1) && space.scale[j] !== 0);
  const m = cols.length;
  const D = new Float64Array(n * n);
  for (let a = 0; a < n; a++) {
    const oa = a * d;
    for (let b = a + 1; b < n; b++) {
      const ob = b * d;
      let s = 0;
      for (let c = 0; c < m; c++) {
        const diff = rows[oa + cols[c]] - rows[ob + cols[c]];
        s += diff * diff;
      }
      D[a * n + b] = s;
      D[b * n + a] = s;
    }
  }
  const out = new Float64Array(n);
  const others = new Float64Array(n - 1);
  for (let a = 0; a < n; a++) {
    let t = 0;
    for (let b = 0; b < n; b++) if (b !== a) others[t++] = D[a * n + b];
    others.sort();
    out[a] = Math.sqrt(others[k - 1]);
  }
  return out;
}

/** Quantile of sorted values, linear interpolation (numpy's default). */
export function quantile(sorted: ArrayLike<number>, q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  const t = pos - lo;
  return sorted[lo] + (sorted[hi] - sorted[lo]) * t;
}

/**
 * The farthest the k-th most similar may be for the neighbours to count as close: as near as for
 * `closeness.quantile` (95%) of the dataset's own patients, on the same compared inputs. From
 * make_cohort.py when every input is compared; recomputed (once per pattern) when some are left out,
 * since distances over fewer inputs are shorter.
 */
export function closeLimit(space: Space, use?: Uint8Array): number {
  const { closeness } = space.data;
  const partial = !!use && space.scale.some((w, j) => w !== 0 && !use[j]);
  if (!partial) return closeness.distance;
  const key = Array.from(use!, (v, j) => (space.scale[j] !== 0 ? v : 1)).join("");
  let limit = space.limits.get(key);
  if (limit === undefined) {
    limit = quantile(kthDistances(space, closeness.k, use).sort(), closeness.quantile);
    space.limits.set(key, limit);
  }
  return limit;
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
