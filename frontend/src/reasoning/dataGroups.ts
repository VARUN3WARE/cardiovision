/**
 * Splits one model's exact additive attributions (log-odds) into data-type groups, so "how much did the
 * ECG move this estimate?" has an exact answer: base + sum of group sums = base + sum of attributions =
 * the model's log-odds.
 *
 * Raw columns follow FEATURE_GROUPS (featureLabels.ts); demographics join history. Derived composites are
 * assigned to one group by a stated convention (DERIVED_GROUP), never split. These are associations in a
 * referral cohort, not causes. Inputs the form left empty were filled in by the API (backend/app/schemas.py: a
 * finding counts as absent, a measurement takes the training cohort's median): findings say so (typicalColumns),
 * so a filled-in value is never presented as the patient's finding.
 */
import type { FeaturesResponse, ShapExplanation } from "../api/client";
import { FEATURE_GROUPS, FEATURE_LABELS } from "../featureLabels";

export type DataGroupId = "history" | "symptoms" | "ecg" | "echo" | "labs" | "pretest" | "other";

export interface DataGroupDef {
  id: DataGroupId;
  label: string;
  /** Short name for the 3D towers. */
  short: string;
  /** Shortest name, for the one-line chips of the narrow 3D view. */
  chip: string;
}

/** Display order (also the tower order in 3D). "other" only appears if an unknown column turns up. */
export const DATA_GROUPS: readonly DataGroupDef[] = [
  { id: "history", label: "History & demographics", short: "History", chip: "History" },
  { id: "symptoms", label: "Symptoms & exam", short: "Symptoms", chip: "Symptoms" },
  { id: "ecg", label: "ECG", short: "ECG", chip: "ECG" },
  { id: "echo", label: "Echo", short: "Echo", chip: "Echo" },
  { id: "labs", label: "Labs", short: "Labs", chip: "Labs" },
  { id: "pretest", label: "Pre-test score", short: "Pre-test score", chip: "Pre-test" },
];
export const OTHER_GROUP: DataGroupDef = { id: "other", label: "Other inputs", short: "Other", chip: "Other" };

const BY_TITLE: Record<string, DataGroupId> = {
  Demographics: "history",
  "History & risk factors": "history",
  "Symptoms & examination": "symptoms",
  ECG: "ecg",
  Echocardiography: "echo",
  Laboratory: "labs",
};

/**
 * Convention for derived composites: each goes to the data type it is computed from. The pre-test group is
 * the Diamond-Forrester pre-test probability and its two ingredients as the model encodes them (chest pain
 * class; age x male).
 */
export const DERIVED_GROUP: Record<string, DataGroupId> = {
  "Risk factor count": "history",
  "ECG abnormality count": "ecg",
  "TG/HDL": "labs",
  "LDL/HDL": "labs",
  "AIP log(TG/HDL)": "labs",
  eGFR: "labs",
  NLR: "labs",
  "FBS 126+": "labs",
  "EF below 50": "echo",
  "RWMA present": "echo",
  "Systolic BP 140+": "symptoms",
  "DF pretest prob": "pretest",
  "Chest pain class": "pretest",
  "Age x Male": "pretest",
};

const GROUP_OF = new Map<string, DataGroupId>([
  ...FEATURE_GROUPS.flatMap(([title, cols]) => cols.map((c) => [c, BY_TITLE[title]] as [string, DataGroupId])),
  ...Object.entries(DERIVED_GROUP),
]);

/** The data-type group of a model column (unknown columns go to "other", so nothing is dropped). */
export function dataGroupOf(feature: string): DataGroupId {
  return GROUP_OF.get(feature) ?? "other";
}

export interface GroupFinding {
  name: string;
  label: string;
  value: number;
  shap: number;
  derived: boolean;
  /** Not entered: how the API filled it (for a calculated score: all its inputs were filled), else null. */
  fill: FillKind | null;
}

export interface GroupContribution extends DataGroupDef {
  /** Exact sum of the group's attributions, in log-odds. */
  sum: number;
  /** Members, largest |attribution| first. */
  findings: GroupFinding[];
  /** Inputs in the group (calculated scores excluded) and how many the form left empty. */
  inputs: number;
  filled: number;
}

export interface GroupedAttribution {
  base: number;
  /** base + sum of all attributions (= base + sum of group sums): the model's log-odds. */
  logOdds: number;
  /** Every group in display order ("other" only when non-empty). */
  groups: GroupContribution[];
  /** Inputs of this model (calculated scores excluded) and how many the form left empty. */
  inputs: number;
  filled: number;
}

/**
 * How the API filled an input the form left empty (backend/app/schemas.py): a measurement takes the training
 * cohort's median, a finding counts as absent (a category: its "none" level); a calculated score whose inputs
 * were all filled is "derived"; "default" when the column's type is unknown.
 */
export type FillKind = "median" | "absent" | "derived" | "default";
export const FILL_TEXT: Record<FillKind, string> = {
  median: "cohort median", absent: "not entered: absent", derived: "from defaults", default: "not entered",
};
export const FILL_SHORT: Record<FillKind, string> = { median: "median", absent: "absent", derived: "defaults", default: "default" };
export const FILL_RULE = "findings count as absent, measurements take the cohort median";

/**
 * Model columns whose value the API filled in, and how: the `imputed` schema keys mapped to columns
 * (FeaturesResponse.schema_to_column; unlisted keys are already column names), typed by feature_info. A
 * calculated score counts only when every input it is computed from was filled in.
 */
export function typicalColumns(
  exp: ShapExplanation,
  imputed: readonly string[],
  info: Pick<FeaturesResponse, "schema_to_column" | "derived_sources" | "feature_info"> | null,
): Map<string, FillKind> {
  const cols = new Set(imputed.map((k) => info?.schema_to_column[k] ?? k));
  const out = new Map<string, FillKind>();
  exp.feature_names.forEach((name, i) => {
    const sources = info?.derived_sources[name] ?? [];
    if (exp.derived?.[i]) {
      if (sources.length > 0 && sources.every((s) => cols.has(s))) out.set(name, "derived");
    } else if (cols.has(name)) {
      const type = info?.feature_info[name]?.type;
      out.set(name, type === "numeric" ? "median" : type === "binary" || type === "category" ? "absent" : "default");
    }
  });
  return out;
}

export function groupAttributions(exp: ShapExplanation, typical?: ReadonlyMap<string, FillKind> | null): GroupedAttribution {
  const defs = [...DATA_GROUPS, OTHER_GROUP];
  const byId = new Map<DataGroupId, GroupContribution>(
    defs.map((d) => [d.id, { ...d, sum: 0, findings: [], inputs: 0, filled: 0 }]),
  );
  let total = 0;
  let inputs = 0;
  let filled = 0;
  exp.feature_names.forEach((name, i) => {
    const shap = exp.shap_values[i];
    const g = byId.get(dataGroupOf(name))!;
    const derived = exp.derived?.[i] ?? false;
    const fill = typical?.get(name) ?? null;
    g.sum += shap;
    total += shap;
    if (!derived) {
      g.inputs += 1;
      inputs += 1;
      if (fill) {
        g.filled += 1;
        filled += 1;
      }
    }
    g.findings.push({
      name,
      label: FEATURE_LABELS[name]?.label ?? name,
      value: exp.feature_values[i],
      shap,
      derived,
      fill,
    });
  });
  for (const g of byId.values()) g.findings.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  const groups = defs.map((d) => byId.get(d.id)!).filter((g) => g.id !== "other" || g.findings.length > 0);
  return { base: exp.base_value, logOdds: exp.base_value + total, groups, inputs, filled };
}

/** "3 of 31 inputs not entered" (empty when every input was entered). */
export function filledNote(filled: number, inputs: number): string {
  return filled > 0 ? `${filled} of ${inputs} inputs not entered` : "";
}

/**
 * Log-odds that fills a group bar or the widest 3D stream. Fixed (not relative to the largest group) so
 * models and patients are comparable; across the five dataset presets group sums stay within about 2.2.
 */
export const GROUP_SCALE = 2.5;

/** Bar length / stream width as a fraction of the maximum: proportional to |sum| up to GROUP_SCALE. */
export function groupWeight(sum: number): number {
  return Math.min(Math.abs(sum) / GROUP_SCALE, 1);
}

/** "×2.4" odds multiplier text (two significant figures below 10). */
export function oddsText(logOdds: number): string {
  const x = Math.exp(logOdds);
  return `×${x >= 10 ? x.toFixed(0) : x.toFixed(x < 1 ? 2 : 1)}`;
}

/** Below this |log-odds| a group counts as "little effect" (grey in the strip and the 3D streams). */
export const MIN_EFFECT = 0.05;

/** Colours shared by the panel strip and the 3D streams (Tailwind red-500 / sky-500, as the bars). */
export const UP_COLOR = "#ef4444";
export const DOWN_COLOR = "#0ea5e9";
export const NEUTRAL_COLOR = "#64748b";

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
