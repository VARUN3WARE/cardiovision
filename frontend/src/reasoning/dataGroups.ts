/**
 * Splits one model's exact additive attributions (log-odds) into data-type groups, so "how much did the
 * ECG move this estimate?" has an exact answer: base + sum of group sums = base + sum of attributions =
 * the model's log-odds.
 *
 * Raw columns follow FEATURE_GROUPS (featureLabels.ts); demographics join history. Derived composites are
 * assigned to one group by a stated convention (DERIVED_GROUP), never split. These are associations in a
 * referral cohort, not causes.
 */
import type { ShapExplanation } from "../api/client";
import { FEATURE_GROUPS, FEATURE_LABELS } from "../featureLabels";

export type DataGroupId = "history" | "symptoms" | "ecg" | "echo" | "labs" | "pretest" | "other";

export interface DataGroupDef {
  id: DataGroupId;
  label: string;
  /** Short name for the 3D towers. */
  short: string;
}

/** Display order (also the tower order in 3D). "other" only appears if an unknown column turns up. */
export const DATA_GROUPS: readonly DataGroupDef[] = [
  { id: "history", label: "History & demographics", short: "History" },
  { id: "symptoms", label: "Symptoms & exam", short: "Symptoms & exam" },
  { id: "ecg", label: "ECG", short: "ECG" },
  { id: "echo", label: "Echo", short: "Echo" },
  { id: "labs", label: "Labs", short: "Labs" },
  { id: "pretest", label: "Pre-test score", short: "Pre-test score" },
];
export const OTHER_GROUP: DataGroupDef = { id: "other", label: "Other inputs", short: "Other" };

const BY_TITLE: Record<string, DataGroupId> = {
  Demographics: "history",
  "History & risk factors": "history",
  "Symptoms & examination": "symptoms",
  ECG: "ecg",
  Echocardiography: "echo",
  Laboratory: "labs",
};

/**
 * Convention for derived composites: each goes to the data type it is computed from; scores that mix
 * symptoms with demographics (Diamond-Forrester, chest pain class, age x male) form the pre-test group.
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
}

export interface GroupContribution extends DataGroupDef {
  /** Exact sum of the group's attributions, in log-odds. */
  sum: number;
  /** Members, largest |attribution| first. */
  findings: GroupFinding[];
}

export interface GroupedAttribution {
  base: number;
  /** base + sum of all attributions (= base + sum of group sums): the model's log-odds. */
  logOdds: number;
  /** Every group in display order ("other" only when non-empty). */
  groups: GroupContribution[];
}

export function groupAttributions(exp: ShapExplanation): GroupedAttribution {
  const defs = [...DATA_GROUPS, OTHER_GROUP];
  const byId = new Map<DataGroupId, GroupContribution>(defs.map((d) => [d.id, { ...d, sum: 0, findings: [] }]));
  let total = 0;
  exp.feature_names.forEach((name, i) => {
    const shap = exp.shap_values[i];
    const g = byId.get(dataGroupOf(name))!;
    g.sum += shap;
    total += shap;
    g.findings.push({
      name,
      label: FEATURE_LABELS[name]?.label ?? name,
      value: exp.feature_values[i],
      shap,
      derived: exp.derived?.[i] ?? false,
    });
  });
  for (const g of byId.values()) g.findings.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  const groups = defs.map((d) => byId.get(d.id)!).filter((g) => g.id !== "other" || g.findings.length > 0);
  return { base: exp.base_value, logOdds: exp.base_value + total, groups };
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

/** Colours shared by the panel strip and the 3D streams (Tailwind red-500 / sky-500, as the bars). */
export const UP_COLOR = "#ef4444";
export const DOWN_COLOR = "#0ea5e9";
