/**
 * Story mode: a captioned ~90 s tour of the 3D view on the current patient (the plan's T6). Six steps, each a
 * view state and a caption; this module holds the script and the words (pure, unit-tested), HeartScene sets the
 * view. Every number in a caption comes from the patient's result, the twin's recorded values, the lab's
 * textbook model, /metrics or the verification copy, and every caption keeps the 3D view's honesty rules.
 * Deep links: ?story=1 plays from the start; ?story=1&step=N (1..6) shows step N as it ends, paused (add
 * &play=1 to play on from there).
 */
import { VESSEL, VESSEL_IDS, type Vessel } from "../anatomy/registry";
import type { ExplainResponse, MetricsResponse, Target } from "../api/client";
import { formatPct, vesselBand, type Band } from "../risk";
import type { LabState } from "../twin/coronaryPhysiology";
import { ecgFindings, efCategory, type TwinParams } from "../twin/twin";
import { VERIFICATION } from "./verification";

export type StoryStepId = "chest" | "twin" | "reasoning" | "arteries" | "lab" | "honesty";

export interface StoryStep {
  id: StoryStepId;
  title: string;
  /** Seconds the step stays on screen while playing. */
  seconds: number;
}

export const STORY_STEPS: readonly StoryStep[] = [
  { id: "chest", title: "A heart in the chest", seconds: 12 },
  { id: "twin", title: "The function twin", seconds: 15 },
  { id: "reasoning", title: "What moved the estimate", seconds: 20 },
  { id: "arteries", title: "One estimate per artery", seconds: 15 },
  { id: "lab", title: "Ischaemia lab (sandbox)", seconds: 18 },
  { id: "honesty", title: "What this is, and is not", seconds: 12 },
];

/** The lab step's demonstration: the narrowing grows, then the heart exercises (seconds into the step). */
export const LAB_DEMO = {
  narrowing: { from: 1, to: 8, max: 85 },
  exercise: { from: 9.5, to: 15 },
};

/** Narrowing (%) and exercise level (0..1) of the lab demonstration `t` seconds into the step. */
export function labDemo(t: number): { narrowing: number; exercise: number } {
  const ramp = (a: number, b: number) => {
    const k = Math.min(1, Math.max(0, (t - a) / (b - a)));
    return k * k * (3 - 2 * k);
  };
  const n = LAB_DEMO.narrowing, e = LAB_DEMO.exercise;
  return {
    narrowing: Math.round(n.max * ramp(n.from, n.to)),
    exercise: Math.round(100 * ramp(e.from, e.to)) / 100,
  };
}

/** Story deep link: the step to open (0-based) and whether to play, or null without ?story=1. */
export function storyFromQuery(search: string): { step: number; playing: boolean; atEnd: boolean } | null {
  const q = new URLSearchParams(search);
  if (q.get("story") !== "1") return null;
  const raw = q.get("step");
  const n = raw === null || raw.trim() === "" ? NaN : Math.round(Number(raw));
  if (!Number.isFinite(n)) return { step: 0, playing: true, atEnd: false };
  const step = Math.min(STORY_STEPS.length, Math.max(1, n)) - 1;
  const playing = q.get("play") === "1";
  return { step, playing, atEnd: !playing };
}

const BAND_RANK: Record<Band, number> = { Lower: 0, Intermediate: 1, Higher: 2 };

/**
 * The artery the story inspects: the one whose estimate is in the highest band; among equals, the highest
 * probability (ties beyond that: registry order). Null without a result.
 */
export function topArtery(result: ExplainResponse | null): Vessel | null {
  if (!result) return null;
  let best: Vessel | null = null;
  for (const v of VESSEL_IDS) {
    if (!best) {
      best = v;
      continue;
    }
    const p = result.results[v].probability, q = result.results[best].probability;
    const rank = BAND_RANK[vesselBand(p)] - BAND_RANK[vesselBand(q)];
    if (rank > 0 || (rank === 0 && p > q)) best = v;
  }
  return best;
}

export interface StoryContext {
  /** The result the colours and streams show (a what-if scenario's while one is on). */
  result: ExplainResponse | null;
  /** The twin from the patient's recorded values. */
  twin: TwinParams;
  /** A what-if scenario colours the arteries. */
  whatIf: boolean;
  /** The estimates on screen are newer than the streams, or the inputs changed since. */
  stale: boolean;
  top: Vessel | null;
  lab: LabState | null;
  /** Seconds into the step (the lab step's caption follows its demonstration). */
  t: number;
}

const list = (xs: string[]) => (xs.length > 1 ? `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}` : xs[0] ?? "");

/** The caption under the story bar for a step (the closing step has its card instead: empty). */
export function storyCaption(id: StoryStepId, c: StoryContext): string {
  const scenario = c.whatIf ? " (what-if scenario)" : "";
  switch (id) {
    case "chest":
      return "A reference heart (BodyParts3D anatomical atlas) in the chest, for orientation: not this patient's imaging.";
    case "twin": {
      if (!c.result) return "Once an estimate is run, it beats like the patient's heart; until then it shows typical values.";
      const tw = c.twin;
      const median = (k: "hr" | "ef" | "bp") => (tw.notRecorded.has(k) ? ", not entered: cohort median" : "");
      const findings = ecgFindings(tw).map((f) => f.text);
      return (
        `Driven by this patient's recorded pulse (${Math.round(tw.hr)} bpm${median("hr")}), EF (${Math.round(tw.ef)}%, ` +
        `${efCategory(tw.ef)}${median("ef")}), blood pressure (${Math.round(tw.bp)} mmHg${median("bp")}) and ECG findings ` +
        `(${findings.length ? list(findings) : "none recorded"}). The ECG trace is illustrative, not the patient's tracing.`
      );
    }
    case "reasoning": {
      if (!c.result) return "Run an estimate to see what moved it.";
      return (
        `Overall CAD estimate ${formatPct(c.result.results.cad.probability)}${scenario}: each stream is one data type's evidence ` +
        "flowing into it (data, not anatomy); width = how much it moved the estimate, red up, blue down. Associations in a " +
        `referral cohort, not causes.${c.stale ? " Updating." : ""}`
      );
    }
    case "arteries": {
      if (!c.result || !c.top) return "Run an estimate to colour the arteries: one model per artery.";
      const r = c.result;
      const each = VESSEL_IDS.map((v) => `${VESSEL[v].label} ${formatPct(r.results[v].probability)}`).join(", ");
      return (
        `Chance of ≥50% narrowing, one model per artery${scenario}: ${each}. Shown: the ${VESSEL[c.top].label} (highest band). ` +
        "The three estimates are independent, not a culprit ranking."
      );
    }
    case "lab": {
      const name = VESSEL[c.lab?.vessel ?? c.top ?? VESSEL_IDS[0]].label;
      const sandbox = `Teaching sandbox: textbook physiology on a generic ${name} narrowing, not this patient's artery.`;
      if (!c.lab || c.lab.exercise < 0.02) {
        return `${sandbox} Up to 85% narrowing, resting flow holds; the most it can carry falls once it passes about 50%.`;
      }
      return (
        `${sandbox} With exercise, demand outgrows what the narrowed artery can carry: ${formatPct(c.lab.ischaemia)} of its ` +
        "territory's need is unmet (blue) and the cascade lights up in order."
      );
    }
    case "honesty":
      return "";
  }
}

export interface HonestyAuc {
  target: Target;
  label: string;
  /** Cross-validated ROC AUC from /metrics (null before it loads). */
  auc: number | null;
  ci: readonly [number, number];
}

/** The closing card's AUC rows: the served models' cross-validated AUC (/metrics) with the bootstrap 95% CI. */
export function honestyAucs(metrics: MetricsResponse | null): HonestyAuc[] {
  return (["cad", ...VESSEL_IDS] as Target[]).map((t) => ({
    target: t,
    label: t === "cad" ? "CAD" : VESSEL[t].label,
    auc: metrics?.metrics[t]?.cv.roc_auc.mean ?? null,
    ci: VERIFICATION.aucCi[t],
  }));
}

/** The closing card's statements (the AUC table sits between the cohort and the caveats). */
export const HONESTY = {
  anatomy: "Reference anatomy (BodyParts3D, © DBCLS, CC BY 4.0), not this patient's imaging: findings are model inputs, never places on the heart.",
  cohort: `Trained and validated on a single-centre referral cohort: ${VERIFICATION.rows} patients referred for angiography (${Math.round(VERIFICATION.prevalenceCad * 100)}% had CAD).`,
  caveats: "Attributions are associations, not causes; the lab is textbook physiology, not this patient's arteries.",
  use: "Decision support and education only: not a diagnosis, and no substitute for angiography or clinical judgement.",
};
