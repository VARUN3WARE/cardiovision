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
  /**
   * Why what is shown may not match the form: the inputs changed since this estimate ("previous estimate"), or
   * the streams trail newer probabilities during a what-if drag ("updating").
   */
  stale: "previous estimate" | "updating" | null;
  top: Vessel | null;
  lab: LabState | null;
  /** Seconds into the step (the lab step's caption follows its demonstration). */
  t: number;
}

const list = (xs: string[]) => (xs.length > 1 ? `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}` : xs[0] ?? "");

/** The legend's names for the bands (the middle one is the "undecided" band). */
const BAND_WORD: Record<Band, string> = { Lower: "lower", Intermediate: "undecided", Higher: "higher" };
const COUNT = ["none", "one", "two", "three", "four", "five", "six"];

/**
 * Why the story shows `top`: alone in the highest band, or the highest chance among arteries that share its band
 * (or, when they also read the same, just that).
 */
function whyShown(r: ExplainResponse, top: Vessel): string {
  const band = (v: Vessel) => vesselBand(r.results[v].probability);
  const pct = (v: Vessel) => formatPct(r.results[v].probability);
  const group = [top, ...VESSEL_IDS.filter((v) => v !== top && band(v) === band(top))];
  if (group.length === 1) return "highest band";
  const who = group.length === VESSEL_IDS.length ? `all ${COUNT[group.length] ?? group.length}` : `the ${list(group.map((v) => VESSEL[v].label))}`;
  const where = `in the ${BAND_WORD[band(top)]} band`;
  return group.every((v) => pct(v) === pct(top)) ? `${who} read ${pct(top)} ${where}` : `highest chance; ${who} are ${where}`;
}

/**
 * The caption under the story bar for a step (the closing step has its card instead: empty). Each starts with its
 * honesty qualifier, so it stays visible when a narrow view shows only the first lines.
 */
export function storyCaption(id: StoryStepId, c: StoryContext): string {
  const scenario = c.whatIf ? " (what-if scenario)" : "";
  const previous = c.stale === "previous estimate" ? " (previous estimate: inputs changed)" : "";
  switch (id) {
    case "chest":
      return "A reference heart, not this patient's imaging: the BodyParts3D anatomical atlas in the chest, for orientation.";
    case "twin": {
      if (!c.result) {
        return "Illustrative heartbeat and ECG. No estimate yet: once one is run, they follow the patient's recorded values; until then, typical ones.";
      }
      const tw = c.twin;
      const median = (k: "hr" | "ef" | "bp") => (tw.notRecorded.has(k) ? ", not entered: cohort median" : "");
      const findings = ecgFindings(tw).map((f) => f.text);
      return (
        `Illustrative ECG, not the patient's tracing${previous}. The heartbeat and trace follow this patient's recorded pulse ` +
        `(${Math.round(tw.hr)} bpm${median("hr")}), EF (${Math.round(tw.ef)}%, ${efCategory(tw.ef)}${median("ef")}), blood ` +
        `pressure (${Math.round(tw.bp)} mmHg${median("bp")}) and ECG findings (${findings.length ? list(findings) : "none recorded"}).`
      );
    }
    case "reasoning": {
      if (!c.result) return "Associations, not causes. No estimate yet: once one is run, the streams show what moved it.";
      const updating = c.stale === "updating" ? " (updating)" : "";
      return (
        `Associations in a referral cohort, not causes. Overall CAD estimate ${formatPct(c.result.results.cad.probability)}` +
        `${scenario}${previous}${updating}: each stream is one data type's evidence flowing into it (data, not anatomy); ` +
        "width = how much it moved the estimate, red up, blue down."
      );
    }
    case "arteries": {
      if (!c.result || !c.top) {
        return "Independent estimates, one model per artery, not a culprit ranking. No estimate yet: once one is run, each artery gets its colour.";
      }
      const r = c.result;
      const each = VESSEL_IDS.map((v) => `${VESSEL[v].label} ${formatPct(r.results[v].probability)}`).join(", ");
      return (
        `Independent estimates, one per artery, not a culprit ranking${scenario}${previous}: chance of ≥50% narrowing ` +
        `${each}. Shown: the ${VESSEL[c.top].label} (${whyShown(r, c.top)}).`
      );
    }
    case "lab": {
      const l = c.lab;
      const name = VESSEL[l?.vessel ?? c.top ?? VESSEL_IDS[0]].label;
      const sandbox = `Teaching sandbox, not this patient's artery: textbook physiology on a generic ${name} narrowing.`;
      if (!l) return sandbox;
      // What the lab panel beside it shows, with its rounding (whole percent).
      const pc = (x: number) => `${Math.round(x * 100)}%`;
      const at = `At ${pc(l.narrowing)} narrowing`;
      const unmet = Math.round(l.ischaemia * 100);
      if (l.exercise < 0.02) {
        if (unmet > 0) {
          return `${sandbox} ${at}, even resting flow falls short at this heart's resting pulse × pressure: ${unmet}% of the territory's need is unmet (blue).`;
        }
        const most = l.qmaxFrac < 0.95 ? `is down to ${pc(l.qmaxFrac)} of normal` : "is still about normal";
        return `${sandbox} ${at}, resting flow still meets the need; the most it can carry ${most} (it falls once the narrowing passes about 50%).`;
      }
      if (unmet === 0) return `${sandbox} With exercise, demand rises; at ${pc(l.narrowing)} narrowing the artery still carries enough.`;
      return (
        `${sandbox} With exercise, demand outgrows what the narrowed artery can carry: ${unmet}% of its territory's need is ` +
        "unmet (blue) and the cascade lights up in order."
      );
    }
    case "honesty":
      return "";
  }
}

/** The closing card's content in one sentence per item, for screen readers (the card itself is not a live region). */
export function honestySpoken(metrics: MetricsResponse | null): string {
  const aucs = honestyAucs(metrics)
    .map((r) => `${r.label}${r.auc === null ? "" : ` ${r.auc.toFixed(2)}`}, 95% CI ${r.ci[0].toFixed(2)} to ${r.ci[1].toFixed(2)}`)
    .join("; ");
  return `What this is, and is not. ${HONESTY.use} ${HONESTY.anatomy} ${HONESTY.cohort} Cross-validated ROC AUC: ${aucs}. ${HONESTY.caveats}`;
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

/** The closing card's statements (the use line leads; the AUC table sits between the cohort and the caveats). */
export const HONESTY = {
  anatomy: "Reference anatomy (BodyParts3D, © DBCLS, CC BY 4.0), not this patient's imaging: findings are model inputs, never places on the heart.",
  cohort: `Trained and validated on a single-centre referral cohort: ${VERIFICATION.rows} patients referred for angiography (${Math.round(VERIFICATION.prevalenceCad * 100)}% had CAD).`,
  caveats: "Attributions are associations, not causes; the lab is textbook physiology, not this patient's arteries.",
  use: "Decision support and education only: not a diagnosis, and no substitute for angiography or clinical judgement.",
};
