/**
 * The single table that ties model outputs to anatomy. Every view (3D arteries and territories on all
 * three heart models, labels, cards, bullseye, inspector, explanation tabs) iterates over this list, so
 * adding an artery or a model output is one entry here plus its geometry.
 *
 * Sources: AHA 17-segment assignment, Cerqueira et al., Circulation 2002;105:539-42 (right dominance);
 * FMA ids as used by BodyParts3D (scripts/bp3d/parts.json).
 */

/** Model outputs shown on an artery. */
export type Vessel = "lad" | "lcx" | "rca";

export interface VesselEntry {
  id: Vessel; // model output key in the API (results, explanations, vessel_probabilities)
  label: string; // short label on the heart and cards
  name: string;
  territory: string; // muscle the artery usually supplies
  aha: number[]; // AHA 17-segment model segments it usually supplies
  fma: string; // FMA concept of the artery tree
  atlasPart: string; // node key of its tree in bp3d_heart.glb
  /** Where its label sits on the atlas heart: a fraction along its trunk (FMA id; null = branch nearest the left main). */
  atlasLabel: { trunkFma: string | null; along: number };
}

export const VESSELS: readonly VesselEntry[] = [
  {
    id: "lad", label: "LAD", name: "Left Anterior Descending",
    territory: "Anterior LV wall, anterior two-thirds of the septum, apex",
    aha: [1, 2, 7, 8, 13, 14, 17], fma: "FMA3862", atlasPart: "lad",
    atlasLabel: { trunkFma: "FMA74912", along: 0.45 },
  },
  {
    id: "lcx", label: "LCX", name: "Left Circumflex",
    territory: "Lateral and posterolateral LV wall",
    aha: [5, 6, 11, 12, 16], fma: "FMA3895", atlasPart: "lcx",
    atlasLabel: { trunkFma: null, along: 0.5 },
  },
  {
    id: "rca", label: "RCA", name: "Right Coronary Artery",
    territory: "Right ventricle, inferior LV wall, posterior septum (via PDA)",
    aha: [3, 4, 9, 10, 15], fma: "FMA50039", atlasPart: "rca",
    atlasLabel: { trunkFma: "FMA3802", along: 0.35 },
  },
];

/** Arteries drawn for anatomy but without a model output (shown neutral, not selectable). */
export const UNMODELLED = [
  { id: "lm", label: "Left main", fma: "FMA3855", atlasPart: "lm", reason: "not labelled in this dataset" },
] as const;

export const VESSEL_IDS: readonly Vessel[] = VESSELS.map((v) => v.id);
export const VESSEL: Record<Vessel, VesselEntry> = Object.fromEntries(VESSELS.map((v) => [v.id, v])) as Record<Vessel, VesselEntry>;

/** Model tabs: the overall CAD model, then one per artery. */
export const MODEL_TABS = ["cad", ...VESSEL_IDS] as const;

/** AHA segment -> the artery that usually supplies it. */
export const SEGMENT_VESSEL: Record<number, Vessel> = Object.fromEntries(
  VESSELS.flatMap((v) => v.aha.map((s) => [s, v.id])),
) as Record<number, Vessel>;

/** Per-artery probabilities (the API's vessel_probabilities). */
export type ArteryProbs = Record<Vessel, number>;
