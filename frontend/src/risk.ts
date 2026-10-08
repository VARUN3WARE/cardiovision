// Shared formatting for probabilities, risk bands and validation reliability.

/** Whole-percent display, clamped: n=303 does not support decimals or certainty claims. */
export function formatPct(p: number): string {
  if (p < 0.05) return "<5%";
  if (p > 0.95) return ">95%";
  return `${Math.round(p * 100)}%`;
}

export const BAND_STYLE: Record<string, { text: string; label: string }> = {
  Lower: { text: "text-green-400", label: "Lower likelihood" },
  Intermediate: { text: "text-amber-300", label: "Intermediate likelihood" },
  Higher: { text: "text-red-400", label: "Higher likelihood" },
};

export type Band = "Lower" | "Intermediate" | "Higher";

/**
 * One colour per likelihood band, shared by the 3D arteries, territories, labels, legend and the
 * vessel cards, so a colour always means the same band. The middle band is drawn striped in 3D
 * ("undecided": its range matches the models' 90% conformal undecided zone).
 */
export const BAND_COLOR: Record<Band, string> = {
  Lower: "#22c55e",
  Intermediate: "#f59e0b",
  Higher: "#ef4444",
};

// Vessel band cut-offs; replaced by the trained models' own cut-offs from /metrics when they load.
let vesselCuts: [number, number] = [1 / 3, 2 / 3];

export function configureBands(metrics: { metrics: Record<string, { bands: { hi: number }[] }> }) {
  const b = metrics.metrics.lad?.bands;
  if (b?.length === 3) vesselCuts = [b[0].hi, b[1].hi];
}

export function getVesselCuts(): [number, number] {
  return vesselCuts;
}

export function vesselBand(p: number): Band {
  return p < vesselCuts[0] ? "Lower" : p < vesselCuts[1] ? "Intermediate" : "Higher";
}

export function bandStyle(band: string) {
  return BAND_STYLE[band] ?? BAND_STYLE.Intermediate;
}

/** Plain-language reliability tier from cross-validated ROC-AUC. */
export function reliability(auc: number): { label: string; weak: boolean } {
  if (auc >= 0.85) return { label: "good discrimination", weak: false };
  if (auc >= 0.75) return { label: "moderate discrimination", weak: false };
  return { label: "limited discrimination", weak: true };
}

export const VESSEL_NAME: Record<string, string> = {
  cad: "Overall CAD",
  lad: "LAD",
  lcx: "LCX",
  rca: "RCA",
};
