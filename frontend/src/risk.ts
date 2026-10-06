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
