import { useEffect, type CSSProperties } from "react";
import type { ExplainResponse, MetricsResponse } from "../api/client";
import { FEATURE_LABELS, formatFeatureValue } from "../featureLabels";
import { BAND_COLOR, BAND_STYLE, formatPct, reliability, vesselBand } from "../risk";
import type { AtlasPick } from "./AtlasHeart";
import { VESSEL, type Vessel } from "../anatomy/registry";


interface Props {
  vessel: Vessel;
  result: ExplainResponse | null;
  metrics: MetricsResponse | null;
  pick: AtlasPick | null;
  onClose: () => void;
  /** Placement from the 3D view (below its toolbar and notices, above its bottom controls). */
  style?: CSSProperties;
}

/** Esc closes the card or chip. */
function useEscape(onClose: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
}

/** Everything about one artery's estimate in one card: what it says, how sure, why, and what it can't say. */
export default function VesselInspector({ vessel, result, metrics, pick, onClose, style }: Props) {
  useEscape(onClose);

  const info = VESSEL[vessel];
  const r = result?.results[vessel];
  const m = metrics?.metrics[vessel];
  const band = r ? vesselBand(r.probability) : null;
  const rel = m ? reliability(m.cv.roc_auc.mean) : null;
  const exp = result?.explanations[vessel];
  const drivers = exp
    ? exp.feature_names
      .map((name, i) => ({ name, shap: exp.shap_values[i], value: exp.feature_values[i] }))
      .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
      .slice(0, 3)
    : [];

  return (
    <div
      role="dialog"
      aria-label={`${vessel.toUpperCase()} details`}
      className="absolute right-3 top-12 z-20 max-h-[calc(100%-7rem)] w-[270px] overflow-auto rounded bg-slate-900/95 p-3 text-xs shadow-lg"
      style={style}
      onPointerDown={(e) => e.stopPropagation()}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="font-semibold text-white">
          {info.label} · {info.name}
        </div>
        <button onClick={onClose} aria-label="Close" title="Close (Esc)" className="-mt-1 px-1 text-base leading-none text-gray-400 hover:text-white">
          ×
        </button>
      </div>

      {pick && pick.vessel === vessel && (
        <div className="mt-1 text-[11px] text-gray-400">
          Clicked: <span className="text-gray-200 first-letter:uppercase">{pick.branch.name}</span> ({pick.branch.fma},
          BodyParts3D {pick.branch.fj}), part of {pick.part} ({pick.partFma}). Branches share their artery's single
          estimate.
        </div>
      )}

      {r && band && (
        <div className="mt-2">
          <div className="text-gray-300">Chance of ≥50% narrowing</div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold" style={{ color: BAND_COLOR[band] }}>{formatPct(r.probability)}</span>
            <span className={`rounded px-1.5 py-0.5 text-[10px] ${BAND_STYLE[band].text} ${band === "Intermediate" ? "border border-dashed border-amber-400" : "bg-slate-800"}`}>
              {band === "Intermediate" ? "undecided" : BAND_STYLE[band].label.toLowerCase()}
            </span>
          </div>
          {r.band_observed_rate != null && (
            <div className="mt-0.5 text-gray-400">
              In validation, {Math.round(r.band_observed_rate * 100)}% of patients in this band had ≥50% narrowing of
              the {vessel.toUpperCase()}.
            </div>
          )}
          <div className="mt-1 text-gray-400">
            {r.probability >= r.threshold ? "Above" : "Below"} this model's decision threshold ({formatPct(r.threshold)}):{" "}
            {r.probability >= r.threshold ? "flagged" : "not flagged"}.
            {m && (
              <>
                {" "}At this threshold the model flagged {formatPct(m.operating_points.f1_threshold.recall)} of patients
                with this narrowing in validation (specificity {formatPct(m.operating_points.f1_threshold.specificity)}).
              </>
            )}
          </div>
        </div>
      )}

      {m && rel && (
        <div className="mt-2 text-gray-400">
          <span className="text-gray-300">How reliable:</span> detects narrowing in this artery with cross-validated AUC{" "}
          {m.cv.roc_auc.mean.toFixed(2)} ({rel.label}). The vessel models only partly tell arteries apart, so read
          the three estimates independently, not as a ranking.
        </div>
      )}

      {drivers.length > 0 && (
        <div className="mt-2">
          <div className="text-gray-300">Main drivers</div>
          <ul className="mt-0.5 space-y-0.5">
            {drivers.map((d) => (
              <li key={d.name} className="flex justify-between gap-2">
                <span className="truncate text-gray-200">
                  {FEATURE_LABELS[d.name]?.label ?? d.name}{" "}
                  <span className="text-gray-500">{formatFeatureValue(d.name, d.value)}</span>
                </span>
                <span className={`shrink-0 ${d.shap >= 0 ? "text-red-400" : "text-sky-400"}`}>
                  {d.shap >= 0 ? "↑" : "↓"} ×{Math.exp(d.shap).toFixed(1)} odds
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-2 text-gray-400">
        <span className="text-gray-300">Usually supplies:</span> {info.territory}; AHA segments {info.aha.join(", ")}
        {vessel === "rca" ? " (right dominance assumed)" : ""}.
      </div>
      <div className="mt-2 border-t border-slate-700 pt-1.5 text-[10px] text-gray-500">
        Not shown: where along the artery a narrowing is, how severe it is, or the left main (not labelled in this
        dataset).
      </div>
    </div>
  );
}

/**
 * One line for the selected artery when the full card would cover the view (narrow 3D views, the Reasoning
 * view): its estimate, band and validation AUC, a way to the full card and a way back to the overall model.
 */
export function VesselChip({ vessel, result, metrics, onClose, onDetails, detailsTitle }: {
  vessel: Vessel;
  result: ExplainResponse | null;
  metrics: MetricsResponse | null;
  onClose: () => void;
  onDetails?: () => void;
  detailsTitle?: string;
}) {
  useEscape(onClose);
  const info = VESSEL[vessel];
  const r = result?.results[vessel];
  const band = r ? vesselBand(r.probability) : null;
  const auc = metrics?.metrics[vessel]?.cv.roc_auc.mean;
  return (
    <div
      role="group"
      aria-label={`${info.label} estimate`}
      className="pointer-events-auto flex max-w-full items-center gap-1.5 whitespace-nowrap rounded bg-slate-900/95 px-2 py-0.5 text-[11px] text-gray-300 shadow"
      onPointerDown={(e) => e.stopPropagation()}
    >
      <b className="text-white">{info.label}</b>
      {r && band && (
        <>
          <span className="font-semibold" style={{ color: BAND_COLOR[band] }}>{formatPct(r.probability)}</span>
          <span>{band === "Intermediate" ? "undecided" : band.toLowerCase()}</span>
        </>
      )}
      {auc !== undefined && <span className="text-gray-500">AUC {auc.toFixed(2)}</span>}
      {onDetails && (
        <button onClick={onDetails} title={detailsTitle} className="rounded px-1 text-sky-300 hover:bg-slate-700 hover:text-sky-200">
          details
        </button>
      )}
      <button onClick={onClose} aria-label="Deselect" title="Back to the overall model (Esc)" className="px-0.5 text-sm leading-none text-gray-400 hover:text-white">
        ×
      </button>
    </div>
  );
}
