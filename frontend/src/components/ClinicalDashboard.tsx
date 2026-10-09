import type { ExplainResponse, MetricsResponse, Target, TargetMetrics } from "../api/client";
import { FIELD_LABEL } from "../fieldSpec";
import { BAND_COLOR, bandStyle, formatPct, reliability, type Band } from "../risk";
import Bullseye from "./Bullseye";
import { VESSELS } from "../anatomy/registry";
import type { Preset } from "./FeatureInput";
import type { Vessel } from "./HeartScene";

interface Props {
  result: ExplainResponse | null;
  metrics: MetricsResponse | null;
  selected: Vessel | null;
  onSelect: (v: Vessel | null) => void;
  /** Hovering the bullseye highlights that artery in the 3D view. */
  onHover?: (v: Vessel | null) => void;
  preset: Preset | null;
  /** The result is a what-if scenario, not the patient's recorded values. */
  scenario?: boolean;
}

export default function ClinicalDashboard({ result, metrics, selected, onSelect, onHover = () => {}, preset, scenario = false }: Props) {
  if (!result) {
    return (
      <div className="text-sm text-gray-400">
        Choose a patient or enter clinical findings, then select <b>Estimate risk</b>.
      </div>
    );
  }
  const cad = result.results.cad;
  const m = metrics?.metrics;
  const cadStyle = bandStyle(cad.band);
  const maxVessel = Math.max(result.results.lad.probability, result.results.lcx.probability, result.results.rca.probability);
  const discordant = (cad.probability < cad.threshold && maxVessel >= 0.5) || (cad.probability > 0.9 && maxVessel < 0.2);

  return (
    <div className="space-y-3" aria-live="polite">
      {scenario && (
        <Notice tone="sky">
          <b>What-if scenario.</b> Estimates for changed model inputs, not this patient's recorded values. Reset in the
          what-if card below.
        </Notice>
      )}
      <div>
        <div className="text-sm text-gray-300">Probability of coronary artery disease</div>
        <div className="flex items-baseline gap-3">
          <div className="text-4xl font-bold" style={{ color: BAND_COLOR[cad.band as Band] }}>{formatPct(cad.probability)}</div>
          <div className={`text-sm font-semibold ${cadStyle.text}`}>{cadStyle.label}</div>
        </div>
        {cad.band_observed_rate != null && (
          <div className="text-xs text-gray-400">
            In validation, {Math.round(cad.band_observed_rate * 100)}% of patients in this band had CAD on angiography.
          </div>
        )}
        {m && <ValidationLine m={m.cad} />}
        <div className="mt-1 text-[11px] text-gray-500">
          Model estimate, not a diagnosis. Confirm with angiography and clinical assessment.
        </div>
        {preset && <GroundTruth target="cad" preset={preset} threshold={cad.threshold} />}
      </div>

      <div>
        <div className="mb-1 text-xs text-gray-400">Probability of ≥50% stenosis per vessel</div>
        <div className="grid grid-cols-3 gap-2">
          {VESSELS.map(({ id: v, label }) => {
            const r = result.results[v];
            const style = bandStyle(r.band);
            const rel = m ? reliability(m[v].cv.roc_auc.mean) : null;
            return (
              <button
                key={v}
                onClick={() => onSelect(selected === v ? null : v)}
                aria-pressed={selected === v}
                title={m ? `Cross-validated AUC ${m[v].cv.roc_auc.mean.toFixed(2)}; decision threshold ${formatPct(r.threshold)}` : undefined}
                className={`rounded border p-2 text-center transition ${selected === v ? "border-white bg-slate-700" : "bg-slate-900 hover:bg-slate-700"} ${
                  rel?.weak ? "border-dashed border-slate-500" : "border-transparent"}`}
              >
                <div className="text-xs uppercase text-gray-300">{label}</div>
                {/* Same band colour as this artery in the 3D view. */}
                <div className="text-xl font-bold" style={{ color: BAND_COLOR[r.band as Band] }}>{formatPct(r.probability)}</div>
                <div className={`text-[10px] ${style.text}`}>{r.band}</div>
                {rel && <div className="mt-0.5 text-[10px] text-gray-500">AUC {m![v].cv.roc_auc.mean.toFixed(2)}</div>}
                {preset && <GroundTruth target={v} preset={preset} threshold={r.threshold} compact />}
              </button>
            );
          })}
        </div>
        {m && (
          <div className="mt-1 text-[11px] text-gray-500">
            Dashed cards: vessel models with limited discrimination (cross-validated AUC &lt; 0.75); treat as indicative.
          </div>
        )}
        <div className="mt-3">
          <Bullseye probs={result.vessel_probabilities} selected={selected} onHover={onHover} onSelect={onSelect} />
        </div>
      </div>

      {discordant && (
        <Notice tone="sky">
          The overall CAD estimate and the per-vessel estimates come from separate models and disagree for this patient,
          so treat both with extra caution.
        </Notice>
      )}

      {result.warnings.length > 0 && (
        <Notice tone="amber">
          {result.warnings.map((w) => (
            <div key={w.field}>
              {FIELD_LABEL[w.field] ?? w.field} {w.value} is outside the training range ({w.train_min}–{w.train_max}); interpret with caution.
            </div>
          ))}
        </Notice>
      )}

      {result.imputed.length > 0 && (
        <details className="rounded border border-slate-700 bg-slate-900/60 px-2 py-1.5 text-[11px] text-gray-400">
          <summary className="cursor-pointer">
            {result.imputed.length} value{result.imputed.length === 1 ? "" : "s"} not entered, filled with typical cohort values
          </summary>
          <div className="mt-1">{result.imputed.map((k) => FIELD_LABEL[k] ?? k).join(", ")}</div>
        </details>
      )}

      {preset && (
        <div className="rounded border border-slate-700 bg-slate-900/60 p-2 text-[11px] text-gray-400">
          <b className="text-gray-200">Dataset patient #{preset.patient_index}</b>. Angiography:{" "}
          <b className="text-gray-200">{preset.description}</b>. The live model was trained on this patient, so its score
          is optimistic; the <i>held-out</i> estimates come from models that never saw this patient.
        </div>
      )}
    </div>
  );
}

function ValidationLine({ m }: { m: TargetMetrics }) {
  return (
    <div className="text-[11px] text-gray-500">
      Validation: AUC {m.cv.roc_auc.mean.toFixed(2)} ({reliability(m.cv.roc_auc.mean).label}), {m.validation}, n={m.n}
    </div>
  );
}

function GroundTruth({ target, preset, threshold, compact }: {
  target: Target; preset: Preset; threshold: number; compact?: boolean;
}) {
  const truth = preset.ground_truth[target];
  const heldOut = preset.oof_probabilities[target];
  const call = heldOut >= threshold; // the model's own learned decision threshold
  const badge = call === truth
    ? { text: "match", cls: "text-green-400" }
    : call ? { text: "false alarm", cls: "text-amber-400" } : { text: "missed", cls: "text-red-400" };
  // Labels in this dataset mean >=50% diameter narrowing; "no" does not mean a plaque-free artery.
  const truthText = target === "cad" ? (truth ? "CAD (≥50% narrowing)" : "no ≥50% narrowing") : truth ? "≥50% narrowing" : "no ≥50% narrowing";
  if (compact) {
    return (
      <div className="mt-1 border-t border-slate-700 pt-1 text-[10px] leading-tight text-gray-400">
        <div>Angio: {truth ? "≥50%" : "<50%"} narrowing</div>
        <div>Held-out {formatPct(heldOut)} <span className={badge.cls}>· {badge.text}</span></div>
      </div>
    );
  }
  return (
    <div className="mt-1 text-[11px] text-gray-400">
      Angiography: <b className="text-gray-200">{truthText}</b> · held-out estimate {formatPct(heldOut)}{" "}
      <span className={badge.cls}>({badge.text})</span>
    </div>
  );
}

function Notice({ tone, children }: { tone: "amber" | "sky"; children: React.ReactNode }) {
  const cls = tone === "amber" ? "border-amber-700 bg-amber-950/40 text-amber-200" : "border-sky-800 bg-sky-950/40 text-sky-200";
  return <div role="note" className={`rounded border px-2 py-1.5 text-[11px] ${cls}`}>{children}</div>;
}
