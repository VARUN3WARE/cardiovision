import type { PredictionResponse } from "../api/client";
import type { Preset } from "./FeatureInput";
import type { Vessel } from "./HeartScene";

interface Props {
  prediction: PredictionResponse | null;
  selected: Vessel | null;
  onSelect: (v: Vessel | null) => void;
  preset: Preset | null;
}

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`;
}

function riskColor(p: number) {
  if (p >= 0.66) return "text-red-400";
  if (p >= 0.33) return "text-yellow-300";
  return "text-green-400";
}

function riskLabel(p: number) {
  if (p >= 0.66) return "High";
  if (p >= 0.33) return "Moderate";
  return "Low";
}

export default function ClinicalDashboard({ prediction, selected, onSelect, preset }: Props) {
  if (!prediction) {
    return (
      <div className="text-sm text-gray-400">
        Load an example patient or enter clinical features, then click <b>Predict Risk</b>.
      </div>
    );
  }
  const vessels = prediction.vessel_probabilities;
  return (
    <div className="space-y-3">
      <div>
        <div className="text-sm text-gray-300">Overall CAD probability</div>
        <div className="flex items-baseline gap-3">
          <div className={`text-4xl font-bold ${riskColor(prediction.cad_probability)}`}>
            {pct(prediction.cad_probability)}
          </div>
          <div className={`text-sm font-semibold ${riskColor(prediction.cad_probability)}`}>
            {riskLabel(prediction.cad_probability)} risk
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {(["lad", "lcx", "rca"] as const).map((v) => (
          <button
            key={v}
            onClick={() => onSelect(selected === v ? null : v)}
            className={`rounded p-2 text-center transition ${
              selected === v ? "bg-slate-700 ring-2 ring-white" : "bg-slate-900 hover:bg-slate-700"
            }`}
          >
            <div className="text-xs uppercase text-gray-300">{v}</div>
            <div className={`text-xl font-bold ${riskColor(vessels[v])}`}>{pct(vessels[v])}</div>
            {preset && (
              <div className={`mt-0.5 text-[10px] ${preset.ground_truth[v] ? "text-red-300" : "text-green-300"}`}>
                Cath: {preset.ground_truth[v] ? "stenotic" : "normal"}
              </div>
            )}
          </button>
        ))}
      </div>

      {preset && (
        <div className="rounded border border-slate-700 bg-slate-900/60 p-2 text-[11px] text-gray-400">
          <b className="text-gray-200">Dataset patient #{preset.patient_index}</b>: angiography ground truth is{" "}
          <b className="text-gray-200">{preset.description.toLowerCase()}</b>. Shown for demonstration;
          this patient was part of the training data.
        </div>
      )}
    </div>
  );
}
