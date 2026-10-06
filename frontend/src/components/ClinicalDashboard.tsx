import type { PredictionResponse } from "../api/client";

interface Props {
  prediction: PredictionResponse | null;
  selected: "lad" | "lcx" | "rca" | null;
}

function pct(p: number) {
  return `${(p * 100).toFixed(1)}%`;
}

function riskColor(p: number) {
  if (p >= 0.66) return "text-red-400";
  if (p >= 0.33) return "text-yellow-300";
  return "text-green-400";
}

export default function ClinicalDashboard({ prediction, selected }: Props) {
  if (!prediction) {
    return (
      <div className="text-gray-400 text-sm">
        Enter clinical features and click <b>Predict Risk</b> to see results.
      </div>
    );
  }
  const vessels = prediction.vessel_probabilities;
  return (
    <div className="space-y-4">
      <div className="rounded bg-slate-800 p-4">
        <div className="text-gray-300 text-sm">Overall CAD Risk</div>
        <div className={`text-4xl font-bold ${riskColor(prediction.cad_probability)}`}>
          {pct(prediction.cad_probability)}
        </div>
        <div className="text-gray-400 text-sm">
          Prediction: {prediction.cad_prediction ? "CAD present" : "No CAD"}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {(["lad", "lcx", "rca"] as const).map((v) => (
          <div
            key={v}
            className={`rounded p-3 text-center ${selected === v ? "ring-2 ring-white" : "bg-slate-800"}`}
          >
            <div className="text-gray-300 text-xs uppercase">{v}</div>
            <div className={`text-xl font-bold ${riskColor(vessels[v])}`}>
              {pct(vessels[v])}
            </div>
          </div>
        ))}
      </div>
      <div className="text-gray-500 text-xs">
        Click an artery in the 3D view to inspect vessel-specific risk.
      </div>
    </div>
  );
}
