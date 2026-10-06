import { useState } from "react";
import FeatureInput from "./components/FeatureInput";
import HeartScene, { type ArteryProbs } from "./components/HeartScene";
import ClinicalDashboard from "./components/ClinicalDashboard";
import ExplanationPanel from "./components/ExplanationPanel";
import { predict, explain, type PredictionResponse, type ShapExplanation } from "./api/client";

export default function App() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [shap, setShap] = useState<ShapExplanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<"lad" | "lcx" | "rca" | null>(null);

  const probs: ArteryProbs = prediction
    ? prediction.vessel_probabilities
    : { lad: 0, lcx: 0, rca: 0 };

  async function onSubmit(features: Record<string, unknown>) {
    setLoading(true);
    setError(null);
    try {
      const [pred, exp] = await Promise.all([
        predict(features),
        explain(features),
      ]);
      setPrediction(pred);
      setShap(exp.explanation);
      setSelected(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen flex-col bg-slate-900 text-white">
      <header className="flex items-center justify-between border-b border-slate-700 px-6 py-3">
        <h1 className="text-lg font-bold">CardioVision</h1>
        <span className="text-xs text-gray-400">
          Cardiovascular Risk Visualization & Prediction
        </span>
      </header>

      {/* Safety disclaimer — required by rubric */}
      <div className="bg-yellow-900/40 border-y border-yellow-700 px-6 py-2 text-center text-xs text-yellow-200">
        ⚠️ For decision support / educational purposes only. Not a substitute for
        formal diagnostic imaging or professional medical advice.
      </div>

      <div className="grid flex-1 grid-cols-[320px_1fr_360px] gap-4 p-4">
        {/* Left: feature input */}
        <div className="overflow-auto rounded bg-slate-800 p-4">
          <h2 className="mb-3 text-sm font-semibold text-gray-200">Clinical Features</h2>
          <FeatureInput onSubmit={onSubmit} loading={loading} />
          {error && <div className="mt-3 text-xs text-red-400">{error}</div>}
        </div>

        {/* Center: 3D heart */}
        <div className="rounded bg-slate-800">
          <HeartScene probs={probs} onSelect={setSelected} selected={selected} />
        </div>

        {/* Right: dashboard + explanation */}
        <div className="space-y-4 overflow-auto">
          <div className="rounded bg-slate-800 p-4">
            <h2 className="mb-3 text-sm font-semibold text-gray-200">Risk Dashboard</h2>
            <ClinicalDashboard prediction={prediction} selected={selected} />
          </div>
          <div className="rounded bg-slate-800 p-4">
            <h2 className="mb-3 text-sm font-semibold text-gray-200">Explanation (SHAP)</h2>
            <ExplanationPanel explanation={shap} selected={selected} />
          </div>
        </div>
      </div>
    </div>
  );
}
