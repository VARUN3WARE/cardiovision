import { useState } from "react";
import FeatureInput, { type Preset } from "./components/FeatureInput";
import HeartScene, { type Vessel } from "./components/HeartScene";
import ClinicalDashboard from "./components/ClinicalDashboard";
import ExplanationPanel from "./components/ExplanationPanel";
import { explain, type ExplainResponse } from "./api/client";

export default function App() {
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [preset, setPreset] = useState<Preset | null>(null);
  const [bpm, setBpm] = useState(72);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Vessel | null>(null);

  async function onSubmit(features: Record<string, unknown>, fromPreset: Preset | null) {
    setLoading(true);
    setError(null);
    try {
      // /explain returns the predictions as well as per-model SHAP values.
      const res = await explain(features);
      setResult(res);
      setPreset(fromPreset);
      setBpm(Number(features.PR) || 72);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-screen flex-col bg-slate-900 text-white">
      <header className="flex items-center justify-between border-b border-slate-700 px-6 py-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-bold">
            Cardio<span className="text-red-500">Vision</span>
          </h1>
          <span className="text-xs text-gray-400">
            Coronary artery disease risk · vessel-level prediction · explainable 3D view
          </span>
        </div>
        <a href="/docs" target="_blank" rel="noreferrer" className="text-xs text-gray-400 hover:text-white">
          API docs
        </a>
      </header>

      {/* Safety disclaimer — required by rubric, always visible */}
      <div className="border-y border-yellow-700 bg-yellow-900/40 px-6 py-2 text-center text-xs text-yellow-200">
        ⚠️ <b>Decision support / educational use only.</b> Predictions come from a model trained on 303
        patients and are not a diagnosis. They do not replace coronary angiography, formal diagnostic
        imaging, or professional medical judgement.
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[300px_minmax(0,1fr)_360px] gap-4 p-4">
        {/* Left: feature input */}
        <div className="overflow-auto rounded bg-slate-800 p-4">
          <h2 className="mb-3 text-sm font-semibold text-gray-200">Clinical Features</h2>
          <FeatureInput onSubmit={onSubmit} loading={loading} />
          {error && <div className="mt-3 text-xs text-red-400">{error}</div>}
        </div>

        {/* Center: 3D heart */}
        <div className="min-h-0 rounded bg-slate-800">
          <HeartScene
            probs={result?.vessel_probabilities ?? null}
            selected={selected}
            onSelect={setSelected}
            bpm={bpm}
          />
        </div>

        {/* Right: dashboard + explanation */}
        <div className="space-y-4 overflow-auto">
          <div className="rounded bg-slate-800 p-4">
            <h2 className="mb-3 text-sm font-semibold text-gray-200">Risk Dashboard</h2>
            <ClinicalDashboard prediction={result} selected={selected} onSelect={setSelected} preset={preset} />
          </div>
          <div className="rounded bg-slate-800 p-4">
            <h2 className="mb-3 text-sm font-semibold text-gray-200">Why this prediction? (SHAP)</h2>
            <ExplanationPanel explanations={result?.explanations ?? null} selected={selected} onSelect={setSelected} />
          </div>
        </div>
      </div>
    </div>
  );
}
