import type { ShapExplanation } from "../api/client";

interface Props {
  explanation: ShapExplanation | null;
  selected: "lad" | "lcx" | "rca" | null;
}

export default function ExplanationPanel({ explanation, selected }: Props) {
  if (!explanation) {
    return <div className="text-gray-400 text-sm">Feature attributions appear after prediction.</div>;
  }
  // Rank features by absolute SHAP value, top 8.
  const ranked = explanation.feature_names
    .map((name, i) => ({
      name,
      value: explanation.feature_values[i],
      shap: explanation.shap_values[i],
    }))
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    .slice(0, 8);

  const maxAbs = Math.max(...ranked.map((r) => Math.abs(r.shap)), 1e-6);

  return (
    <div className="space-y-3">
      <div className="text-gray-300 text-sm font-semibold">
        {selected ? `${selected.toUpperCase()} risk drivers` : "Overall CAD risk drivers"}
      </div>
      <div className="text-gray-500 text-xs">
        SHAP value = contribution toward CAD risk (red) vs. no-CAD (blue).
      </div>
      {ranked.map((r) => {
        const w = (Math.abs(r.shap) / maxAbs) * 100;
        const positive = r.shap >= 0;
        return (
          <div key={r.name} className="text-sm">
            <div className="flex justify-between text-gray-300">
              <span>{r.name}</span>
              <span className="text-gray-500">value: {r.value.toFixed(2)}</span>
            </div>
            <div className="relative mt-1 h-2 w-full rounded bg-slate-700">
              <div
                className={`absolute h-2 rounded ${positive ? "bg-red-500" : "bg-blue-500"}`}
                style={{ width: `${w}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
