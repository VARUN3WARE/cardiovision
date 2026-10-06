import type { ExplainResponse } from "../api/client";
import { FEATURE_LABELS, formatFeatureValue } from "../featureLabels";
import type { Vessel } from "./HeartScene";

interface Props {
  explanations: ExplainResponse["explanations"] | null;
  selected: Vessel | null;
  onSelect: (v: Vessel | null) => void;
}

const TOP_N = 10;
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export default function ExplanationPanel({ explanations, selected, onSelect }: Props) {
  if (!explanations) {
    return <div className="text-sm text-gray-400">Feature attributions appear after prediction.</div>;
  }
  const model = selected ?? "cad";
  const exp = explanations[model];

  const rows = exp.feature_names.map((name, i) => ({
    name,
    value: exp.feature_values[i],
    shap: exp.shap_values[i],
  }));
  const ranked = [...rows].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)).slice(0, TOP_N);
  const maxAbs = Math.max(...ranked.map((r) => Math.abs(r.shap)), 1e-6);
  const total = rows.reduce((s, r) => s + r.shap, 0);
  const rest = total - ranked.reduce((s, r) => s + r.shap, 0);

  return (
    <div className="space-y-3">
      <div className="flex gap-1 text-xs">
        {(["cad", "lad", "lcx", "rca"] as const).map((m) => (
          <button
            key={m}
            onClick={() => onSelect(m === "cad" ? null : m)}
            className={`flex-1 rounded px-2 py-1 ${model === m ? "bg-red-700 text-white" : "bg-slate-900 text-gray-300 hover:bg-slate-700"}`}
          >
            {m === "cad" ? "Overall" : m.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="text-xs text-gray-400">
        Top drivers of the <b className="text-gray-200">{model === "cad" ? "overall CAD" : `${model.toUpperCase()} stenosis`}</b>{" "}
        prediction for this patient. Bars are SHAP values (log-odds):{" "}
        <span className="text-red-400">red raises risk</span>, <span className="text-sky-400">blue lowers it</span>.
      </div>

      <div className="space-y-1.5">
        {ranked.map((r) => {
          const w = (Math.abs(r.shap) / maxAbs) * 50;
          const positive = r.shap >= 0;
          return (
            <div key={r.name} className="text-xs" title={`SHAP ${r.shap >= 0 ? "+" : ""}${r.shap.toFixed(3)}`}>
              <div className="flex justify-between gap-2">
                <span className="truncate text-gray-200">{FEATURE_LABELS[r.name]?.label ?? r.name}</span>
                <span className="shrink-0 text-gray-400">{formatFeatureValue(r.name, r.value)}</span>
              </div>
              <div className="relative mt-0.5 h-2 w-full rounded bg-slate-700/60">
                <div className="absolute left-1/2 top-0 h-2 w-px bg-slate-400" />
                <div
                  className={`absolute top-0 h-2 ${positive ? "rounded-r bg-red-500" : "rounded-l bg-sky-500"}`}
                  style={positive ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="rounded bg-slate-900/70 p-2 text-[11px] text-gray-400">
        <div className="flex justify-between">
          <span>Model baseline</span>
          <span>{(sigmoid(exp.base_value) * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span>+ top {TOP_N} features</span>
          <span>{(total - rest >= 0 ? "+" : "") + (total - rest).toFixed(2)} log-odds</span>
        </div>
        <div className="flex justify-between">
          <span>+ {rows.length - TOP_N} other features</span>
          <span>{(rest >= 0 ? "+" : "") + rest.toFixed(2)} log-odds</span>
        </div>
        <div className="mt-1 flex justify-between border-t border-slate-700 pt-1 text-gray-200">
          <span>= This patient</span>
          <span>{(sigmoid(exp.base_value + total) * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}
