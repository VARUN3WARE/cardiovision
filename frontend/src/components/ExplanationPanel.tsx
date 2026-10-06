import type { ExplainResponse, FeaturesResponse, MetricsResponse, Target } from "../api/client";
import { FEATURE_LABELS, formatFeatureValue } from "../featureLabels";
import { formatPct, reliability } from "../risk";
import type { Vessel } from "./HeartScene";

interface Props {
  result: ExplainResponse | null;
  metrics: MetricsResponse | null;
  info: FeaturesResponse | null;
  selected: Vessel | null;
  onSelect: (v: Vessel | null) => void;
}

const TOP_N = 8;
const BAR_SCALE = 1.5; // log-odds that fills half the bar width (fixed, so tabs are comparable)
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

interface Row {
  name: string;
  label: string;
  value: number;
  shap: number;
  derived: boolean;
  imputed: boolean;
  sources: string[];
}

function strength(shap: number) {
  const a = Math.abs(shap);
  return a >= 1 ? "strong" : a >= 0.3 ? "moderate" : "slight";
}

export default function ExplanationPanel({ result, metrics, info, selected, onSelect }: Props) {
  if (!result) {
    return <div className="text-sm text-gray-400">The main reasons behind each estimate appear here.</div>;
  }
  const model: Target = selected ?? "cad";
  const exp = result.explanations[model];
  const probability = result.results[model].probability;
  const imputedColumns = new Set(result.imputed.map((k) => info?.schema_to_column[k] ?? k));

  const rows: Row[] = exp.feature_names.map((name, i) => {
    const sources = info?.derived_sources[name] ?? [];
    return {
      name,
      label: FEATURE_LABELS[name]?.label ?? name,
      value: exp.feature_values[i],
      shap: exp.shap_values[i],
      derived: exp.derived[i],
      // A derived score counts as imputed only if all its inputs were imputed.
      imputed: exp.derived[i] ? sources.length > 0 && sources.every((s) => imputedColumns.has(s)) : imputedColumns.has(name),
      sources,
    };
  });
  const ranked = [...rows].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  const top = ranked.slice(0, TOP_N);
  const restShap = ranked.slice(TOP_N).reduce((s, r) => s + r.shap, 0);
  const m = metrics?.metrics[model];
  const weak = m ? reliability(m.cv.roc_auc.mean).weak : false;
  const what = model === "cad" ? "coronary artery disease" : `${model.toUpperCase()} stenosis`;

  // Waterfall in probability points, adding the top drivers in order of magnitude.
  let acc = exp.base_value;
  const steps = top.map((r) => {
    const before = sigmoid(acc);
    acc += r.shap;
    return { row: r, delta: sigmoid(acc) - before };
  });
  const restDelta = sigmoid(acc + restShap) - sigmoid(acc);

  return (
    <div className="space-y-3">
      <div role="tablist" aria-label="Model" className="flex gap-1 text-xs">
        {(["cad", "lad", "lcx", "rca"] as const).map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={model === t}
            onClick={() => onSelect(t === "cad" ? null : t)}
            className={`flex-1 rounded px-2 py-1 ${model === t ? "bg-red-700 text-white" : "bg-slate-900 text-gray-300 hover:bg-slate-700"}`}
          >
            {t === "cad" ? "Overall" : t.toUpperCase()}
          </button>
        ))}
      </div>

      <p className="text-sm text-gray-200">{summarySentence(ranked, probability, what)}</p>
      {weak && m && (
        <p className="text-[11px] text-amber-300">
          This vessel model is weaker (cross-validated AUC {m.cv.roc_auc.mean.toFixed(2)}): treat these drivers as indicative only.
        </p>
      )}

      <div className="text-[11px] text-gray-400">
        Each bar shows how much a finding pushed the estimate <span className="text-red-400">up</span> or{" "}
        <span className="text-sky-400">down</span> from the model's starting point.
      </div>

      <ul className="space-y-1.5">
        {top.map((r) => {
          const w = Math.min(Math.abs(r.shap) / BAR_SCALE, 1) * 50;
          const up = r.shap >= 0;
          return (
            <li key={r.name} className={`text-xs ${r.imputed ? "opacity-60" : ""}`}>
              <div className="flex justify-between gap-2">
                <span className="truncate text-gray-200" title={r.derived ? `Calculated from: ${r.sources.join(", ")}` : undefined}>
                  {r.label}
                  {r.derived && <span className="ml-1 rounded bg-slate-700 px-1 text-[9px] text-gray-300">score</span>}
                </span>
                <span className="shrink-0 text-gray-400">
                  {r.imputed ? <i>typical value</i> : formatFeatureValue(r.name, r.value)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="relative h-2 flex-1 rounded bg-slate-700/60" aria-hidden>
                  <div className="absolute left-1/2 top-0 h-2 w-px bg-slate-400" />
                  <div
                    className={`absolute top-0 h-2 ${up ? "rounded-r bg-red-500" : "rounded-l bg-sky-500"}`}
                    style={up ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
                  />
                </div>
                <span className={`w-28 shrink-0 text-right text-[10px] ${up ? "text-red-300" : "text-sky-300"}`}>
                  {strength(r.shap)} {up ? "↑" : "↓"} ×{Math.exp(r.shap).toFixed(1)} odds
                </span>
              </div>
            </li>
          );
        })}
      </ul>

      <div className="rounded bg-slate-900/70 p-2 text-[11px] text-gray-400">
        <div className="mb-1 font-semibold text-gray-300">From starting point to this patient</div>
        <div className="flex justify-between"><span>Model starting point</span><span>{formatPct(sigmoid(exp.base_value))}</span></div>
        {steps.map(({ row, delta }) => (
          <div key={row.name} className="flex justify-between">
            <span className="truncate pr-2">{row.label}</span>
            <span className={delta >= 0 ? "text-red-300" : "text-sky-300"}>{pts(delta)}</span>
          </div>
        ))}
        <div className="flex justify-between">
          <span>{rows.length - top.length} other findings</span>
          <span className={restDelta >= 0 ? "text-red-300" : "text-sky-300"}>{pts(restDelta)}</span>
        </div>
        <div className="mt-1 flex justify-between border-t border-slate-700 pt-1 font-semibold text-gray-200">
          <span>Estimate for this patient</span><span>{formatPct(probability)}</span>
        </div>
        <details className="mt-1">
          <summary className="cursor-pointer text-gray-500">Technical details</summary>
          <div className="mt-1 space-y-0.5 text-gray-500">
            <div>Exact additive attributions in log-odds (linear terms and TreeSHAP, averaged over the model blend).</div>
            <div>Starting point {exp.base_value.toFixed(2)} + contributions {(exp.shap_values.reduce((a, b) => a + b, 0)).toFixed(2)} = {(exp.base_value + exp.shap_values.reduce((a, b) => a + b, 0)).toFixed(2)} log-odds.</div>
            {m && <div>Model: {m.model}.</div>}
            <div>Steps in points depend on order; bars and odds ratios do not.</div>
          </div>
        </details>
      </div>
    </div>
  );
}

function pts(delta: number) {
  const v = Math.round(delta * 100);
  if (v === 0) return delta >= 0 ? "+<1 pt" : "−<1 pt";
  return `${v > 0 ? "+" : "−"}${Math.abs(v)} pts`;
}

function summarySentence(ranked: Row[], p: number, what: string) {
  const level = p >= 0.66 ? "high" : p >= 0.33 ? "intermediate" : "low";
  const fmt = (r: Row) => r.imputed ? r.label.toLowerCase() + " (typical value)" : describe(r);
  const ups = ranked.filter((r) => r.shap > 0.15).slice(0, 3).map(fmt);
  const down = ranked.find((r) => r.shap < -0.15);
  let s = `Estimated probability of ${what} is ${level} (${formatPct(p)})`;
  if (ups.length) s += `, mainly because of ${joinList(ups)}`;
  if (down) s += `${ups.length ? ";" : ","} ${fmt(down)} lowered it`;
  return s + ".";
}

function describe(r: Row) {
  const value = formatFeatureValue(r.name, r.value);
  const label = r.label.charAt(0).toLowerCase() + r.label.slice(1);
  if (value === "Yes") return label;
  if (value === "No") return `no ${label}`;
  return `${label} ${value}`;
}

function joinList(items: string[]) {
  return items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}
