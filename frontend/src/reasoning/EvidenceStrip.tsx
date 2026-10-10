import type { ShapExplanation } from "../api/client";
import { formatPct } from "../risk";
import {
  DOWN_COLOR, FILL_RULE, FILL_TEXT, GROUP_SCALE, MIN_EFFECT, UP_COLOR, filledNote, groupAttributions, groupWeight, oddsText,
  sigmoid, type FillKind,
} from "./dataGroups";
import { reasoningHover, useReasoningHover } from "./hoverStore";

interface Props {
  exp: ShapExplanation;
  /** Model columns the API filled in because the form left them empty, and how (dataGroups.typicalColumns). */
  typical?: ReadonlyMap<string, FillKind> | null;
}

/**
 * "Evidence by data type": the selected model's exact attributions summed per data type (the same numbers
 * the 3D reasoning streams are drawn from). Hovering or focusing a row highlights its stream in 3D, and
 * vice versa.
 */
export default function EvidenceStrip({ exp, typical = null }: Props) {
  const all = groupAttributions(exp, typical);
  const { base, logOdds, groups } = all;
  const hovered = useReasoningHover();
  const capped = groups.some((g) => Math.abs(g.sum) > GROUP_SCALE);
  const filled = filledNote(all.filled, all.inputs);
  return (
    <div className="rounded border border-slate-700 bg-slate-900/40 p-2 text-[11px]" onMouseLeave={() => reasoningHover.set(null)}>
      <div className="mb-1">
        <div className="font-semibold text-gray-300">Evidence by data type</div>
        <div className="text-[10px] text-gray-500">
          How much each kind of data moved this estimate: associations in this referral cohort, not causes
        </div>
      </div>
      <ul className="space-y-0.5">
        {groups.map((g) => {
          const up = g.sum >= 0;
          const w = groupWeight(g.sum) * 50;
          const top = g.findings
            .slice(0, 3)
            .map((f) => `${f.label}${f.fill ? ` (${FILL_TEXT[f.fill]})` : ""} ${oddsText(f.shap)}`)
            .join(" · ");
          const note = filledNote(g.filled, g.inputs);
          return (
            <li
              key={g.id}
              tabIndex={0}
              onMouseEnter={() => reasoningHover.set(g.id)}
              onFocus={() => reasoningHover.set(g.id)}
              onBlur={() => reasoningHover.get() === g.id && reasoningHover.set(null)}
              title={`${g.label}: ${g.findings.length} inputs, sum ${g.sum >= 0 ? "+" : "−"}${Math.abs(g.sum).toFixed(2)} log-odds. Top: ${top}${note ? `. ${note}` : ""}`}
              className={`grid cursor-default grid-cols-[150px_minmax(0,1fr)_46px] items-center gap-2 rounded px-0.5 outline-none focus-visible:ring-1 focus-visible:ring-slate-400 ${hovered === g.id ? "bg-slate-700/60" : ""}`}
            >
              <span className="truncate text-gray-300">{g.label}</span>
              <span className="relative h-2 rounded bg-slate-700/60" aria-hidden>
                <span className="absolute left-1/2 top-0 h-2 w-px bg-slate-400" />
                <span
                  className={`absolute top-0 h-2 ${up ? "rounded-r" : "rounded-l"}`}
                  style={{ background: up ? UP_COLOR : DOWN_COLOR, ...(up ? { left: "50%" } : { right: "50%" }), width: `${w}%` }}
                />
              </span>
              <span className={`text-right ${Math.abs(g.sum) < MIN_EFFECT ? "text-gray-500" : up ? "text-red-300" : "text-sky-300"}`}>
                {oddsText(g.sum)}
              </span>
            </li>
          );
        })}
      </ul>
      <div className="mt-1 text-[10px] text-gray-500">
        Starting odds ({formatPct(sigmoid(base))}) × these factors = final odds ({formatPct(sigmoid(logOdds))}). Each row
        is the exact sum of that data type&apos;s attributions{capped && ` (bars stop at ±${GROUP_SCALE} log-odds)`}; grey
        means little effect. The same numbers drive the 3D Reasoning streams.
        {filled && <> {filled}: {FILL_RULE}.</>}
      </div>
      <details className="text-[10px] text-gray-500">
        <summary className="cursor-pointer">How calculated scores are grouped</summary>
        Each score counts once, in the data type it is computed from: risk factor count → history; ECG abnormality
        count → ECG; lipid ratios, eGFR, NLR and glucose ≥ 126 → labs; EF below 50% and any wall-motion abnormality →
        echo; systolic BP ≥ 140 → symptoms &amp; exam. The pre-test score is the Diamond-Forrester pre-test
        probability and its two ingredients as the model encodes them (chest pain class; age × male).
      </details>
    </div>
  );
}
