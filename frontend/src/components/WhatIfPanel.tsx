import type { Dispatch } from "react";
import type { ExplainResponse, FeaturesResponse } from "../api/client";
import { BAND_COLOR, formatPct, type Band } from "../risk";
import {
  CONTRASTS, LEVERS, LOCKED, LOCKED_NOTE, SMOKING, baselineOf, contrastDirection, scenarioDeltas, sliderRange,
  type Lever, type Payload, type Scenario, type ScenarioAction,
} from "../whatif/scenario";

interface Props {
  open: boolean;
  onToggle: () => void;
  scenario: Scenario;
  dispatch: Dispatch<ScenarioAction>;
  active: boolean;
  /** The form's request body (the patient's recorded values). */
  payload: Payload;
  /** The patient's own estimate. */
  base: ExplainResponse | null;
  /** The scenario's estimate (null while inactive or before its first reply). */
  scenarioResult: ExplainResponse | null;
  info: FeaturesResponse | null;
  updating: boolean;
  error: string | null;
  onDragStart: () => void;
}

/** "What moves this estimate?": modifiable risk factors and diagnostic contrasts, re-run through the models. */
export default function WhatIfPanel(props: Props) {
  const { open, onToggle, scenario, dispatch, active, payload, base, scenarioResult, info, updating, error, onDragStart } = props;
  const fi = info?.feature_info ?? {};
  const value = (key: string, column: string) => baselineOf(key, column, payload, base);
  const imputed = new Set(base?.imputed ?? []);
  const smoker = value(SMOKING.key, SMOKING.column) === 1;

  return (
    <div className="rounded bg-slate-800 p-4">
      <button onClick={onToggle} aria-expanded={open} className="flex w-full items-center justify-between gap-2 text-left">
        <h2 className="text-sm font-semibold text-gray-200">
          What-if <span className="font-normal text-gray-400">· what moves this estimate?</span>
        </h2>
        <span className="flex shrink-0 items-center gap-2">
          {active && <span className="rounded bg-sky-600 px-1.5 py-0.5 text-[10px] font-semibold whitespace-nowrap text-white">Scenario active</span>}
          <span className="text-xs text-gray-400" aria-hidden>{open ? "▾" : "▸"}</span>
        </span>
      </button>

      {open && (
        <div className="mt-3 space-y-3">
          <div role="note" className="rounded border border-sky-800 bg-sky-950/40 px-2 py-1.5 text-[11px] text-sky-200">
            <b>How the model's estimate responds (sensitivity), not the effect of treatment.</b> Associations learned from
            303 referred patients, not causes. The form keeps the patient's recorded values.
          </div>

          {!base ? (
            <div className="text-xs text-gray-400">Run an estimate first.</div>
          ) : (
            <>
              <DeltaStrip base={base} scen={active ? scenarioResult : null} updating={updating} error={error} />

              <Group title="Modifiable risk factors">
                {LEVERS.map((l) => (
                  <LeverSlider
                    key={l.key}
                    lever={l}
                    baseline={value(l.key, l.column)}
                    current={scenario.levers[l.key]}
                    info={fi[l.column]}
                    imputed={imputed.has(l.key)}
                    onChange={(v, baseline) => dispatch({ type: "lever", key: l.key, value: v, baseline })}
                    onDragStart={onDragStart}
                  />
                ))}
                <Switch
                  on={scenario.quitSmoking}
                  disabled={!smoker}
                  onChange={(on) => dispatch({ type: "quit", on })}
                  label="Quit smoking (current → ex-smoker)"
                  hint={smoker ? undefined : "Not recorded as a current smoker."}
                />
                <div className="space-y-1 rounded border border-slate-700 bg-slate-900/50 px-2 py-1.5">
                  {LOCKED.map((l) => {
                    const v = value(l.key, l.column);
                    return (
                      <div key={l.key} className="flex items-center justify-between text-xs text-gray-400">
                        <span>{l.label}</span>
                        <span>{v === undefined ? "–" : `${round(v)} ${l.unit}`} · locked</span>
                      </div>
                    );
                  })}
                  <div className="text-[10px] leading-snug text-gray-500">{LOCKED_NOTE}</div>
                </div>
              </Group>

              <Group title="Diagnostic contrast: if this finding were absent/normal">
                <div className="text-[10px] leading-snug text-gray-500">
                  Shows how much of the estimate rests on each finding. Findings already absent are shown as if present
                  (contrast values as in the model verification).
                </div>
                {CONTRASTS.map((c) => {
                  const v = value(c.key, c.column);
                  const dir = contrastDirection(c, v);
                  const recorded = dir === "normal" ? c.abnormalText : c.absentText ?? c.normalText;
                  const target = dir === "normal" ? c.normalText : c.abnormalText;
                  const shownRecorded = c.id === "ef" && v !== undefined ? `${round(v)}%` : c.id === "rwma" && v !== undefined ? `grade ${round(v)}` : recorded;
                  return (
                    <Switch
                      key={c.id}
                      on={!!scenario.contrasts[c.id]}
                      disabled={!dir}
                      onChange={(on) => dispatch({ type: "contrast", id: c.id, on })}
                      label={`${c.label}: ${shownRecorded} → ${target}`}
                      hint={[
                        dir === "abnormal" ? (c.id === "ef" ? "Recorded as 50% or more: shown as if reduced." : "Recorded as absent: shown as if present.") : "",
                        imputed.has(c.key) ? "Not entered: cohort typical value." : "",
                      ].filter(Boolean).join(" ") || undefined}
                    />
                  );
                })}
              </Group>

              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] text-gray-500">Shaded track: outside the cohort's 5th–95th percentile.</span>
                <button
                  onClick={() => dispatch({ type: "reset" })}
                  disabled={!active}
                  className="shrink-0 rounded bg-slate-600 px-3 py-1 text-xs font-semibold text-white hover:bg-slate-500 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Reset to patient
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

const round = (v: number) => Math.round(v * 10) / 10;

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <fieldset className="space-y-2">
      <legend className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-gray-400">{title}</legend>
      {children}
    </fieldset>
  );
}

/** Patient vs scenario per model, in band colours. */
function DeltaStrip({ base, scen, updating, error }: {
  base: ExplainResponse; scen: ExplainResponse | null; updating: boolean; error: string | null;
}) {
  if (error) return <div role="alert" className="text-xs text-red-300">{error}</div>;
  if (!scen) {
    return (
      <div className="text-xs text-gray-400">
        {updating ? "Calculating the scenario…" : "Move a slider or switch a finding to see how each estimate responds."}
      </div>
    );
  }
  return (
    <div aria-live="polite">
      <div className="grid grid-cols-2 gap-1.5">
        {scenarioDeltas(base, scen).map((d) => {
          const pts = Math.round(d.to * 100) - Math.round(d.from * 100); // matches the whole percents shown
          return (
            <div key={d.target} className="whitespace-nowrap rounded bg-slate-900 px-2 py-1 text-xs">
              <span className="font-semibold text-gray-300">{d.label}</span>{" "}
              <span style={{ color: BAND_COLOR[d.fromBand as Band] }}>{formatPct(d.from)}</span>
              <span className="text-gray-500"> → </span>
              <b style={{ color: BAND_COLOR[d.toBand as Band] }}>{formatPct(d.to)}</b>
              <span className="ml-1 text-[10px] text-gray-500">{pts === 0 ? "±0" : `${pts > 0 ? "+" : "−"}${Math.abs(pts)}`} pts</span>
            </div>
          );
        })}
      </div>
      <div className="mt-1 h-3 text-[10px] text-gray-500">{updating ? "Updating…" : ""}</div>
    </div>
  );
}

function LeverSlider({ lever, baseline, current, info, imputed, onChange, onDragStart }: {
  lever: Lever;
  baseline: number | undefined;
  current: number | undefined;
  info: FeaturesResponse["feature_info"][string] | undefined;
  imputed: boolean;
  onChange: (v: number, baseline: number | undefined) => void;
  onDragStart: () => void;
}) {
  const { lo, hi, p5, p95 } = sliderRange(info, baseline, lever.step);
  const value = current ?? baseline ?? (lo + hi) / 2;
  const pos = (v: number) => `${Math.max(0, Math.min(100, ((v - lo) / (hi - lo || 1)) * 100))}%`;
  const shade = "#273449";
  const inner = "#64748b";
  const track = `linear-gradient(to right, ${shade} 0%, ${shade} ${pos(p5)}, ${inner} ${pos(p5)}, ${inner} ${pos(p95)}, ${shade} ${pos(p95)}, ${shade} 100%)`;
  const changed = current !== undefined && current !== baseline;
  const id = `whatif-${lever.key}`;
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <label htmlFor={id} className="text-gray-300">{lever.label}</label>
        <span className={changed ? "font-semibold text-sky-300" : "text-gray-300"}>
          {round(value)} {lever.unit}
          {changed && baseline !== undefined && <span className="ml-1 font-normal text-gray-500">(recorded {round(baseline)}{imputed ? ", typical" : ""})</span>}
        </span>
      </div>
      <div className="relative h-5">
        <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full" style={{ background: track }} />
        {baseline !== undefined && (
          <div title={`Recorded: ${round(baseline)} ${lever.unit}`} className="absolute top-0.5 h-4 w-0.5 -translate-x-1/2 bg-white/70" style={{ left: pos(baseline) }} />
        )}
        <input
          id={id}
          type="range"
          min={lo}
          max={hi}
          step={lever.step}
          value={value}
          onPointerDown={onDragStart}
          onChange={(e) => onChange(Number(e.target.value), baseline)}
          aria-valuetext={`${round(value)} ${lever.unit}`}
          className="relative h-5 w-full cursor-pointer appearance-none bg-transparent focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400
            [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-slate-900 [&::-moz-range-thumb]:bg-sky-400
            [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-slate-900 [&::-webkit-slider-thumb]:bg-sky-400"
        />
      </div>
      <div className="flex justify-between text-[10px] text-gray-500">
        <span>{round(lo)}</span>
        {lever.note && <span className="px-2 text-center">{lever.note}</span>}
        <span>{round(hi)}</span>
      </div>
    </div>
  );
}

function Switch({ on, disabled, onChange, label, hint }: {
  on: boolean; disabled?: boolean; onChange: (on: boolean) => void; label: string; hint?: string;
}) {
  return (
    <button
      role="switch"
      aria-checked={on}
      disabled={disabled}
      onClick={() => onChange(!on)}
      className="flex w-full items-start gap-2 text-left text-xs disabled:cursor-not-allowed disabled:opacity-50"
    >
      <span className={`mt-0.5 flex h-4 w-7 shrink-0 items-center rounded-full p-0.5 transition ${on ? "bg-sky-500" : "bg-slate-600"}`}>
        <span className={`h-3 w-3 rounded-full bg-white transition ${on ? "translate-x-3" : ""}`} />
      </span>
      <span>
        <span className={on ? "font-semibold text-sky-300" : "text-gray-300"}>{label}</span>
        {hint && <span className="block text-[10px] text-gray-500">{hint}</span>}
      </span>
    </button>
  );
}
