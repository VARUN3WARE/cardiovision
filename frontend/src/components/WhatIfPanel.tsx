import type { Dispatch, ReactNode } from "react";
import type { ExplainResponse, FeaturesResponse } from "../api/client";
import { BAND_COLOR, formatPct, type Band } from "../risk";
import {
  CONTRASTS, LEVERS, LOCKED, LOCKED_NOTE, SMOKING, baselineOf, contrastView, deltaPts, scenarioDeltas, sliderRange,
  type Contrast, type Lever, type Payload, type Scenario, type ScenarioAction, type ScenarioView,
} from "../whatif/scenario";

interface Props {
  open: boolean;
  onToggle: () => void;
  /** The scenario as it applies to this patient. */
  scenario: Scenario;
  dispatch: Dispatch<ScenarioAction>;
  on: boolean;
  /** The patient's submitted values: the request body behind `base` (never unsubmitted form edits). */
  payload: Payload | null;
  /** The patient's own estimate. */
  base: ExplainResponse | null;
  /** What the views show (scenarioView). */
  view: ScenarioView;
  error: string | null;
  info: FeaturesResponse | null;
  onDragStart: () => void;
  onRetry: () => void;
}

/** "What moves this estimate?": modifiable risk factors and recorded findings, re-run through the models. */
export default function WhatIfPanel(props: Props) {
  const { open, onToggle, scenario, dispatch, on, payload, base, view, error, info, onDragStart, onRetry } = props;
  const fi = info?.feature_info ?? {};
  const imputed = new Set(base?.imputed ?? []);

  return (
    <div className="rounded bg-slate-800 p-4">
      <button onClick={onToggle} aria-expanded={open} className="flex w-full items-center justify-between gap-2 text-left">
        <h2 className="text-sm font-semibold text-gray-200">
          What-if <span className="font-normal text-gray-400">· what moves this estimate?</span>
        </h2>
        <span className="flex shrink-0 items-center gap-2">
          {on && <span className="rounded bg-sky-600 px-1.5 py-0.5 text-[10px] font-semibold whitespace-nowrap text-white">Scenario active</span>}
          <span className="text-xs text-gray-400" aria-hidden>{open ? "▾" : "▸"}</span>
        </span>
      </button>

      {open && (
        <div className="mt-3 space-y-3">
          <div role="note" className="rounded border border-sky-800 bg-sky-950/40 px-2 py-1.5 text-[11px] text-sky-200">
            <b>How the model&apos;s estimate responds (sensitivity), not the effect of treatment.</b> Associations learned
            from 303 referred patients, not causes. Scenarios start from the values behind the current estimate; the
            form keeps the patient&apos;s recorded values.
          </div>

          {!payload || !base ? (
            <div className="text-xs text-gray-400">Run an estimate first.</div>
          ) : (
            <Controls
              {...{ scenario, dispatch, payload, base, fi, imputed, onDragStart }}
              strip={<DeltaStrip base={base} view={view} error={error} onRetry={onRetry} />}
              reset={
                <div className="flex justify-end">
                  <button
                    onClick={() => dispatch({ type: "reset" })}
                    disabled={!on}
                    className="rounded bg-slate-600 px-3 py-1 text-xs font-semibold text-white hover:bg-slate-500 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Reset to patient
                  </button>
                </div>
              }
            />
          )}
        </div>
      )}
    </div>
  );
}

function Controls({ scenario, dispatch, payload, base, fi, imputed, onDragStart, strip, reset }: {
  scenario: Scenario;
  dispatch: Dispatch<ScenarioAction>;
  payload: Payload;
  base: ExplainResponse;
  fi: FeaturesResponse["feature_info"];
  imputed: Set<string>;
  onDragStart: () => void;
  strip: ReactNode;
  reset: ReactNode;
}) {
  const value = (key: string, column: string) => baselineOf(key, column, payload, base);
  const smoker = value(SMOKING.key, SMOKING.column) === 1;
  const contrasts = CONTRASTS.map((c) => ({ c, ...contrastView(c, payload, base) }));
  const contrastSwitch = ({ c, dir, from, to }: { c: Contrast; dir: "normal" | "abnormal" | null; from: string; to: string }) => {
    const isOn = !!scenario.contrasts[c.id];
    return (
      <Switch
        key={c.id}
        on={isOn}
        disabled={!dir && !isOn}
        onChange={(v) => dispatch({ type: "contrast", id: c.id, on: v })}
        label={`${c.label}: ${from} → ${to}`}
        hint={imputed.has(c.key) ? "Not entered: the model used the default (findings absent, measurements at the cohort median)." : undefined}
      />
    );
  };
  const present = contrasts.filter((x) => x.dir === "normal");
  const absent = contrasts.filter((x) => x.dir === "abnormal");

  return (
    <>
      {strip}

      <Group title="Modifiable risk factors">
        <Hint>Dashed track: outside the cohort&apos;s 5th–95th percentile. White tick: this patient&apos;s value.</Hint>
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
          // A switch that is on can always be switched off.
          disabled={!smoker && !scenario.quitSmoking}
          onChange={(on) => dispatch({ type: "quit", on })}
          label="Quit smoking (current → ex-smoker)"
          hint={smoker ? undefined : "Not recorded as a current smoker."}
        />
        <div className="space-y-1.5 rounded border border-slate-700 bg-slate-900/50 px-2 py-1.5">
          <Hint>{LOCKED_NOTE}</Hint>
          {LOCKED.map((l) => {
            const v = value(l.key, l.column);
            return (
              <div key={l.key}>
                <div className="flex items-baseline justify-between gap-2 text-xs text-gray-300">
                  <span>{l.label}</span>
                  <span className="whitespace-nowrap">
                    {v === undefined ? "–" : `${round(v)} ${l.unit}`}
                    {imputed.has(l.key) ? " (not entered: typical)" : ""} · locked
                  </span>
                </div>
                <Hint>{l.note}</Hint>
              </div>
            );
          })}
        </div>
      </Group>

      <Group title="Flip one recorded finding">
        <Hint>
          How much of the estimate rests on each finding: one model input flipped at a time, to the contrast values
          used in Model verification §7.
        </Hint>
        {present.length > 0 && <SubGroup title="Present → if absent/normal">{present.map(contrastSwitch)}</SubGroup>}
        {absent.length > 0 && <SubGroup title="Absent → if present (hypothetical)">{absent.map(contrastSwitch)}</SubGroup>}
      </Group>

      {reset}
    </>
  );
}

const round = (v: number) => Math.round(v * 10) / 10;

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="space-y-2">
      <legend className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-gray-400">{title}</legend>
      {children}
    </fieldset>
  );
}

function SubGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div role="group" aria-label={title} className="space-y-2 border-l-2 border-slate-700 pl-2">
      <div className="text-[11px] font-semibold text-gray-300">{title}</div>
      {children}
    </div>
  );
}

function Hint({ children }: { children: ReactNode }) {
  return <div className="text-[11px] leading-snug text-gray-400">{children}</div>;
}

/** Patient vs scenario per model, in band colours. Not a live region: the 3D strip announces settled results. */
function DeltaStrip({ base, view, error, onRetry }: {
  base: ExplainResponse; view: ScenarioView; error: string | null; onRetry: () => void;
}) {
  if (view.status === "error") {
    return (
      <div role="alert" className="flex items-center justify-between gap-2 rounded border border-red-800 bg-red-950/40 px-2 py-1 text-xs text-red-200">
        <span>Could not run the scenario: {error}</span>
        <button onClick={onRetry} className="shrink-0 rounded bg-red-700 px-2 py-0.5 font-semibold text-white hover:bg-red-600">
          Retry
        </button>
      </div>
    );
  }
  const scen = view.reply ? view.shown : null;
  if (!scen) {
    return (
      <div className="text-xs text-gray-400">
        {view.status === "calculating" ? "Calculating the scenario…" : "Move a slider or flip a finding to see how each estimate responds."}
      </div>
    );
  }
  return (
    <div>
      <div className="grid grid-cols-2 gap-1.5">
        {scenarioDeltas(base, scen).map((d) => {
          const pts = deltaPts(d.from, d.to); // matches the percentages shown
          return (
            <div key={d.target} className="whitespace-nowrap rounded bg-slate-900 px-2 py-1 text-xs">
              <span className="font-semibold text-gray-300">{d.label}</span>{" "}
              <span style={{ color: BAND_COLOR[d.fromBand as Band] }}>{formatPct(d.from)}</span>
              <span className="text-gray-500"> → </span>
              <b style={{ color: BAND_COLOR[d.toBand as Band] }}>{formatPct(d.to)}</b>
              <span className="ml-1 text-[10px] text-gray-400">{pts === 0 ? "±0" : `${pts > 0 ? "+" : "−"}${Math.abs(pts)}`} pts</span>
            </div>
          );
        })}
      </div>
      <div className="mt-1 h-4 text-[11px] text-gray-400">{view.status === "updating" ? "Updating…" : ""}</div>
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
  const pos = (v: number) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo || 1)) * 100));
  const solid = "#64748b";
  // Solid inside the cohort's 5th-95th percentile; dashed outside it.
  const track = `linear-gradient(to right, transparent ${pos(p5)}%, ${solid} ${pos(p5)}%, ${solid} ${pos(p95)}%, transparent ${pos(p95)}%),
    repeating-linear-gradient(to right, ${solid} 0 3px, transparent 3px 6px)`;
  const changed = current !== undefined && current !== baseline;
  const note = baseline === undefined ? null
    : imputed ? `not entered: cohort typical${changed ? ` ${round(baseline)}` : ""}`
    : changed ? `recorded ${round(baseline)}` : null;
  const id = `whatif-${lever.key}`;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <label htmlFor={id} className="text-gray-300">{lever.label}</label>
        <span className={changed ? "font-semibold text-sky-300" : "text-gray-300"}>
          {round(value)} {lever.unit}
          {note && <span className="ml-1 font-normal text-gray-400">({note})</span>}
        </span>
      </div>
      <div className="relative h-5">
        <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full" style={{ background: track }} />
        {baseline !== undefined && (
          <div
            title={imputed ? `Not entered: cohort typical ${round(baseline)} ${lever.unit}` : `Recorded: ${round(baseline)} ${lever.unit}`}
            className={`absolute top-0.5 h-4 w-0.5 -translate-x-1/2 ${imputed ? "bg-white/40" : "bg-white/80"}`}
            style={{ left: `${pos(baseline)}%` }}
          />
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
          aria-valuetext={`${round(value)} ${lever.unit}${note ? `, ${note}` : ""}`}
          className="relative h-5 w-full cursor-pointer appearance-none bg-transparent focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400
            [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-slate-900 [&::-moz-range-thumb]:bg-sky-400
            [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-slate-900 [&::-webkit-slider-thumb]:bg-sky-400"
        />
      </div>
      <div className="flex justify-between text-[10px] text-gray-400">
        <span>{round(lo)}</span>
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
      className="flex w-full items-start gap-2 text-left text-xs disabled:cursor-not-allowed"
    >
      <span className={`mt-0.5 flex h-4 w-7 shrink-0 items-center rounded-full p-0.5 transition ${on ? "bg-sky-500" : "bg-slate-600"} ${disabled ? "opacity-50" : ""}`}>
        <span className={`h-3 w-3 rounded-full bg-white transition ${on ? "translate-x-3" : ""}`} />
      </span>
      <span>
        <span className={`${on ? "font-semibold text-sky-300" : "text-gray-300"} ${disabled ? "opacity-50" : ""}`}>{label}</span>
        {/* Readable even when disabled: it says why. */}
        {hint && <span className="block text-[11px] text-gray-400">{hint}</span>}
      </span>
    </button>
  );
}
