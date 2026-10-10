import { useEffect, useMemo, useRef } from "react";
import type { FeaturesResponse } from "../api/client";
import {
  CHEST_PAIN_KEYS, DEMO_PATIENT, BLANK, QUICK_KEYS, SECTIONS, bmiOf, fromFeatures,
  type FieldSpec, type FormValues,
} from "../fieldSpec";
import PRESETS from "../presets.json";

export interface Preset {
  id: string;
  label: string;
  description: string;
  patient_index: number;
  ground_truth: { cad: boolean; lad: boolean; lcx: boolean; rca: boolean };
  oof_probabilities: { cad: number; lad: number; lcx: number; rca: number };
  features: Record<string, string | number>;
}

// Real patients from the dataset, chosen by held-out predictions (backend/make_presets.py).
export const presets = PRESETS as Preset[];

export type Source = { kind: "demo" } | { kind: "new" } | { kind: "preset"; preset: Preset } | { kind: "edited" };

interface Props {
  values: FormValues;
  quick: boolean;
  source: Source;
  errors: Record<string, string>;
  loading: boolean;
  live: boolean;
  info: FeaturesResponse | null;
  onChange: (values: FormValues, source: Source) => void;
  onQuickChange: (quick: boolean) => void;
  onLiveChange: (live: boolean) => void;
  onSubmit: () => void;
  focusField: string | null;
}

export default function FeatureInput(props: Props) {
  const { values, quick, source, errors, loading, live, info, onChange } = props;
  const formRef = useRef<HTMLFormElement>(null);

  // Scroll to the first invalid field when asked (after a failed submit).
  useEffect(() => {
    if (!props.focusField) return;
    const el = formRef.current?.querySelector<HTMLElement>(`[name="${props.focusField}"]`);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
    el?.focus();
  }, [props.focusField]);

  const set = (patch: FormValues) => onChange({ ...values, ...patch }, { kind: "edited" });
  const keyDrivers = useMemo(() => topDrivers(info), [info]);
  const errorCount = Object.keys(errors).length;
  const bmi = bmiOf(values);

  return (
    <form
      ref={formRef}
      className="flex min-h-full flex-col"
      noValidate
      onSubmit={(e) => {
        e.preventDefault();
        props.onSubmit();
      }}
    >
      <div className="space-y-4 pb-4">
        {/* Patient source + presets */}
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wide text-red-300">Patient</span>
            <SourceChip source={source} />
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            <PresetButton active={source.kind === "new"} onClick={() => onChange({ ...BLANK }, { kind: "new" })}
              title="Clear the form">New patient</PresetButton>
            <PresetButton active={source.kind === "demo"} onClick={() => onChange({ ...DEMO_PATIENT }, { kind: "demo" })}
              title="A synthetic high-risk patient (not from the dataset)">Demo patient</PresetButton>
          </div>
          <div className="mb-1 mt-2 text-[11px] text-gray-400">Real patients from the dataset:</div>
          <div className="grid grid-cols-2 gap-1.5">
            {presets.map((p) => (
              <PresetButton
                key={p.id}
                active={source.kind === "preset" && source.preset.id === p.id}
                onClick={() => onChange(fromFeatures(p.features), { kind: "preset", preset: p })}
                title={`Dataset patient #${p.patient_index}: ${p.description}`}
              >
                {p.label}
              </PresetButton>
            ))}
          </div>
        </div>

        {/* Quick / full mode */}
        <div role="radiogroup" aria-label="Input mode" className="grid grid-cols-2 gap-1 rounded bg-slate-900 p-1 text-xs">
          {([true, false] as const).map((q) => (
            <button
              key={String(q)}
              type="button"
              role="radio"
              aria-checked={quick === q}
              onClick={() => props.onQuickChange(q)}
              className={`rounded px-2 py-1 ${quick === q ? "bg-slate-600 text-white" : "text-gray-400 hover:text-white"}`}
            >
              {q ? `Quick (${QUICK_KEYS.size - 2} key inputs)` : "Full (all 55)"}
            </button>
          ))}
        </div>
        <p className="-mt-2 text-[11px] text-gray-500">
          {quick
            ? "Only the most informative inputs. Anything not asked counts as absent (findings) or takes the 303-patient cohort's median (measurements)."
            : "Leave any field you don't have empty: an unrecorded finding counts as absent, a missing measurement takes the cohort median, and every filled value is listed in the results."}
        </p>

        {errorCount > 0 && (
          <div role="alert" className="rounded border border-red-700 bg-red-950/50 px-2 py-1.5 text-xs text-red-200">
            {errorCount} field{errorCount === 1 ? "" : "s"} need{errorCount === 1 ? "s" : ""} attention:{" "}
            {Object.keys(errors).map((k) => labelOf(k)).join(", ")}
          </div>
        )}

        {SECTIONS.map((section) => {
          const fields = section.fields.filter((f) => !quick || QUICK_KEYS.has(f.key));
          const showChestPain = section.id === "symptoms";
          const showSmoking = section.id === "risk";
          if (!fields.length && !showChestPain && !showSmoking) return null;
          return (
            <fieldset key={section.id}>
              <legend className="mb-1 text-xs font-semibold uppercase tracking-wide text-red-300">{section.title}</legend>
              {showChestPain && (
                <ChestPain values={values} onChange={set} error={errors.TypicalChestPain ?? errors.Atypical ?? errors.Nonanginal} />
              )}
              {showSmoking && <Smoking values={values} quick={quick} onChange={set} />}
              <div className="grid grid-cols-2 gap-x-2 gap-y-1.5">
                {fields.map((f) => (
                  <Field key={f.key} spec={f} value={values[f.key] ?? ""} error={errors[f.key]}
                    keyDriver={keyDrivers.has(f.key)} info={info} onChange={(v) => set({ [f.key]: v })} />
                ))}
                {section.id === "patient" && (
                  <div className="text-xs">
                    <span className="text-gray-400">BMI (calculated)</span>
                    <div className="mt-0.5 rounded bg-slate-900/50 px-2 py-1 text-gray-300">
                      {bmi ? `${bmi.toFixed(1)} kg/m²${bmi >= 25 ? " · overweight" : ""}` : "—"}
                    </div>
                  </div>
                )}
              </div>
            </fieldset>
          );
        })}
        <p className="text-[11px] text-gray-500">
          <span className="text-sky-400">●</span> marks the inputs with the most influence on the current models.
        </p>
      </div>

      {/* Sticky action bar */}
      <div className="sticky bottom-0 -mx-4 mt-auto border-t border-slate-700 bg-slate-800/95 px-4 py-2 backdrop-blur">
        <label className="mb-1.5 flex items-center gap-2 text-xs text-gray-300">
          <input type="checkbox" checked={live} onChange={(e) => props.onLiveChange(e.target.checked)} />
          Update results automatically as I type
        </label>
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded bg-red-700 px-4 py-2 font-semibold text-white hover:bg-red-800 disabled:opacity-50"
        >
          {loading ? "Estimating…" : "Estimate risk"}
        </button>
      </div>
    </form>
  );
}

function labelOf(key: string) {
  for (const s of SECTIONS) for (const f of s.fields) if (f.key === key) return f.label;
  return CHEST_PAIN_KEYS.includes(key as never) ? "Chest pain" : key;
}

/** Schema keys of the ~10 inputs with the largest summed influence across the four models. */
function topDrivers(info: FeaturesResponse | null): Set<string> {
  if (!info) return new Set();
  const colToKey = Object.fromEntries(Object.entries(info.schema_to_column).map(([k, c]) => [c, k]));
  const total: Record<string, number> = {};
  for (const target of Object.values(info.influence)) {
    for (const [name, v] of Object.entries(target)) {
      for (const col of info.derived_sources[name] ?? [name]) {
        const n = (info.derived_sources[name] ?? [name]).length;
        const key = colToKey[col] ?? col;
        total[key] = (total[key] ?? 0) + v / n;
      }
    }
  }
  return new Set(Object.entries(total).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([k]) => k));
}

function SourceChip({ source }: { source: Source }) {
  const text = source.kind === "demo" ? "Demo patient (synthetic)"
    : source.kind === "new" ? "New patient"
      : source.kind === "preset" ? `Dataset patient #${source.preset.patient_index}`
        : "Edited";
  return <span className="rounded-full bg-slate-700 px-2 py-0.5 text-[10px] text-gray-300">{text}</span>;
}

function PresetButton({ active, onClick, title, children }: {
  active: boolean; onClick: () => void; title: string; children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-pressed={active}
      className={`rounded border px-2 py-1 text-left text-xs transition ${active
        ? "border-red-400 bg-red-900/40 text-white"
        : "border-slate-600 bg-slate-900 text-gray-300 hover:border-slate-400"}`}
    >
      {children}
    </button>
  );
}

function Field({ spec, value, error, keyDriver, info, onChange }: {
  spec: FieldSpec; value: string; error?: string; keyDriver: boolean; info: FeaturesResponse | null;
  onChange: (v: string) => void;
}) {
  const id = `f-${spec.key}`;
  const column = info?.schema_to_column[spec.key] ?? spec.key;
  const fi = info?.feature_info[column];
  const n = Number(value);
  const outOfRange = spec.kind === "number" && value !== "" && fi?.min !== undefined && fi?.max !== undefined
    && isFinite(n) && (n < fi.min || n > fi.max);
  const base = `mt-0.5 w-full rounded bg-slate-900 px-2 py-1 text-white outline-none focus:ring-1 focus:ring-sky-500 ${
    error ? "ring-1 ring-red-500" : outOfRange ? "ring-1 ring-amber-500" : ""}`;

  return (
    <div className="text-xs">
      <label htmlFor={id} className="flex items-center gap-1 text-gray-300" title={spec.hint}>
        {spec.label}
        {spec.required && <span className="text-red-400">*</span>}
        {keyDriver && <span className="text-sky-400" aria-label="key input">●</span>}
      </label>
      {spec.kind === "number" ? (
        <div className="relative">
          <input
            id={id}
            name={spec.key}
            type="number"
            inputMode="decimal"
            min={spec.min}
            max={spec.max}
            step={spec.step}
            value={value}
            placeholder={fi?.median !== undefined ? `e.g. ${fi.median}` : undefined}
            aria-invalid={!!error}
            aria-describedby={error ? `${id}-err` : undefined}
            onWheel={(e) => e.currentTarget.blur()}
            onChange={(e) => onChange(e.target.value)}
            className={`${base} ${spec.unit ? "pr-12" : ""} placeholder:text-gray-600`}
          />
          {spec.unit && (
            <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-gray-500">
              {spec.unit}
            </span>
          )}
        </div>
      ) : (
        <select
          id={id}
          name={spec.key}
          value={value}
          aria-invalid={!!error}
          onChange={(e) => onChange(e.target.value)}
          className={`${base} ${value === "" ? "text-gray-500" : ""}`}
        >
          <option value="">{spec.required ? "Select…" : "Not recorded"}</option>
          {spec.kind === "flag" && (<><option value="0">No</option><option value="1">Yes</option></>)}
          {spec.kind === "yn" && (<><option value="N">No</option><option value="Y">Yes</option></>)}
          {spec.kind === "select" && spec.options!.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
      )}
      {error ? (
        <div id={`${id}-err`} className="mt-0.5 text-[10px] text-red-400">{error}</div>
      ) : outOfRange ? (
        <div className="mt-0.5 text-[10px] text-amber-400">Outside training range {fi!.min}–{fi!.max}</div>
      ) : null}
    </div>
  );
}

const CHEST_PAIN_OPTIONS: [string, string][] = [
  ["", "Not recorded"], ["none", "None"], ["typical", "Typical angina"], ["atypical", "Atypical"], ["nonanginal", "Non-anginal"],
];

function ChestPain({ values, onChange, error }: { values: FormValues; onChange: (p: FormValues) => void; error?: string }) {
  const current = values.TypicalChestPain === "1" ? "typical"
    : values.Atypical === "Y" ? "atypical"
      : values.Nonanginal === "Y" ? "nonanginal"
        : values.TypicalChestPain === "0" ? "none" : "";
  const choose = (v: string) => onChange(v === "" ? { TypicalChestPain: "", Atypical: "", Nonanginal: "" } : {
    TypicalChestPain: v === "typical" ? "1" : "0",
    Atypical: v === "atypical" ? "Y" : "N",
    Nonanginal: v === "nonanginal" ? "Y" : "N",
  });
  return (
    <div className="mb-1.5 text-xs">
      <label htmlFor="f-chest-pain" className="flex items-center gap-1 text-gray-300">
        Chest pain <span className="text-sky-400" aria-label="key input">●</span>
      </label>
      <select id="f-chest-pain" name="TypicalChestPain" value={current} onChange={(e) => choose(e.target.value)}
        className={`mt-0.5 w-full rounded bg-slate-900 px-2 py-1 text-white ${error ? "ring-1 ring-red-500" : ""}`}>
        {CHEST_PAIN_OPTIONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
      {error && <div className="mt-0.5 text-[10px] text-red-400">{error}</div>}
    </div>
  );
}

function Smoking({ values, quick, onChange }: { values: FormValues; quick: boolean; onChange: (p: FormValues) => void }) {
  const current = values.CurrentSmoker === "1" ? "current" : values.EX_Smoker === "1" ? "former"
    : values.CurrentSmoker === "0" ? "never" : "";
  const choose = (v: string) => onChange(v === "" ? { CurrentSmoker: "", EX_Smoker: "" } : {
    CurrentSmoker: v === "current" ? "1" : "0",
    EX_Smoker: v === "former" ? "1" : "0",
  });
  return (
    <div className="mb-1.5 text-xs">
      <label htmlFor="f-smoking" className="text-gray-300">Smoking</label>
      <select id="f-smoking" name="CurrentSmoker" value={current} onChange={(e) => choose(e.target.value)}
        className="mt-0.5 w-full rounded bg-slate-900 px-2 py-1 text-white">
        <option value="">Not recorded</option>
        <option value="never">Never</option>
        {!quick && <option value="former">Former</option>}
        <option value="current">Current</option>
      </select>
    </div>
  );
}
