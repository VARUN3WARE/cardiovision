import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FeatureInput, { type Source } from "./components/FeatureInput";
import HeartScene, { type Vessel } from "./components/HeartScene";
import ClinicalDashboard from "./components/ClinicalDashboard";
import ExplanationPanel from "./components/ExplanationPanel";
import ErrorBoundary from "./components/ErrorBoundary";
import {
  ApiError, explain, getFeatures, getHealth, getMetrics,
  type ExplainResponse, type FeaturesResponse, type MetricsResponse,
} from "./api/client";
import { DEMO_PATIENT, toPayload, validate, type FormValues } from "./fieldSpec";
import { configureBands, formatPct } from "./risk";

type ApiStatus = "checking" | "online" | "models_missing" | "offline";

export default function App() {
  const [values, setValues] = useState<FormValues>(DEMO_PATIENT);
  const [source, setSource] = useState<Source>({ kind: "demo" });
  const [quick, setQuick] = useState(false);
  const [live, setLive] = useState(true);

  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [resultKey, setResultKey] = useState<string | null>(null);
  const [resultSource, setResultSource] = useState<Source | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [serverFieldErrors, setServerFieldErrors] = useState<Record<string, string>>({});
  const [focusField, setFocusField] = useState<string | null>(null);
  const [showErrors, setShowErrors] = useState(false);

  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [info, setInfo] = useState<FeaturesResponse | null>(null);
  const [status, setStatus] = useState<ApiStatus>("checking");
  const [selected, setSelected] = useState<Vessel | null>(null);

  const payload = useMemo(() => toPayload(values, quick), [values, quick]);
  const key = useMemo(() => JSON.stringify(payload), [payload]);
  const clientErrors = useMemo(() => validate(values, quick), [values, quick]);
  const valid = Object.keys(clientErrors).length === 0;
  const stale = result !== null && resultKey !== key;

  const requestId = useRef(0);
  const controller = useRef<AbortController | null>(null);

  const checkApi = useCallback(async () => {
    try {
      await getHealth();
      setStatus("online");
      const [m, f] = await Promise.all([getMetrics(), getFeatures()]);
      configureBands(m);
      setMetrics(m);
      setInfo(f);
    } catch (e) {
      setStatus(e instanceof ApiError && e.status === 503 ? "models_missing" : "offline");
    }
  }, []);

  useEffect(() => {
    checkApi();
  }, [checkApi]);

  const run = useCallback(async (body: Record<string, unknown>, bodyKey: string, src: Source) => {
    controller.current?.abort();
    const ctrl = new AbortController();
    controller.current = ctrl;
    const id = ++requestId.current;
    setLoading(true);
    try {
      const res = await explain(body, ctrl.signal);
      if (id !== requestId.current) return; // a newer request superseded this one
      setResult(res);
      setResultKey(bodyKey);
      setResultSource(src);
      setError(null);
      setServerFieldErrors({});
      if (status !== "online") setStatus("online");
    } catch (e) {
      if (id !== requestId.current || ctrl.signal.aborted) return;
      const err = e instanceof ApiError ? e : new ApiError("Unexpected error.", 0);
      setError(err);
      setServerFieldErrors(err.fieldErrors);
      if (err.status === 0) setStatus("offline");
      if (err.status === 503) setStatus("models_missing");
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }, [status]);

  // Live mode: re-estimate shortly after the last valid edit.
  useEffect(() => {
    if (!live || !valid || key === resultKey || status === "offline" || status === "checking") return;
    const t = setTimeout(() => run(payload, key, source), 450);
    return () => clearTimeout(t);
  }, [live, valid, key, resultKey, payload, source, run, status]);

  const submit = () => {
    setShowErrors(true);
    if (!valid) {
      setFocusField(Object.keys(clientErrors)[0]);
      return;
    }
    run(payload, key, source);
  };

  const onChange = (v: FormValues, src: Source) => {
    setValues(v);
    // Keep the dataset-patient label until the user actually edits a field.
    setSource(src.kind === "edited" && source.kind !== "preset" && source.kind !== "edited" ? source : src);
    setServerFieldErrors({});
    setFocusField(null);
    if (src.kind !== "edited") setShowErrors(false);
  };

  // Invalid values are flagged as you type; "Required" only after a submit attempt.
  const visibleClientErrors = showErrors ? clientErrors
    : Object.fromEntries(Object.entries(clientErrors).filter(([k]) => (values[k] ?? "").trim() !== ""));
  const errors = { ...visibleClientErrors, ...serverFieldErrors };
  const shownPreset = !stale && resultSource?.kind === "preset" ? resultSource.preset : null;
  const bpm = Number(values.PR) > 0 ? Number(values.PR) : 72;

  return (
    <div className="flex min-h-screen flex-col bg-slate-900 text-white lg:h-screen">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700 px-4 py-3 lg:px-6">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-bold">
            Cardio<span className="text-red-500">Vision</span>
          </h1>
          <span className="hidden text-xs text-gray-400 md:inline">
            Coronary artery disease risk · vessel-level estimates · explained
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <StatusPill status={status} onRetry={checkApi} />
          <a href="/docs" target="_blank" rel="noreferrer" className="text-gray-400 hover:text-white">API docs</a>
        </div>
      </header>

      {/* Safety disclaimer: required, always visible, outside every error boundary */}
      <div role="note" className="border-y border-yellow-700 bg-yellow-900/40 px-4 py-2 text-center text-sm text-yellow-200 lg:px-6">
        ⚠️ <b>Decision support / educational use only.</b> Estimates come from a model trained on 303 patients referred
        for angiography at one centre (71% had CAD). Not validated for screening and not a diagnosis; it does not replace
        angiography, formal diagnostic imaging or professional medical judgement.
      </div>

      <main className="grid flex-1 grid-cols-1 gap-4 p-4 lg:min-h-0 lg:grid-cols-[320px_minmax(0,1fr)_380px]">
        <section aria-label="Clinical findings" className="order-3 overflow-auto rounded bg-slate-800 px-4 pt-4 lg:order-1">
          <h2 className="mb-3 text-sm font-semibold text-gray-200">Clinical findings</h2>
          <FeatureInput
            values={values}
            quick={quick}
            source={source}
            errors={errors}
            loading={loading}
            live={live}
            info={info}
            onChange={onChange}
            onQuickChange={setQuick}
            onLiveChange={setLive}
            onSubmit={submit}
            focusField={focusField}
          />
        </section>

        <section aria-label="3D heart" className="order-2 h-[55vh] min-h-0 rounded bg-slate-800 lg:h-auto">
          <ErrorBoundary fallback={<HeartFallback result={result} />}>
            <HeartScene
              // Keep the last estimate while inputs change (colours glide to the new one; no grey flash).
              probs={result ? result.vessel_probabilities : null}
              pending={stale}
              result={result}
              metrics={metrics}
              selected={selected}
              onSelect={setSelected}
              bpm={bpm}
            />
          </ErrorBoundary>
        </section>

        <section aria-label="Results" className="order-1 space-y-4 overflow-auto lg:order-3">
          {error && (
            <div role="alert" className="rounded border border-red-700 bg-red-950/60 p-3 text-xs text-red-200">
              <b>Could not get an estimate.</b> {error.message}
            </div>
          )}
          {stale && (
            <div role="status" className="flex items-center justify-between gap-2 rounded border border-amber-600 bg-amber-950/50 px-3 py-2 text-xs text-amber-200">
              <span>{loading ? "Updating…" : valid ? "Inputs changed: results below are for the previous values." : "Fix the highlighted fields to update the results."}</span>
              {!loading && valid && (
                <button onClick={submit} className="rounded bg-amber-600 px-2 py-0.5 font-semibold text-slate-900 hover:bg-amber-500">
                  Re-run
                </button>
              )}
            </div>
          )}
          <ErrorBoundary fallback={<div className="rounded bg-slate-800 p-4 text-sm text-red-300">Could not render the results.</div>}>
            <div className={`space-y-4 transition-opacity ${stale ? "opacity-50" : ""}`}>
              <div className="rounded bg-slate-800 p-4">
                <h2 className="mb-3 text-sm font-semibold text-gray-200">Risk estimate</h2>
                <ClinicalDashboard result={result} metrics={metrics} selected={selected} onSelect={setSelected} preset={shownPreset} />
              </div>
              <div className="rounded bg-slate-800 p-4">
                <h2 className="mb-3 text-sm font-semibold text-gray-200">Why this estimate?</h2>
                <ExplanationPanel result={result} metrics={metrics} info={info} selected={selected} onSelect={setSelected} />
              </div>
            </div>
          </ErrorBoundary>
        </section>
      </main>
    </div>
  );
}

function StatusPill({ status, onRetry }: { status: ApiStatus; onRetry: () => void }) {
  const map: Record<ApiStatus, [string, string]> = {
    checking: ["bg-slate-500", "Connecting…"],
    online: ["bg-green-500", "Model online"],
    models_missing: ["bg-amber-500", "Models not trained"],
    offline: ["bg-red-500", "API offline"],
  };
  const [dot, text] = map[status];
  return (
    <span className="flex items-center gap-1.5 text-gray-300" role="status">
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {text}
      {(status === "offline" || status === "models_missing") && (
        <button onClick={onRetry} className="underline hover:text-white">retry</button>
      )}
    </span>
  );
}

function HeartFallback({ result }: { result: ExplainResponse | null }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center text-sm text-gray-400">
      <div>3D view unavailable on this device. Estimates and explanations still work.</div>
      {result && (
        <div className="flex gap-2">
          {(["lad", "lcx", "rca"] as const).map((v) => (
            <span key={v} className="rounded bg-slate-900 px-2 py-1 text-xs text-gray-200">
              {v.toUpperCase()} {formatPct(result.results[v].probability)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
