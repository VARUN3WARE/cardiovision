import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { ApiError, explain, predict, type FeaturesResponse } from "../api/client";
import {
  normaliseScenario, scenarioReducer, scenarioRequest, scenarioView, singleFlight, throttle,
  type Payload, type Scenario, type ScenarioBase, type ScenarioError, type ScenarioReply,
} from "./scenario";

/** Fast estimates at most this often while a slider is dragged. */
export const DRAG_THROTTLE_MS = 120;
/** Settle time before the full explanation is fetched (keyboard steps, toggles, a new patient run). */
const EXPLAIN_DELAY_MS = 150;

interface Job {
  kind: "predict" | "explain";
  key: string;
  baseKey: string;
  body: Payload;
  scenario: Scenario;
}

/**
 * Runs the models on the patient's submitted values (`base`: the request that produced the patient's result, never
 * the live form) with the what-if scenario applied; the form is never edited. While dragging: POST /predict,
 * throttled; otherwise one POST /explain for the settled scenario so the explanations match. One request in flight
 * at a time (a newer one waits, see singleFlight); replies are keyed by request body and patient run, and
 * scenarioView decides what every view shows.
 */
export function useScenario(base: ScenarioBase | null, info: FeaturesResponse | null, initial: Scenario) {
  const [raw, dispatch] = useReducer(scenarioReducer, initial);
  const scenario = useMemo(
    () => normaliseScenario(raw, base?.body ?? null, base?.result ?? null, info?.feature_info ?? null),
    [raw, base, info],
  );
  // What was dropped or clamped stays so (it must not come back with a later patient run).
  useEffect(() => {
    if (scenario !== raw) dispatch({ type: "load", scenario });
  }, [scenario, raw]);

  const req = useMemo(() => scenarioRequest(base, scenario), [base, scenario]);
  const [dragging, setDragging] = useState(false);
  const [reply, setReply] = useState<ScenarioReply | null>(null);
  const [error, setError] = useState<ScenarioError | null>(null);

  const baseRef = useRef(base);
  baseRef.current = base;
  const ctrl = useRef<AbortController | null>(null);

  const queue = useMemo(() => singleFlight<Job>((job, done) => {
    const c = new AbortController();
    ctrl.current = c;
    setError((e) => (e?.key === job.key ? null : e));
    const meta = { key: job.key, baseKey: job.baseKey, scenario: job.scenario };
    const ok = (next: (prev: ScenarioReply | null) => ScenarioReply | null) => {
      if (c.signal.aborted) return;
      setReply(next);
      done();
    };
    const fail = (e: unknown) => {
      if (c.signal.aborted) return;
      setError({ key: job.key, kind: job.kind, message: e instanceof ApiError ? e.message : "Could not run the scenario." });
      done();
    };
    if (job.kind === "explain") {
      explain(job.body, c.signal).then((data) => ok(() => ({ ...meta, data, explained: true })), fail);
    } else {
      predict(job.body, c.signal).then((p) => ok((prev) => {
        // Fresh probabilities over the last explanations on this patient run.
        const b = baseRef.current;
        const under = prev?.baseKey === job.baseKey ? prev.data : b?.key === job.baseKey ? b.result : null;
        return under ? { ...meta, data: { ...under, ...p }, explained: false } : prev;
      }), fail);
    }
  }), []);

  const cancel = useCallback(() => {
    ctrl.current?.abort();
    ctrl.current = null;
    queue.clear();
  }, [queue]);

  const throttled = useMemo(() => throttle((job: Job) => queue.submit(job), DRAG_THROTTLE_MS), [queue]);
  useEffect(() => () => {
    throttled.cancel();
    cancel();
  }, [throttled, cancel]);

  // A new patient run, or the scenario switched off: replies and errors on the old run no longer apply.
  const baseKey = base?.key ?? null;
  const on = req !== null;
  useEffect(() => {
    throttled.cancel();
    cancel();
    setReply(null);
    setError(null);
  }, [baseKey, on, throttled, cancel]);

  // Dragging: fast estimates.
  useEffect(() => {
    if (dragging && req) throttled.call({ kind: "predict", ...req, scenario });
  }, [dragging, req?.key]);

  // Settled: one full explanation (not retried by itself after a failure; see retry).
  const settled = !!req && reply?.key === req.key && reply.baseKey === req.baseKey && reply.explained;
  const failed = !!req && error?.key === req.key && error.kind === "explain";
  useEffect(() => {
    if (dragging || !req || settled || failed) return;
    throttled.cancel();
    const job: Job = { kind: "explain", ...req, scenario };
    const t = setTimeout(() => queue.submit(job), EXPLAIN_DELAY_MS);
    return () => clearTimeout(t);
  }, [dragging, req?.key, req?.baseKey, settled, failed]);

  // A drag ends on pointer release anywhere on the page (or when the window loses focus).
  const beginDrag = useCallback(() => {
    setDragging(true);
    const end = () => {
      setDragging(false);
      window.removeEventListener("pointerup", end);
      window.removeEventListener("pointercancel", end);
      window.removeEventListener("blur", end);
    };
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", end);
    window.addEventListener("blur", end);
  }, []);

  const retry = useCallback(() => setError(null), []);
  const view = scenarioView(req, reply, error, base?.result ?? null);

  return {
    /** The scenario as it applies to this patient (dropped parts removed, levers clamped). */
    scenario,
    dispatch,
    /** A scenario changes the patient's submitted values. */
    on,
    view,
    error: view.status === "error" ? error?.message ?? null : null,
    dragging,
    beginDrag,
    retry,
  };
}
