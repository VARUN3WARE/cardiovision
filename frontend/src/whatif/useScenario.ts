import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { ApiError, explain, predict, type ExplainResponse } from "../api/client";
import { applyScenario, isActive, scenarioReducer, throttle, type Payload, type Scenario } from "./scenario";

/** Fast estimates at most this often while a slider is dragged. */
export const DRAG_THROTTLE_MS = 120;
/** Settle time before the full explanation is fetched (keyboard steps, toggles, form edits). */
const EXPLAIN_DELAY_MS = 150;

interface ScenarioResult {
  key: string; // request body the result belongs to
  data: ExplainResponse; // during a drag: fresh probabilities over the last explanations
  explained: boolean; // explanations match the probabilities
}

/**
 * Runs the model on the form's payload with the what-if scenario applied. The form is never edited.
 * While dragging: POST /predict, throttled, one request in flight (older ones aborted, stale replies
 * ignored). Otherwise: one POST /explain for the settled scenario so the explanations match.
 */
export function useScenario(basePayload: Payload, baseResult: ExplainResponse | null, initial: Scenario) {
  const [scenario, dispatch] = useReducer(scenarioReducer, initial);
  const [dragging, setDragging] = useState(false);
  const [res, setRes] = useState<ScenarioResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const active = isActive(scenario);
  const body = useMemo(
    () => (active ? applyScenario(basePayload, scenario, baseResult) : null),
    [active, basePayload, scenario, baseResult],
  );
  const bodyKey = body ? JSON.stringify(body) : null;

  const reqId = useRef(0);
  const ctrl = useRef<AbortController | null>(null);
  const baseRef = useRef(baseResult);
  baseRef.current = baseResult;

  const send = useCallback(async (kind: "predict" | "explain", b: Payload, k: string) => {
    ctrl.current?.abort(); // never more than one scenario request in flight
    const c = new AbortController();
    ctrl.current = c;
    const id = ++reqId.current;
    try {
      if (kind === "explain") {
        const data = await explain(b, c.signal);
        if (id !== reqId.current) return;
        setRes({ key: k, data, explained: true });
      } else {
        const p = await predict(b, c.signal);
        if (id !== reqId.current) return;
        setRes((prev) => {
          const under = prev?.data ?? baseRef.current;
          return under ? { key: k, data: { ...under, ...p }, explained: false } : prev;
        });
      }
      setError(null);
    } catch (e) {
      if (id !== reqId.current || c.signal.aborted) return;
      setError(e instanceof ApiError ? e.message : "Could not run the scenario.");
    }
  }, []);

  const throttled = useMemo(() => throttle((b: Payload, k: string) => send("predict", b, k), DRAG_THROTTLE_MS), [send]);
  useEffect(() => () => throttled.cancel(), [throttled]);

  // Inactive (or reset): drop everything in flight and show the patient's own result.
  useEffect(() => {
    if (active) return;
    throttled.cancel();
    ctrl.current?.abort();
    reqId.current++;
    setRes(null);
    setError(null);
  }, [active, throttled]);

  // Dragging: fast estimates.
  useEffect(() => {
    if (dragging && body && bodyKey) throttled.call(body, bodyKey);
  }, [dragging, bodyKey]);

  // Settled: one full explanation.
  const settled = res !== null && res.key === bodyKey && res.explained;
  useEffect(() => {
    if (dragging || !body || !bodyKey || settled) return;
    throttled.cancel();
    const t = setTimeout(() => send("explain", body, bodyKey), EXPLAIN_DELAY_MS);
    return () => clearTimeout(t);
  }, [dragging, bodyKey, settled]);

  // A drag ends on pointer release anywhere on the page.
  const beginDrag = useCallback(() => {
    setDragging(true);
    const end = () => {
      setDragging(false);
      window.removeEventListener("pointerup", end);
      window.removeEventListener("pointercancel", end);
    };
    window.addEventListener("pointerup", end);
    window.addEventListener("pointercancel", end);
  }, []);

  return {
    scenario,
    dispatch,
    active,
    /** The scenario's result (null when inactive or before its first reply). */
    result: active ? res?.data ?? null : null,
    /** A newer scenario is being calculated, or the explanations lag the probabilities. */
    updating: active && !settled,
    explanationsLag: active && res !== null && !res.explained,
    error: active ? error : null,
    beginDrag,
  };
}
