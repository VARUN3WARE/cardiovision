// Typed client for the CardioVision API (backend/app/main.py).

import { MODEL_TABS, type Vessel } from "../anatomy/registry";

export type Target = "cad" | Vessel;
export const TARGETS: readonly Target[] = MODEL_TABS;

export interface VesselProbability {
  lad: number;
  lcx: number;
  rca: number;
}

export interface TargetResult {
  probability: number;
  prediction: number;
  threshold: number;
  band: string;
  band_observed_rate: number | null;
}

export interface RangeWarning {
  field: string;
  column: string;
  value: number;
  train_min: number;
  train_max: number;
}

export interface PredictionResponse {
  cad_probability: number;
  vessel_probabilities: VesselProbability;
  cad_prediction: number;
  vessel_predictions: VesselProbability;
  results: Record<Target, TargetResult>;
  imputed: string[];
  warnings: RangeWarning[];
  model_version: string;
  safety_disclaimer: string;
}

export interface ShapExplanation {
  base_value: number;
  feature_names: string[];
  shap_values: number[];
  feature_values: number[];
  derived: boolean[];
}

export interface ExplainResponse extends PredictionResponse {
  explanation: ShapExplanation; // overall CAD model
  explanations: Record<Target, ShapExplanation>;
}

interface Stat {
  mean: number;
  std: number;
}

export interface TargetMetrics {
  model: string;
  n: number;
  positive: number;
  prevalence: number;
  validation: string;
  cv: Record<"roc_auc" | "pr_auc" | "brier" | "log_loss" | "accuracy" | "precision" | "recall" | "specificity" | "f1", Stat>;
  operating_points: {
    f1_threshold: Record<"accuracy" | "precision" | "recall" | "specificity" | "f1", number>;
    rule_out_threshold: Record<"accuracy" | "precision" | "recall" | "specificity" | "f1", number> & { npv: number | null };
  };
  thresholds: { f1: number; rule_out: number; balanced: number };
  bands: { label: string; lo: number; hi: number; share_of_patients: number; observed_rate: number | null; mean_predicted: number | null }[];
}

export interface MetricsResponse {
  metrics: Record<Target, TargetMetrics>;
  model_version: string;
  safety_disclaimer: string;
}

export interface FeatureInfo {
  type: "numeric" | "binary" | "category";
  default: number | string;
  min?: number;
  max?: number;
  median?: number;
  prevalence?: number | null;
}

export interface FeaturesResponse {
  features: string[];
  schema_to_column: Record<string, string>;
  quick_fields: string[];
  feature_info: Record<string, FeatureInfo>;
  influence: Record<Target, Record<string, number>>;
  derived_sources: Record<string, string[]>;
  safety_disclaimer: string;
}

/** Error with a user-facing message and optional per-field messages (from 422 responses). */
export class ApiError extends Error {
  constructor(message: string, public status: number, public fieldErrors: Record<string, string> = {}) {
    super(message);
  }
}

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";
const OFFLINE = "Cannot reach the prediction API. Start it with: cd backend && uvicorn app.main:app --port 8000";

async function request<T>(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  const timeout = AbortSignal.timeout(15000);
  const combined = signal ? anySignal([signal, timeout]) : timeout;
  let res: Response;
  try {
    res = await fetch(BASE + path, { ...init, signal: combined });
  } catch (e) {
    if (signal?.aborted) throw e; // superseded by a newer request; caller ignores it
    if (timeout.aborted) throw new ApiError("The prediction API did not respond within 15 s.", 0);
    throw new ApiError(OFFLINE, 0);
  }
  if (res.ok) return res.json() as Promise<T>;
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // Empty/HTML body: typically the dev proxy reporting that the backend is down.
  }
  throw toApiError(res.status, body);
}

function anySignal(signals: AbortSignal[]): AbortSignal {
  const ctrl = new AbortController();
  for (const s of signals) {
    if (s.aborted) return s;
    s.addEventListener("abort", () => ctrl.abort(s.reason), { once: true });
  }
  return ctrl.signal;
}

interface PydanticError {
  loc: (string | number)[];
  msg: string;
}

function toApiError(status: number, body: unknown): ApiError {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (status === 422 && Array.isArray(detail)) {
    const fieldErrors: Record<string, string> = {};
    const general: string[] = [];
    for (const err of detail as PydanticError[]) {
      const field = err.loc.length > 1 ? String(err.loc[err.loc.length - 1]) : null;
      const msg = err.msg.replace(/^Value error, /, "");
      if (field && field !== "body") fieldErrors[field] = msg;
      else general.push(msg);
    }
    const n = Object.keys(fieldErrors).length;
    const message = general.length ? general.join(" ") : `${n} field${n === 1 ? "" : "s"} need${n === 1 ? "s" : ""} attention.`;
    return new ApiError(message, status, fieldErrors);
  }
  if (status === 503) {
    const msg = typeof detail === "string" ? detail : (detail as { message?: string } | undefined)?.message;
    return new ApiError(msg ?? "Models are not trained yet. Run: cd backend && python train.py", status);
  }
  // The dev proxy answers an empty 5xx when the backend is down: treat it as offline (status 0).
  if (status >= 500 && !body) return new ApiError(OFFLINE, 0);
  return new ApiError(typeof detail === "string" ? detail : `Request failed (${status}).`, status);
}

export function explain(features: Record<string, unknown>, signal?: AbortSignal): Promise<ExplainResponse> {
  return request<ExplainResponse>("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  }, signal);
}

export const getMetrics = () => request<MetricsResponse>("/metrics");
export const getFeatures = () => request<FeaturesResponse>("/features");
export const getHealth = () => request<{ status: string; models_loaded: boolean; model_version: string }>("/health");
