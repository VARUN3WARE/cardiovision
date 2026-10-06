export interface VesselProbability {
  lad: number;
  lcx: number;
  rca: number;
}

export interface PredictionResponse {
  cad_probability: number;
  vessel_probabilities: VesselProbability;
  cad_prediction: number;
  vessel_predictions: VesselProbability;
}

export interface ShapExplanation {
  base_value: number;
  feature_names: string[];
  shap_values: number[];
  feature_values: number[];
}

export interface ExplainResponse extends PredictionResponse {
  explanation: ShapExplanation;
}

export async function predict(features: Record<string, number>): Promise<PredictionResponse> {
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });
  if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
  return res.json();
}

export async function explain(features: Record<string, number>): Promise<ExplainResponse> {
  const res = await fetch("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  });
  if (!res.ok) throw new Error(`Explain failed: ${res.status}`);
  return res.json();
}
