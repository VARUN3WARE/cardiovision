/**
 * Facts the story's closing card states about the models' validation, copied from
 * backend/models/verification.json (the independent verification, docs/model-verification.md): the bootstrap
 * 95% confidence interval of each model's cross-validated ROC AUC, and the cohort's size and CAD prevalence.
 * verification.test.ts checks this copy against the file.
 */
import type { Target } from "../api/client";

export const VERIFICATION: {
  rows: number;
  prevalenceCad: number;
  aucCi: Record<Target, readonly [number, number]>;
} = {
  rows: 303,
  prevalenceCad: 0.713,
  aucCi: {
    cad: [0.914, 0.967],
    lad: [0.822, 0.907],
    lcx: [0.706, 0.812],
    rca: [0.702, 0.811],
  },
};
