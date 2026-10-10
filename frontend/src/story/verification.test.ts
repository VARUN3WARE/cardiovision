import { describe, expect, it } from "vitest";
import { MODEL_TABS } from "../anatomy/registry";
import { VERIFICATION } from "./verification";

interface VerificationFile {
  targets: Record<string, { auc_95ci: [number, number] }>;
  data: { rows: number; prevalence: { cad: number } };
}

// Node's fs, loaded by a computed name so the browser build and its type check never see it.
async function readVerification(): Promise<VerificationFile> {
  const fs = (await import(/* @vite-ignore */ `node:${"fs"}`)) as { readFileSync: (path: URL, encoding: "utf8") => string };
  return JSON.parse(fs.readFileSync(new URL("../../../backend/models/verification.json", import.meta.url), "utf8"));
}

describe("story closing card facts", () => {
  it("match backend/models/verification.json (bootstrap 95% CIs of the cross-validated AUC, cohort size)", async () => {
    const file = await readVerification();
    for (const t of MODEL_TABS) {
      const [lo, hi] = file.targets[t].auc_95ci;
      expect(VERIFICATION.aucCi[t][0]).toBeCloseTo(lo, 3);
      expect(VERIFICATION.aucCi[t][1]).toBeCloseTo(hi, 3);
    }
    expect(VERIFICATION.rows).toBe(file.data.rows);
    expect(VERIFICATION.prevalenceCad).toBeCloseTo(file.data.prevalence.cad, 3);
  });

  it("round to the intervals the card shows (CAD 0.91-0.97, LAD 0.82-0.91, LCX 0.71-0.81, RCA 0.70-0.81)", () => {
    const shown = Object.fromEntries(MODEL_TABS.map((t) => [t, VERIFICATION.aucCi[t].map((x) => x.toFixed(2)).join("-")]));
    expect(shown).toEqual({ cad: "0.91-0.97", lad: "0.82-0.91", lcx: "0.71-0.81", rca: "0.70-0.81" });
  });
});
