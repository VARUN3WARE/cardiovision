import { describe, expect, it } from "vitest";
import { DEFAULT_TWIN, beatTiming, ecgSample, mechanics } from "./twin";

describe("function twin", () => {
  it("keeps physiological order within a beat", () => {
    const p = { ...DEFAULT_TWIN, notRecorded: new Set<never>() };
    const tm = beatTiming(p);
    const peak = (f: (t: number) => number) => {
      let best = 0, at = 0;
      for (let t = 0; t < tm.rr; t += 0.002) if (f(t) > best) { best = f(t); at = t; }
      return at;
    };
    const atrial = peak((t) => mechanics(t, tm, p).atrial);
    const rWave = peak((t) => ecgSample(t, tm, p, "II"));
    const ventricular = peak((t) => mechanics(t, tm, p).ventricular);
    expect(atrial).toBeLessThan(rWave);
    expect(rWave).toBeLessThan(ventricular);
  });
  it("shortens QT at faster rates and widens QRS in bundle branch block", () => {
    expect(beatTiming({ hr: 120, bbb: "none" }).qt).toBeLessThan(beatTiming({ hr: 60, bbb: "none" }).qt);
    expect(beatTiming({ hr: 70, bbb: "LBBB" }).qrs).toBeGreaterThanOrEqual(0.12);
  });
  it("scales the squeeze with ejection fraction", () => {
    const tm = beatTiming({ hr: 70, bbb: "none" });
    const low = mechanics(0.4, tm, { ...DEFAULT_TWIN, ef: 35 }).squeeze;
    const high = mechanics(0.4, tm, { ...DEFAULT_TWIN, ef: 60 }).squeeze;
    expect(low).toBeLessThan(high);
  });
});
