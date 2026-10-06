> **Status: parked.** Generated 2026-10-07 by a judge-panel design workflow (3 designs, 2 judges, synthesis). Not implemented yet — the team paused 3D work to focus on the ML models and usability. Resume once the Sketchfab "Realistic Human Heart" (CC BY 4.0, neshallads) glTF is in `frontend/public/models/`.

# CardioVision: plan for adding the Sketchfab heart model with risk overlays

No files were changed. Every path below is relative to `/home/varun/Desktop/someProj/multimodal AI Hackathon/cardiovision/frontend/` unless it starts with `/`. I checked the code against what is installed: three 0.169, drei 9.122, R3F 8.18, three-mesh-bvh 0.7.8, postprocessing 6.39.5 and Vite 5.4.21.

## 0. Decisions

**The base is "pragmatic".** It had the best combined judge score (57, against 51 for fidelity and 50.5 for autofit). The procedural heart path stays as it is today: `Heart`, `Myocardium`, `Veins`, `Atria` and `GreatVessels` are not changed. `Artery` gets three optional props whose defaults behave exactly as today. The specimen model is baked once into the procedural heart's local frame. After that, `territoryWeights`, the (u,v) artery paths, the camera views, the heartbeat group and the label logic are all reused.

What is taken from each design:

| Idea | Source | Why |
|---|---|---|
| Bake node transforms, then put the mesh in the heart-local frame | pragmatic | Everything downstream is reused unchanged |
| 4-click landmark registration (apex, LAD-mid, AV-left, AV-right; optional crux, PDA, ostia), committed as JSON | fidelity | Deterministic for any orientation or scale |
| Circular angle warp for the LAD/RCA/PDA grooves, plus an AV-groove height warp per azimuth (new) | fidelity, extended | LAD and PDA/RCA fit independently; the tilted AV groove is followed |
| Zero-click auto-guess that assumes the model's own pose (glTF +Y up, +Z front), not HEART_ROTATION | fidelity, fixed | The thumbnail shows an upright heart |
| Inside-out ray from the long axis taking the first **exit** hit; closest-point snap as fallback | fidelity + pragmatic | Tubes sit on the epicardium under overhanging auricles and the PA root |
| Winding sign taken from the model's own vertex normals | new | Does not depend on the registration being right |
| Radial outlier rejection and Laplacian smoothing of the snapped samples | autofit + pragmatic | Stops kinks |
| Per-vertex `aTerr` (lad, lcx, rca, mask) attribute; all mixing on the GPU, driven by uniforms; damped cross-fades; no recompile | autofit | Mode or selection change is only a uniform write |
| Recolouring that keeps luminance, so the texture detail survives the tint | fidelity | |
| Ventricle mask = height under the warped AV groove × "outer-shell" ray test | fidelity / autofit, simplified | Auricles, atria and the aortic root are not tinted |
| HEAD probe that rejects `text/html`, with a ranged GET if HEAD returns 405 | autofit | Vite's SPA fallback returns index.html with status 200 |
| Error boundary *inside* the Canvas, `useGLTF.clear`, a timeout, and `compileAsync` before the loading overlay hides | autofit | |
| Float32 de-quantise, keep `uv1`, physical material built field by field, `metalness = 0`, attach `disposeBoundsTree`, no Draco | judges / fidelity / pragmatic | Fixes for the flaws the judges listed |
| Not used: ridge snapping, a vertex-shader heartbeat, a Neutral tone-mapping switch, a rewrite of the tube or artery material, autofit's automatic azimuth estimator | | The model's coronaries are engraved, not raised; scope; changes to the procedural look |

**Commit the processed asset:** a 2048 px WebP `public/models/heart.glb` of about 3-6 MB, plus `LICENSE-heart.txt` and `CREDITS.md`. CC BY 4.0 allows redistribution with credit, a licence link and a note of changes. Judges who clone the repo must see the realistic heart. Gitignore the raw export.

## 1. Data flow

```
probe (HEAD, reject text/html) -> useGLTF(url,false,true) [Suspense, ModelBoundary inside Canvas]
  -> fitHeartModel(gltf.scene):
       bakeModel (node matrices baked, Float32, merge)          raw model space
       landmarks = config | calibration picks | autoGuess
       registerFromLandmarks -> similarity matrix + Warp(v->theta, uScale(v))
       geometry.applyMatrix4(matrix) ; own MeshBVH                heart frame (apex y=-1.15, AV y=0.475, +X lateral, +Z anterior)
       projector (first exit from the axis, closest-point fallback)
       arteries = ARTERIES.map(placeVesselOnMesh)                 curves + label anchors
       writeTerritoryAttributes (aTerr, param)                    per-vertex, once
       tissue material (field copy + onBeforeCompile tint)        one program
       views / placement / diagnostics
  -> <GlbHeart> inside the existing HEART_ROTATION group; the ventricles group beats; arteries go in a <Bvh key={fit.id}>
fail anywhere -> boundary or timeout -> today's procedural <Heart> unchanged
```

## 2. Implementation steps, in order

### Step 0. Housekeeping (15 min)
- `npm i three-mesh-bvh@^0.7.8`. It is already in package-lock as drei's dependency, so this only adds an explicit entry. Network installs need the sandbox permission noted in HANDOFF.
- Create `src/vite-env.d.ts` containing `/// <reference types="vite/client" />`. There is no such file today, so `import.meta.env` would fail `tsc -b`.
- Add to the root `.gitignore`:
  ```
  frontend/public/models/scene.gltf
  frontend/public/models/scene.bin
  frontend/public/models/textures/
  ```

### Step 1. Asset prep and licence files (20 min)
```bash
cd frontend/public/models          # unzip the Sketchfab glTF export here
npx -y @gltf-transform/cli@4 inspect scene.gltf     # note: mesh count, vertex count, texture sizes, extensions (no Draco/meshopt expected)
npx -y @gltf-transform/cli@4 copy   scene.gltf heart.glb
npx -y @gltf-transform/cli@4 resize heart.glb heart.glb --width 2048 --height 2048
npx -y @gltf-transform/cli@4 webp   heart.glb heart.glb --quality 90   # check flags with --help; exclude the normal map via --slots if it shows blocks
ls -la heart.glb                   # expect 3-6 MB; never Draco
```
- Copy `license.txt` from the zip, unchanged, to `public/models/LICENSE-heart.txt`.
- Write `public/models/CREDITS.md` (text in section 6) and rewrite `README.txt` to describe these commands.
- Add the script `"model:prep"` to package.json, chaining the three commands above.
- If the download was Sketchfab's auto-converted `.glb`, drop it in as `scene.glb`. The probe accepts it.

### Step 2. `src/components/heartAnatomy.ts`: additive changes only (30 min)
- Change `const APEX_Y`, `const BASE_Y` and `const AV` to `export const`. Do not touch `surfacePoint`, `vesselCurve` or the other functions.

```ts
export const LONG_AXIS = BASE_Y - APEX_Y;            // 1.65
export const LAD_GROOVE_V = 1.2;                     // anterior IV groove (surfacePoint groove term)
export const PDA_GROOVE_V = 4.55;                    // posterior IV groove
/** Camera table shared by the procedural and mesh views: [vessel, u, v, elevation]. */
export const VESSEL_VIEW_TABLE = [["lad", 0.5, 1.3, 0.2], ["lcx", 0.65, -0.1, 0.15], ["rca", 0.55, 4.1, -0.45]] as const;

/** Bare ventricular radius profile (no RV/groove bumps): used to aim projection rays. */
export function baseRadius(u: number) {
  const uc = Math.min(Math.max(u, 0), 1);
  let r = R_MAX * Math.pow(Math.sin(Math.min(1, uc * 1.2) * Math.PI / 2), 0.68);
  if (u > 1) { const t = Math.min((u - 1) / CAP, 1); r *= Math.sqrt(Math.max(0, 1 - t * t)); }
  return r;
}

/** (u, v, t) samples of a vessel path with the same tortuosity maths as vesselCurve (copied, not refactored). */
export function vesselSamples(seg: VesselPath, wiggleScale = 1): { u: number; v: number; t: number }[] {
  if (seg.path.length === 1) return [{ u: seg.path[0][0], v: seg.path[0][1], t: 0 }];
  const param = new THREE.CatmullRomCurve3(seg.path.map(([u, v]) => new THREE.Vector3(u, v, 0)));
  const count = Math.max(32, seg.path.length * 14);
  const amp = (seg.wiggle ?? 0) * wiggleScale, phase = seg.name.length * 1.7;
  const bends = Math.max(1.2, param.getLength() * 2.2);
  return param.getSpacedPoints(count).map((s, i) => {
    const t = i / count;
    if (amp) {
      const tan = param.getTangentAt(t);
      const w = amp * 1.8 * noise3(t * bends + phase, phase * 0.37, 0.5) * (0.3 + 0.7 * t);
      s.x += -tan.y * w; s.y += tan.x * w;
    }
    return { u: s.x, v: s.y, t };
  });
}
```

In HeartScene, rebuild `VESSEL_VIEWS` from `VESSEL_VIEW_TABLE` through `vesselViews(point)`. This is the same maths, so the result is identical:
```ts
function vesselViews(point: (u: number, v: number) => THREE.Vector3) {
  return Object.fromEntries(VESSEL_VIEW_TABLE.map(([vessel, u, v, elevation]) => {
    const p = point(u, v).applyEuler(HEART_ROTATION); p.y = 0; p.normalize().setY(elevation);
    return [vessel, p.normalize()];
  })) as Record<Vessel, THREE.Vector3>;
}
const VESSEL_VIEWS = vesselViews((u, v) => surfacePoint(u, v));
```

### Step 3. `src/components/heartModel/config.ts` (20 min)
```ts
import type { Landmarks } from "./register";
const BASE = import.meta.env.BASE_URL;
const qs = new URLSearchParams(location.search);

export const HEART_MODEL = {
  candidates: ["models/heart.glb", "models/scene.glb", "models/scene.gltf"].map((p) => BASE + p),
  loadTimeoutMs: 25_000,
  /** Paste from ?calibrate "Copy JSON". null = auto-guess from the artist's pose. */
  landmarks: null as Landmarks | null,
  tune: {} as Partial<HeartTune>,                      // committed overrides (also from Copy JSON)
  credit: {
    title: "Realistic Human Heart", author: "neshallads", authorUrl: "https://sketchfab.com/neshallads",
    source: "https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089",
    license: "CC BY 4.0", licenseUrl: "https://creativecommons.org/licenses/by/4.0/",
  },
};
export const DEFAULT_TUNE = {
  twistDeg: 0, tubeScale: 1.15, embed: 0.45, wiggle: 0,          // sculpt has its own tortuosity
  minorBranches: "selected" as "selected" | "always",            // radius < 0.02 branches only when their vessel is selected
  autoAvFraction: 0.52,                                          // auto-guess only: AV height as a fraction of model height
  clearcoat: 0.6, clearcoatRoughness: 0.2, roughnessScale: 1, sheen: 0, aoMapIntensity: 0.7,
  bloomThreshold: 0.95, aoPass: 1.6, offset: [0, 0, 0] as [number, number, number], hideMeshes: [] as string[],
};
export type HeartTune = typeof DEFAULT_TUNE;
export const HEART_DEBUG = {
  forced: qs.get("heart"),                                       // glb | procedural
  overlay: qs.has("heartdebug") || qs.has("calibrate"),
  calibrate: import.meta.env.DEV && qs.has("calibrate"),
};
const num = (k: string) => { const s = qs.get(k); return s !== null && Number.isFinite(+s) ? +s : undefined; };
export function readTune(): HeartTune {
  const t: HeartTune = { ...DEFAULT_TUNE, ...HEART_MODEL.tune };
  const url: Partial<Record<keyof HeartTune, number | undefined>> = {
    twistDeg: num("twist"), tubeScale: num("tube"), embed: num("embed"), autoAvFraction: num("av"),
    clearcoat: num("cc"), bloomThreshold: num("bloom"), aoPass: num("ao"), roughnessScale: num("rough"),
  };
  for (const [k, v] of Object.entries(url)) if (v !== undefined) (t as Record<string, unknown>)[k] = v;
  if (qs.get("minor") === "always") t.minorBranches = "always";
  const off = qs.get("offset")?.split(",").map(Number);
  if (off?.length === 3 && off.every(Number.isFinite)) t.offset = off as [number, number, number];
  return t;
}
const KEY = "cv.heartCalib.v1";                                  // read only in ?calibrate mode
export const readSaved = (): { landmarks?: Partial<Landmarks> } => { try { return JSON.parse(localStorage.getItem(KEY) ?? "{}"); } catch { return {}; } };
export const writeSaved = (v: object) => { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* private mode */ } };
```

### Step 4. `heartModel/bake.ts` (30 min)
```ts
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
const KEEP = ["position", "normal", "uv", "uv1", "tangent", "color"];      // uv1: aoMap may use texCoord 1
const GET = ["getX", "getY", "getZ", "getW"] as const;
function toFloat32(a: THREE.BufferAttribute | THREE.InterleavedBufferAttribute) {   // de-interleave + de-quantise
  const k = a.itemSize, out = new Float32Array(a.count * k);
  for (let i = 0; i < a.count; i++) for (let j = 0; j < k; j++) out[i * k + j] = a[GET[j]](i);
  return new THREE.BufferAttribute(out, k);
}
export interface BakedModel { geometry: THREE.BufferGeometry; material: THREE.MeshStandardMaterial; meshes: { name: string; verts: number; material: string }[]; warnings: string[] }
export function bakeModel(root: THREE.Object3D, hide: string[] = []): BakedModel {
  root.updateMatrixWorld(true);
  const parts: THREE.BufferGeometry[] = [], meshes: BakedModel["meshes"] = [], warnings: string[] = [];
  let material: THREE.MeshStandardMaterial | null = null;
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh || !m.visible) return;
    const mat = (Array.isArray(m.material) ? m.material[0] : m.material) as THREE.MeshStandardMaterial;
    meshes.push({ name: m.name, verts: m.geometry.attributes.position.count, material: mat.name });
    if (hide.includes(m.name) || /plane|ground|floor|backdrop|shadow/i.test(m.name)) return;
    if (material && mat !== material) warnings.push(`extra material on "${m.name}" ignored`);
    const g = new THREE.BufferGeometry();
    for (const k of KEEP) { const a = m.geometry.getAttribute(k); if (a && (k !== "color" || mat.vertexColors)) g.setAttribute(k, toFloat32(a)); }
    const n = g.attributes.position.count;
    g.setIndex(new THREE.BufferAttribute(m.geometry.index ? new Uint32Array(m.geometry.index.array) : Uint32Array.from({ length: n }, (_, i) => i), 1));
    g.applyMatrix4(m.matrixWorld);                       // Sketchfab Z-up root, unit scale, origin: all gone
    if (m.matrixWorld.determinant() < 0) {               // baking loses three's automatic front-face flip
      const ix = g.index!.array as Uint32Array;
      for (let i = 0; i < ix.length; i += 3) { const t = ix[i + 1]; ix[i + 1] = ix[i + 2]; ix[i + 2] = t; }
    }
    if (!g.getAttribute("normal")) g.computeVertexNormals();
    parts.push(g); material ??= mat;
  });
  if (!parts.length || !material) throw new Error("[heart] model contains no meshes");
  const common = Object.keys(parts[0].attributes).filter((k) => parts.every((p) => p.getAttribute(k)));
  for (const p of parts) for (const k of Object.keys(p.attributes)) if (!common.includes(k)) p.deleteAttribute(k);
  const geometry = parts.length === 1 ? parts[0] : mergeGeometries(parts, false);
  if (!geometry) throw new Error("[heart] could not merge model meshes");
  if (import.meta.env.DEV) console.table(meshes);        // tells us whether the coronaries are separate meshes
  return { geometry, material, meshes, warnings };
}
```

### Step 5. `heartModel/register.ts`: alignment (1.25 h)
```ts
import * as THREE from "three";
import { APEX_Y, AV, LONG_AXIS, LAD_GROOVE_V, PDA_GROOVE_V } from "../heartAnatomy";
export type V3 = [number, number, number];
/** Raw model space (node transforms baked). Required: apex, ladMid, avLeft, avRight. */
export interface Landmarks { apex: V3; ladMid: V3; avLeft: V3; avRight: V3; avCrux?: V3; pdaMid?: V3; leftOstium?: V3; rightOstium?: V3 }
export const REQUIRED = ["apex", "ladMid", "avLeft", "avRight"] as const;
export interface Warp { toModel(v: number): number; toAnatomical(theta: number): number; uScale(v: number): number; anchors: [number, number][] }
export interface Registration { matrix: THREE.Matrix4; inverse: THREE.Matrix4; scale: number; warp: Warp;
  ostia: { left: THREE.Vector3 | null; right: THREE.Vector3 | null }; warnings: string[] }
const TAU = Math.PI * 2, wrap = (a: number) => ((a % TAU) + TAU) % TAU;

export function registerFromLandmarks(L: Landmarks, twistDeg = 0): Registration {
  const warnings: string[] = [], P = (a: V3) => new THREE.Vector3(a[0], a[1], a[2]);
  const apex = P(L.apex), left = P(L.avLeft), right = P(L.avRight);
  const top = left.clone().add(right).multiplyScalar(0.5);
  const y = top.clone().sub(apex).normalize();                                   // apex -> base
  const x = left.clone().sub(top); x.addScaledVector(y, -x.dot(y)).normalize();  // v = 0, lateral LV
  const z = new THREE.Vector3().crossVectors(x, y);                              // v = pi/2, anterior
  const av = [left, right, ...(L.avCrux ? [P(L.avCrux)] : [])];
  const avHeight = av.reduce((s, p) => s + p.clone().sub(apex).dot(y), 0) / av.length;
  if (!(avHeight > 0)) throw new Error("[heart] landmarks: AV groove is not above the apex");
  const scale = (AV * LONG_AXIS) / avHeight;                                     // mean AV groove -> y = 0.475
  const matrix = new THREE.Matrix4().makeTranslation(0, APEX_Y, 0)
    .multiply(new THREE.Matrix4().makeScale(scale, scale, scale))
    .multiply(new THREE.Matrix4().makeBasis(x, y, z).transpose())                // orthonormal: transpose = inverse
    .multiply(new THREE.Matrix4().makeTranslation(-apex.x, -apex.y, -apex.z));
  const H = (a?: V3) => (a ? P(a).applyMatrix4(matrix) : null);
  const th = (q: THREE.Vector3 | null) => (q ? wrap(Math.atan2(q.z, q.x)) : null);
  const uOf = (q: THREE.Vector3) => (q.y - APEX_Y) / LONG_AXIS;
  const lad = H(L.ladMid)!, lft = H(L.avLeft)!, rgt = H(L.avRight)!, crux = H(L.avCrux), pda = H(L.pdaMid);
  const tLad = th(lad)!;
  if (tLad < 0.3 || tLad > 2.6) warnings.push(`LAD azimuth ${(tLad * 57.3).toFixed(0)} deg is not anterior: landmarks swapped or model mirrored`);
  const angle = angleWarp([[0, 0], [LAD_GROOVE_V, tLad], [Math.PI, th(rgt)], [PDA_GROOVE_V, th(crux) ?? th(pda)]], twistDeg, warnings);
  const uA: [number, number][] = [[0, uOf(lft) / AV], [Math.PI, uOf(rgt) / AV]];
  if (crux) uA.push([PDA_GROOVE_V, uOf(crux) / AV]);
  const uScale = circularScalar(uA.map(([v, s]) => [v, THREE.MathUtils.clamp(s, 0.85, 1.15)] as [number, number]));
  return { matrix, inverse: matrix.clone().invert(), scale, warp: { ...angle, uScale },
    ostia: { left: H(L.leftOstium), right: H(L.rightOstium) }, warnings };
}

/** Circular, monotone, piecewise-linear v -> theta through the groove anchors (fidelity), plus a global twist. */
function angleWarp(raw: [number, number | null][], twistDeg: number, warnings: string[]) {
  const pts: [number, number][] = [];
  for (const [v, t0] of raw) {
    if (t0 == null) continue;
    let t = t0; while (t < v - Math.PI) t += TAU; while (t > v + Math.PI) t -= TAU;
    const prev = pts.length ? pts[pts.length - 1][1] : -Infinity;
    if (t <= prev + 0.15 || (pts.length > 0 && t >= pts[0][1] + TAU - 0.15)) { warnings.push(`landmark at v=${v.toFixed(2)} breaks groove order; ignored`); continue; }
    pts.push([v, t]);
  }
  const closed: [number, number][] = [...pts, [pts[0][0] + TAU, pts[0][1] + TAU]];
  const pl = (q: number, a: 0 | 1) => {
    const b = (1 - a) as 0 | 1, x = closed[0][a] + wrap(q - closed[0][a]);
    for (let i = 0; i < closed.length - 1; i++) { const p = closed[i], n = closed[i + 1];
      if (x <= n[a]) return p[b] + ((x - p[a]) / (n[a] - p[a])) * (n[b] - p[b]); }
    return x;
  };
  const tw = THREE.MathUtils.degToRad(twistDeg);
  return { toModel: (v: number) => pl(v, 0) + tw, toAnatomical: (t: number) => pl(t - tw, 1), anchors: pts };
}
function circularScalar(a: [number, number][]) {
  const pts = [...a].sort((p, q) => p[0] - q[0]); pts.push([pts[0][0] + TAU, pts[0][1]]);
  return (v: number) => { const x = pts[0][0] + wrap(v - pts[0][0]);
    for (let i = 0; i < pts.length - 1; i++) if (x <= pts[i + 1][0]) { const p = pts[i], n = pts[i + 1]; return p[1] + ((x - p[0]) / (n[0] - p[0])) * (n[1] - p[1]); }
    return pts[0][1]; };
}

/**
 * Zero-click first guess. It assumes the artist's own pose: glTF +Y up (vessels on top, apex lowest),
 * anterior surface towards +Z, patient's left towards +X. A 3ds Max export converted by Sketchfab ends up
 * like this. Only heights and azimuths matter to the registration, so the points need not lie on the surface.
 */
export function autoGuessLandmarks(geo: THREE.BufferGeometry, avFraction = 0.52): Landmarks {
  const pos = geo.getAttribute("position") as THREE.BufferAttribute;
  const box = new THREE.Box3().setFromBufferAttribute(pos), h = box.max.y - box.min.y, p = new THREE.Vector3();
  const slice = (f0: number, f1: number) => { const c = new THREE.Vector3(); let n = 0;
    for (let i = 0; i < pos.count; i++) { p.fromBufferAttribute(pos, i); const f = (p.y - box.min.y) / h; if (f >= f0 && f <= f1) { c.add(p); n++; } }
    return n ? c.divideScalar(n) : null; };
  const apex = slice(0, 0.02)!;
  const c = (slice(0.18, 0.22) ?? apex).clone().add(slice(0.33, 0.37) ?? apex).multiplyScalar(0.5);
  let axis = c.sub(apex).normalize();
  if (!(axis.y > 0.5)) axis = new THREE.Vector3(0, 1, 0);         // lying model: give up on tilt, calibrate
  const t = (box.min.y + avFraction * h - apex.y) / axis.y;
  const avC = apex.clone().addScaledVector(axis, t), mid = apex.clone().addScaledVector(axis, 0.5 * t);
  const X = new THREE.Vector3(1, 0, 0).addScaledVector(axis, -axis.x).normalize();
  const Z = new THREE.Vector3().crossVectors(X, axis), r = 0.3 * h;
  const at = (o: THREE.Vector3, v: number) => o.clone().addScaledVector(X, r * Math.cos(v)).addScaledVector(Z, r * Math.sin(v)).toArray() as V3;
  return { apex: apex.toArray() as V3, avLeft: at(avC, 0), avRight: at(avC, Math.PI), ladMid: at(mid, LAD_GROOVE_V) };
}
```

### Step 6. `heartModel/projector.ts`: coronary projection (1.25 h)
```ts
import * as THREE from "three";
import type { MeshBVH } from "three-mesh-bvh";
import { APEX_Y, LONG_AXIS, baseRadius, vesselRadius, vesselSamples, type VesselPath } from "../heartAnatomy";
import type { Warp } from "./register";

export interface MeshProjector {
  project(u: number, v: number, p: THREE.Vector3, n: THREE.Vector3): boolean;
  snap(p: THREE.Vector3, n: THREE.Vector3, maxDist?: number): boolean;
  raycastAll(r: THREE.Ray): THREE.Intersection[];
  isExit(h: THREE.Intersection, dir: THREE.Vector3): boolean;
  firstExit(r: THREE.Ray): THREE.Intersection | null;
  axisOrigin(uModel: number, out: THREE.Vector3): THREE.Vector3;
  stats: { rays: number; misses: number };
}
/** +1 if CCW faces agree with the authored (outward) vertex normals. Independent of the registration. */
export function windingSign(geo: THREE.BufferGeometry): 1 | -1 {
  const pos = geo.getAttribute("position"), nrm = geo.getAttribute("normal"), ix = geo.index!;
  const tri = new THREE.Triangle(), fn = new THREE.Vector3(), vn = new THREE.Vector3(), t = new THREE.Vector3();
  let s = 0;
  for (let f = 0; f < ix.count; f += 3) {
    const a = ix.getX(f), b = ix.getX(f + 1), c = ix.getX(f + 2);
    tri.setFromAttributeAndIndices(pos, a, b, c).getNormal(fn);
    vn.fromBufferAttribute(nrm, a).add(t.fromBufferAttribute(nrm, b)).add(t.fromBufferAttribute(nrm, c));
    s += Math.sign(fn.dot(vn));
  }
  return s >= 0 ? 1 : -1;
}
export function makeMeshProjector(bvh: MeshBVH, warp: Warp): MeshProjector {
  const geo = bvh.geometry, ws = windingSign(geo), pos = geo.getAttribute("position"), ix = geo.index!;  // index read AFTER the BVH reordered it
  const ray = new THREE.Ray(), aim = new THREE.Vector3(), tri = new THREE.Triangle(), stats = { rays: 0, misses: 0 };
  const raycastAll = (r: THREE.Ray) => bvh.raycast(r, THREE.DoubleSide).sort((a, b) => a.distance - b.distance);
  const isExit = (h: THREE.Intersection, d: THREE.Vector3) => ws * h.face!.normal.dot(d) > 0;   // face.normal = raw geometric normal
  // From inside the solid, the first exit is the epicardium, even under an overhanging auricle or the PA root.
  const firstExit = (r: THREE.Ray) => { const hs = raycastAll(r); return hs.find((h) => isExit(h, r.direction)) ?? hs[0] ?? null; };
  const axisOrigin = (uu: number, out: THREE.Vector3) => out.set(0, APEX_Y + LONG_AXIS * THREE.MathUtils.clamp(uu, 0.3, 0.92), 0);
  const snap = (p: THREE.Vector3, n: THREE.Vector3, maxDist = 0.5) => {
    const hit = bvh.closestPointToPoint(p, { point: new THREE.Vector3(), distance: 0, faceIndex: 0 }, 0, maxDist);
    if (!hit) return false;
    const f = 3 * hit.faceIndex;
    tri.setFromAttributeAndIndices(pos, ix.getX(f), ix.getX(f + 1), ix.getX(f + 2)).getNormal(n).multiplyScalar(ws);
    p.copy(hit.point); return true;
  };
  return { stats, raycastAll, isExit, firstExit, axisOrigin, snap,
    project(u, v, p, n) {
      const theta = warp.toModel(v), uu = Math.max(u, 0) * warp.uScale(v), r = 0.9 * baseRadius(uu);
      aim.set(r * Math.cos(theta), APEX_Y + LONG_AXIS * uu, r * Math.sin(theta));
      axisOrigin(uu, ray.origin);
      ray.direction.subVectors(aim, ray.origin);
      if (ray.direction.lengthSq() < 1e-10) ray.direction.set(0, -1, 0); else ray.direction.normalize();
      stats.rays++;
      const hit = firstExit(ray);
      if (hit) {
        p.copy(hit.point);
        n.copy(hit.normal ?? hit.face!.normal);          // three-mesh-bvh turns .normal towards the ray origin...
        if (n.dot(ray.direction) < 0) n.negate();         // ...so flip it back outward
        return true;
      }
      stats.misses++; p.copy(aim); n.copy(ray.direction);
      return snap(p, n);                                   // never leave a sample floating
    } };
}

export interface PlacedVessel { curve: THREE.CatmullRomCurve3; labelPos: THREE.Vector3; hitRatio: number; outliers: number }
export function placeVesselOnMesh(seg: VesselPath, proj: MeshProjector,
  o: { embed: number; radiusScale: number; wiggle: number; ostium?: THREE.Vector3 | null }): PlacedVessel {
  // The procedural 3-D lead-ins were placed for the procedural aorta. On the mesh, use a short surface stub
  // so the LM and RCA emerge from the real surface, or the clicked ostium (below).
  const path: [number, number][] = seg.lead && !o.ostium ? [[seg.path[0][0] + 0.09, seg.path[0][1] + 0.06], ...seg.path] : seg.path;
  const samples = vesselSamples({ ...seg, path }, o.wiggle), radius = vesselRadius(seg);
  const pts: THREE.Vector3[] = [], nrm: THREE.Vector3[] = [];
  let hits = 0;
  for (const s of samples) { const p = new THREE.Vector3(), n = new THREE.Vector3(); if (proj.project(s.u, s.v, p, n)) hits++; pts.push(p); nrm.push(n); }
  const outliers = rejectRadialOutliers(pts, nrm);
  smooth(pts, 2); smooth(nrm, 2, true);
  pts.forEach((p, i) => p.addScaledVector(nrm[i], radius(samples[i].t) * o.radiusScale * o.embed));   // tapered radius: thin ends do not float
  if (o.ostium) {
    const os = o.ostium.clone(), on = new THREE.Vector3(), mid = new THREE.Vector3(), mn = new THREE.Vector3();
    if (proj.snap(os, on, 0.3)) os.addScaledVector(on, seg.radius * o.radiusScale * 0.5);
    mid.lerpVectors(os, pts[0], 0.5);
    if (proj.snap(mid, mn, 0.3)) mid.addScaledVector(mn, seg.radius * o.radiusScale * o.embed);
    pts.unshift(os, mid); nrm.unshift(on, mn);
  }
  const i = Math.round(0.3 * (pts.length - 1));
  return { curve: new THREE.CatmullRomCurve3(pts, false, "centripetal"), labelPos: clearLabel(pts[i], nrm[i], proj),
           hitRatio: hits / samples.length, outliers };
}
function rejectRadialOutliers(pts: THREE.Vector3[], nrm: THREE.Vector3[], k = 3, tol = 0.12) {
  const rho = pts.map((p) => Math.hypot(p.x, p.z));
  const bad = rho.map((r, i) => { const w = [...rho.slice(Math.max(0, i - k), i), ...rho.slice(i + 1, i + 1 + k)].sort((a, b) => a - b);
    return w.length > 0 && Math.abs(r - w[w.length >> 1]) > tol; });
  for (let i = 0; i < pts.length; i++) if (bad[i]) {
    let a = i - 1; while (a >= 0 && bad[a]) a--; let b = i + 1; while (b < pts.length && bad[b]) b++;
    if (a < 0 && b >= pts.length) continue;
    const pa = a >= 0 ? a : b, pb = b < pts.length ? b : a, f = pa === pb ? 0 : (i - pa) / (pb - pa);
    pts[i].lerpVectors(pts[pa], pts[pb], f); nrm[i].lerpVectors(nrm[pa], nrm[pb], f).normalize();
  }
  return bad.filter(Boolean).length;
}
function smooth(a: THREE.Vector3[], passes: number, normalize = false) {
  const m = new THREE.Vector3();
  for (let k = 0; k < passes; k++) for (let i = 1; i < a.length - 1; i++) { a[i].lerp(m.addVectors(a[i - 1], a[i + 1]).multiplyScalar(0.5), 0.5); if (normalize) a[i].normalize(); }
}
function clearLabel(p: THREE.Vector3, n: THREE.Vector3, proj: MeshProjector, lift = 0.18) {
  const d = n.clone().normalize();
  const hs = proj.raycastAll(new THREE.Ray(p.clone().addScaledVector(d, 0.02), d)).filter((h) => h.distance < 0.8);
  return p.clone().addScaledVector(d, Math.max(lift, hs.length ? hs[hs.length - 1].distance + 0.12 : 0));  // escape an overhanging auricle
}
```

### Step 7. `heartModel/territory.ts`: per-vertex territory data and the shader patch (45 min)
```ts
import * as THREE from "three";
import { APEX_Y, LONG_AXIS, territoryWeights } from "../heartAnatomy";
import type { MeshProjector } from "./projector";
import type { Warp } from "./register";
import type { HeartTune } from "./config";

export function writeTerritoryAttributes(geo: THREE.BufferGeometry, proj: MeshProjector, warp: Warp) {
  const pos = geo.getAttribute("position"), n = pos.count, ss = THREE.MathUtils.smoothstep;
  const terr = new Float32Array(n * 4), param = new Float32Array(n * 2), p = new THREE.Vector3(), ray = new THREE.Ray();
  for (let i = 0; i < n; i++) {
    p.fromBufferAttribute(pos, i);
    const uu = (p.y - APEX_Y) / LONG_AXIS, v = warp.toAnatomical(Math.atan2(p.z, p.x)), u = uu / warp.uScale(v);
    let mask = 1 - ss(u, 0.97, 1.05);                         // ventricles only; follows the (tilted) clicked AV groove
    if (mask > 0) {                                           // outer-shell test: auricle tips over the groove are beyond the first exit
      proj.axisOrigin(uu, ray.origin); ray.direction.subVectors(p, ray.origin);
      const d = ray.direction.length();
      if (d > 1e-6) { ray.direction.divideScalar(d); const ex = proj.firstExit(ray); if (ex) mask *= 1 - ss(d - ex.distance, 0.02, 0.07); }
    }
    const w = territoryWeights(Math.max(0, u), v);            // same function as the procedural heart
    terr[4 * i] = w.lad; terr[4 * i + 1] = w.lcx; terr[4 * i + 2] = w.rca; terr[4 * i + 3] = mask;
    param[2 * i] = u; param[2 * i + 1] = v;
  }
  geo.setAttribute("aTerr", new THREE.BufferAttribute(terr, 4));
  geo.setAttribute("param", new THREE.BufferAttribute(param, 2));
}

export function makeTerritoryUniforms() {
  return { uRisk: { value: [new THREE.Color(), new THREE.Color(), new THREE.Color()] }, uSel: { value: new THREE.Vector3() },
    uSelColor: { value: new THREE.Color() }, uTerrMix: { value: 0 }, uSelMix: { value: 0 }, uPulse: { value: 0 } };
}
export type TerritoryUniforms = ReturnType<typeof makeTerritoryUniforms>;

/** Built field by field: MeshStandardMaterial.copy() into a physical material resets defines in r169. */
export function makeTissueMaterial(src: THREE.MeshStandardMaterial, U: TerritoryUniforms, t: HeartTune, maxAniso: number) {
  const m = new THREE.MeshPhysicalMaterial({
    map: src.map, color: src.color.clone(), normalMap: src.normalMap, normalMapType: src.normalMapType, normalScale: src.normalScale.clone(),
    roughnessMap: src.roughnessMap, roughness: Math.min(1, src.roughness * t.roughnessScale),
    metalness: 0,                                             // Sketchfab often exports metallic 1; tissue is a dielectric. No metalnessMap.
    aoMap: src.aoMap, aoMapIntensity: t.aoMapIntensity, emissiveMap: src.emissiveMap, emissive: src.emissive.clone(),
    vertexColors: src.vertexColors, side: src.side, transparent: false, alphaTest: src.alphaTest,
    clearcoat: t.clearcoat, clearcoatRoughness: t.clearcoatRoughness,          // wet film over the sculpted base
    sheen: t.sheen, sheenRoughness: 0.5, sheenColor: new THREE.Color("#ff5a48"), envMapIntensity: 0.9,
  });
  for (const tx of [m.map, m.normalMap]) if (tx) tx.anisotropy = Math.min(8, maxAniso);
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, U);                            // shared {value} objects: UI changes = uniform writes
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nattribute vec4 aTerr;\nvarying vec4 vTerr;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvTerr = aTerr;");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", `#include <common>
varying vec4 vTerr;
uniform vec3 uRisk[3]; uniform vec3 uSel; uniform vec3 uSelColor;
uniform float uTerrMix; uniform float uSelMix; uniform float uPulse;
float cvSel = 0.0;
vec3 cvRecolour(vec3 tint, float lum) { return tint * (0.3 + 1.6 * sqrt(lum)); }   // keeps fat/fibre/painted detail`)
      .replace("#include <color_fragment>", `#include <color_fragment>
{
  float lum = dot(diffuseColor.rgb, vec3(0.2126, 0.7152, 0.0722));
  float fatty = smoothstep(0.15, 0.4, lum);                    // pink basal fat takes less tint (procedural: 0.62 - 0.25*fat)
  vec3 terr = uRisk[0] * vTerr.x + uRisk[1] * vTerr.y + uRisk[2] * vTerr.z;
  diffuseColor.rgb = mix(diffuseColor.rgb, cvRecolour(terr, lum), uTerrMix * vTerr.w * (0.72 - 0.3 * fatty));
  cvSel = vTerr.w * uSelMix * clamp((dot(vTerr.xyz, uSel) - 0.35) / 0.65, 0.0, 1.0);
  diffuseColor.rgb = mix(diffuseColor.rgb, cvRecolour(uSelColor, lum), 0.55 * cvSel);
}`)
      .replace("#include <emissivemap_fragment>", `#include <emissivemap_fragment>
totalEmissiveRadiance += uSelColor * cvSel * (0.04 + 0.06 * uPulse);`);   // faint glow in phase with the artery pulse
  };
  m.customProgramCacheKey = () => "cv-heart-tissue-v1";
  return m;
}
```
The injection point order in r169 meshphysical (map, then color, then normal, then emissivemap) has been checked.

### Step 8. `heartModel/fitHeart.ts`: one function that does everything (40 min)
```ts
import * as THREE from "three";
import { SAH, computeBoundsTree, disposeBoundsTree } from "three-mesh-bvh";
import { ARTERIES, VESSEL_VIEW_TABLE, baseRadius, type Vessel } from "../heartAnatomy";
// + imports of bake/register/projector/territory/config
export type LandmarkSource = "config" | "calibration" | "auto";
const PROC_WORLD_CENTER = new THREE.Vector3(-0.31, 0.38, -0.16);   // procedural heart world bbox centre (scratch bbox.ts)
const PROC_WORLD_HEIGHT = 2.95;
export interface HeartFit {
  id: string; geometry: THREE.BufferGeometry; material: THREE.MeshPhysicalMaterial; uniforms: TerritoryUniforms;
  projector: MeshProjector; registration: Registration; landmarks: Landmarks; landmarkSource: LandmarkSource;
  arteries: Record<string, PlacedVessel>; views: Record<Vessel, THREE.Vector3>;
  placement: { position: THREE.Vector3; floorY: number; cameraScale: number };
  diag: { ms: Record<string, number>; gates: ReturnType<typeof checkRegistration>; warnings: string[]; meshes: BakedModel["meshes"] };
}
export function fitHeartModel(scene: THREE.Object3D, o: { landmarks: Landmarks | null; landmarkSource: LandmarkSource; tune: HeartTune;
  rotation: THREE.Euler; maxAnisotropy: number; strict: boolean }): HeartFit {
  const t0 = performance.now(), ms: Record<string, number> = {}, mark = (k: string) => (ms[k] = Math.round(performance.now() - t0));
  const baked = bakeModel(scene, o.tune.hideMeshes); mark("bake");
  const landmarks = o.landmarks ?? autoGuessLandmarks(baked.geometry, o.tune.autoAvFraction);
  const registration = registerFromLandmarks(landmarks, o.tune.twistDeg);
  const geometry = baked.geometry.applyMatrix4(registration.matrix);     // det > 0: winding preserved
  geometry.computeBoundsTree = computeBoundsTree;
  geometry.disposeBoundsTree = disposeBoundsTree;                         // hygiene; this mesh is NOT inside drei <Bvh>
  geometry.computeBoundsTree({ strategy: SAH, maxLeafTris: 10 }); mark("bvh");
  const projector = makeMeshProjector(geometry.boundsTree!, registration.warp);
  const gates = checkRegistration(projector, registration.warp);
  if (o.strict && !gates.ok) throw new Error(`[heart] registration rejected (${gates.reason}); open ?calibrate`);
  const arteries = Object.fromEntries(ARTERIES.map((seg) => [seg.name, placeVesselOnMesh(seg, projector, {
    embed: o.tune.embed, radiusScale: o.tune.tubeScale, wiggle: o.tune.wiggle,
    ostium: seg.vessel === "lm" ? registration.ostia.left : seg.name === "RCA" ? registration.ostia.right : null })])); mark("arteries");
  writeTerritoryAttributes(geometry, projector, registration.warp); mark("territories");
  geometry.computeBoundingBox(); geometry.computeBoundingSphere();
  const uniforms = makeTerritoryUniforms();
  const material = makeTissueMaterial(baked.material, uniforms, o.tune, o.maxAnisotropy);
  const warnings = [...baked.warnings, ...registration.warnings];
  for (const tx of [material.map, material.normalMap, material.roughnessMap, material.aoMap]) {
    const w = (tx?.image as { width?: number } | undefined)?.width; if (w && w > 2048) warnings.push(`texture ${w}px > 2048: run npm run model:prep`);
  }
  const p = new THREE.Vector3(), n = new THREE.Vector3();
  const views = Object.fromEntries(VESSEL_VIEW_TABLE.map(([vessel, u, v, el]) => { projector.project(u, v, p, n);
    const d = p.clone().applyEuler(o.rotation); d.y = 0; d.normalize().setY(el); return [vessel, d.normalize()]; })) as Record<Vessel, THREE.Vector3>;
  const fit: HeartFit = { id: THREE.MathUtils.generateUUID(),                      // not crypto.randomUUID: undefined on http LAN demos
    geometry, material, uniforms, projector, registration, landmarks, landmarkSource: o.landmarks ? o.landmarkSource : "auto",
    arteries, views, placement: placeInScene(geometry, o.rotation, o.tune.offset), diag: { ms, gates, warnings, meshes: baked.meshes } };
  mark("total");
  if (warnings.length || fit.landmarkSource === "auto") console.warn("[heart]", fit.landmarkSource, warnings, ms);
  return fit;
}
/** Non-trivial gates: (a) the long axis runs inside the ventricles, (b) the epicardium sits at a heart-sized radius. */
function checkRegistration(proj: MeshProjector, warp: Warp) {
  const ray = new THREE.Ray(), p = new THREE.Vector3(), n = new THREE.Vector3(), ratios: number[] = [];
  let exits = 0, total = 0;
  for (const u of [0.35, 0.55, 0.75, 0.9]) for (let k = 0; k < 12; k++) {
    const v = (k / 12) * Math.PI * 2;
    proj.axisOrigin(u, ray.origin); ray.direction.set(Math.cos(v), 0, Math.sin(v));
    const first = proj.raycastAll(ray)[0]; total++; if (first && proj.isExit(first, ray.direction)) exits++;
    proj.project(u, v, p, n); ratios.push(Math.hypot(p.x, p.z) / Math.max(0.05, baseRadius(u * warp.uScale(v))));
  }
  ratios.sort((a, b) => a - b);
  const inside = exits / total, radius = ratios[ratios.length >> 1], ok = inside >= 0.85 && radius > 0.6 && radius < 1.7;
  return { ok, inside, radius, reason: ok ? "" : `axis-inside ${inside.toFixed(2)}, radius ratio ${radius.toFixed(2)}` };
}
function placeInScene(geo: THREE.BufferGeometry, rotation: THREE.Euler, offset: [number, number, number]) {
  const q = new THREE.Quaternion().setFromEuler(rotation), pos = geo.getAttribute("position"), p = new THREE.Vector3(), box = new THREE.Box3();
  for (let i = 0; i < pos.count; i++) box.expandByPoint(p.fromBufferAttribute(pos, i).applyQuaternion(q));
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const position = PROC_WORLD_CENTER.clone().sub(c).add(new THREE.Vector3(...offset));   // sit where the schematic heart sits
  return { position, floorY: box.min.y + position.y - 0.05, cameraScale: THREE.MathUtils.clamp(Math.max(size.x, size.y) / PROC_WORLD_HEIGHT, 1, 1.5) };
}
```
React StrictMode is on in `main.tsx`, so this runs twice in dev. That is about 0.2-0.6 s behind the loading overlay and does not happen in production.

### Step 9. `heartModel/useHeartSource.ts`: loading and fallback (40 min)
```ts
import { useGLTF } from "@react-three/drei";
import { useEffect, useState } from "react";
import { HEART_DEBUG, HEART_MODEL } from "./config";
export type HeartSource = "probing" | "glb" | "procedural";
export async function probeModel(urls: string[]) {
  for (const url of urls) {
    try {
      let res = await fetch(url, { method: "HEAD", cache: "no-cache" });
      if (res.status === 405 || res.status === 501) res = await fetch(url, { headers: { Range: "bytes=0-3" }, cache: "no-cache" });
      // Vite's htmlFallbackMiddleware (and Netlify/Vercel rewrites) answer a missing .glb with index.html and 200.
      if (res.ok && !(res.headers.get("content-type") ?? "").includes("text/html")) return url;
    } catch { /* offline or blocked: try the next candidate */ }
  }
  return null;
}
const PREF = "cv.heartSource";
export function useHeartSource() {
  const [url, setUrl] = useState<string | null | undefined>(undefined);
  const [failed, setFailed] = useState<string | null>(null);
  const [pref, setPref] = useState<"glb" | "procedural">(() => {
    const f = HEART_DEBUG.forced; if (f === "procedural" || f === "schematic") return "procedural"; if (f === "glb") return "glb";
    try { return (localStorage.getItem(PREF) as "glb" | "procedural" | null) ?? "glb"; } catch { return "glb"; }
  });
  useEffect(() => { let alive = true;
    probeModel(HEART_MODEL.candidates).then((u) => { if (!alive) return; if (u) useGLTF.preload(u, false, true); setUrl(u); });  // no Draco: no gstatic fetch
    return () => { alive = false; }; }, []);
  const source: HeartSource = url === undefined ? "probing" : url && pref === "glb" && !failed ? "glb" : "procedural";
  return { source, url: url ?? null, available: !!url, failed,
    fail(reason: unknown) { console.warn("[heart] realistic model unavailable, using procedural heart:", reason); if (url) useGLTF.clear(url); setFailed(String(reason)); },
    choose(p: "glb" | "procedural") { try { localStorage.setItem(PREF, p); } catch { /* private mode */ } if (p === "glb") setFailed(null); setPref(p); } };
}
```
Do not preload at module scope: on a fresh clone without the asset that would fire a request that fails.

### Step 10. `HeartScene.tsx` integration (1.5 h)

**a) Imports.** Add `Component, Suspense, type ReactNode` from react; `Line, useGLTF, useProgress` from drei; `acceleratedRaycast` from three-mesh-bvh; plus the heartModel modules.

**b) Error boundary.** It must sit **inside** `<Canvas>`. An uncaught error there would otherwise unmount the whole Canvas through R3F's own boundary.
```tsx
class ModelBoundary extends Component<{ onError: (e: unknown) => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(e: unknown) { this.props.onError(e); }
  render() { return this.state.failed ? null : this.props.children; }
}
```

**c) State in `HeartScene`.**
```tsx
const heart = useHeartSource(); const glb = heart.source === "glb";
const tune = useMemo(readTune, []);                                  // must be referentially stable
const calib = useCalibration(HEART_DEBUG.calibrate);                // calib.active is memoised by JSON key (see Step 11)
const [fit, setFit] = useState<HeartFit | null>(null);
const { progress } = useProgress();
useEffect(() => { setReady(false); if (!glb) setFit(null); }, [heart.source]);          // overlay covers every switch
useEffect(() => { if (!glb || ready) return; const id = setTimeout(() => heart.fail("load timeout"), HEART_MODEL.loadTimeoutMs);
  return () => clearTimeout(id); }, [glb, ready]);
```
The `landmarks` and `tune` objects passed to `GlbHeart` must be stable references. Otherwise `onFit` causes a re-render, which builds a new object, which triggers a re-fit, in an endless loop.

**d) Canvas body.** Remove the root `<FirstFrame>`: it would reveal the canvas while the model is still suspended.
```tsx
{glb && heart.url && (
  <ModelBoundary key={heart.url} onError={(e) => heart.fail(e)}>
    <Suspense fallback={null}>
      <GlbHeart url={heart.url} landmarks={calib.active} landmarkSource={calib.source} tune={tune}
        probs={probs} selected={selected} hovered={hovered} mode={mode} pins={pins}
        bpm={beating && !HEART_DEBUG.calibrate ? bpm : 0} calibrating={HEART_DEBUG.calibrate} showArteries={calib.showTubes}
        onPick={calib.pick} onSelect={onSelect} onHover={setHovered} onFit={setFit} onReady={() => setReady(true)} />
    </Suspense>
  </ModelBoundary>)}
{heart.source === "procedural" && (<><Bvh firstHitOnly><Heart probs={probs} selected={selected} hovered={hovered} mode={mode}
  pins={pins} bpm={beating ? bpm : 0} onSelect={onSelect} onHover={setHovered} /></Bvh><FirstFrame onReady={() => setReady(true)} /></>)}
<ContactShadows position={[0, glb && fit ? fit.placement.floorY : -1.85, 0]} opacity={0.6} scale={7} blur={2.6} far={3.5} resolution={512} />
{fx && (<EffectComposer multisampling={4}>
  <N8AO aoRadius={0.3} distanceFalloff={0.8} intensity={glb ? tune.aoPass : 2.2} quality="medium" halfRes />
  <Bloom mipmapBlur luminanceThreshold={glb ? tune.bloomThreshold : 0.9} luminanceSmoothing={0.2} intensity={0.8} radius={0.6} />
  <ToneMapping mode={ToneMappingMode.ACES_FILMIC} /><Vignette offset={0.3} darkness={0.7} /></EffectComposer>)}
<FitCamera scale={glb && fit ? fit.placement.cameraScale : 1} />
<CameraRig focus={selected} resetKey={resetKey} views={glb && fit ? fit.views : VESSEL_VIEWS} />
```
- `FitCamera`: add a `scale = 1` prop and multiply the 6.8 base distance by it.
- `CameraRig`: add a `views = VESSEL_VIEWS` prop; use `views[focus]`; deps `[focus, views]`.

**e) `GlbHeart`.** It lives in HeartScene.tsx next to `Heart`, which avoids a circular import of `Artery`.
```tsx
function GlbHeart(p: HeartProps & { url: string; landmarks: Landmarks | null; landmarkSource: LandmarkSource; tune: HeartTune;
  calibrating: boolean; showArteries: boolean; onPick: (raw: V3) => void; onFit: (f: HeartFit) => void; onReady: () => void }) {
  const { probs, selected, hovered, mode, pins, bpm, onSelect, onHover, tune } = p;
  const gltf = useGLTF(p.url, false, true);                          // no Draco: drei would fetch the gstatic decoder
  const { gl, scene, camera } = useThree();
  const fit = useMemo(() => fitHeartModel(gltf.scene, { landmarks: p.landmarks, landmarkSource: p.landmarkSource, tune,
    rotation: HEART_ROTATION, maxAnisotropy: gl.capabilities.getMaxAnisotropy(), strict: !p.calibrating }),
    [gltf.scene, p.landmarks, p.landmarkSource, tune, p.calibrating, gl]);       // throws -> ModelBoundary -> procedural
  const mesh = useRef<THREE.Mesh>(null), ventricles = useRef<THREE.Group>(null), phase = useRef(0);
  const selTarget = useMemo(() => new THREE.Vector3(), []), selColor = useMemo(() => new THREE.Color(), []);
  const U = fit.uniforms;
  useEffect(() => { let live = true; p.onFit(fit);
    gl.compileAsync(scene, camera).catch(() => undefined)                  // KHR_parallel_shader_compile; overlay hides the hitch
      .then(() => requestAnimationFrame(() => requestAnimationFrame(() => live && p.onReady())));
    return () => { live = false; }; }, [fit]);
  useEffect(() => {
    (["lad", "lcx", "rca"] as const).forEach((v, i) => (probs ? riskColor(probs[v], U.uRisk.value[i]) : U.uRisk.value[i].copy(NEUTRAL_ARTERY)));
    if (selected) (probs ? riskColor(probs[selected], selColor) : selColor.copy(NEUTRAL_ARTERY));
    selTarget.set(+(selected === "lad"), +(selected === "lcx"), +(selected === "rca"));
  }, [U, probs, selected]);
  useFrame(({ clock }, dt) => {
    phase.current = (phase.current + (dt * bpm) / 60) % 1;              // same beat curve as Heart; the specimen is one mesh
    const t = phase.current, pulse = (c: number, w: number) => (bpm > 0 ? Math.exp(-(((t - c) / w) ** 2)) : 0);
    const v = pulse(0.16, 0.07) + 0.25 * pulse(0.34, 0.05);
    ventricles.current?.scale.set(1 - 0.035 * v, 1 - 0.018 * v, 1 - 0.035 * v);
    const terr = probs && mode === "territories" ? 1 : 0, k = 1 - Math.exp(-8 * dt);   // same precedence as Myocardium's effect
    U.uTerrMix.value = THREE.MathUtils.damp(U.uTerrMix.value, terr, 8, dt);
    U.uSelMix.value = THREE.MathUtils.damp(U.uSelMix.value, selected && !terr ? 1 : 0, 8, dt);
    U.uSel.value.lerp(selTarget, k); U.uSelColor.value.lerp(selColor, k);
    U.uPulse.value = 0.5 + 0.5 * Math.sin(clock.elapsedTime * 4);          // in phase with the Artery glow
  });
  const onWall = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation(); if (!e.face || !mesh.current) return;
    if (p.calibrating) { p.onPick(mesh.current.worldToLocal(e.point.clone()).applyMatrix4(fit.registration.inverse).toArray() as V3); return; }
    const f = e.face, A = fit.geometry.getAttribute("aTerr") as THREE.BufferAttribute;
    if (HEART_DEBUG.overlay && e.nativeEvent.shiftKey) { const pr = fit.geometry.getAttribute("param"); console.info("[heart] u,v", pr.getX(f.a).toFixed(3), pr.getY(f.a).toFixed(3)); return; }
    const avg = (c: number) => (A.getComponent(f.a, c) + A.getComponent(f.b, c) + A.getComponent(f.c, c)) / 3;
    if (avg(3) < 0.4) return;                                              // atria, auricles, great vessels: inert, as today
    const w = { lad: avg(0), lcx: avg(1), rca: avg(2) };
    const vessel = (Object.keys(w) as Vessel[]).reduce((a, b) => (w[a] >= w[b] ? a : b));
    onSelect(vessel === selected ? null : vessel);
  };
  return (
    <group rotation={HEART_ROTATION} position={fit.placement.position}>
      <group ref={ventricles}>
        {/* Own BVH, deliberately OUTSIDE drei <Bvh>. Its mount-only effect and its unmount dispose would strip
            the tree under StrictMode or on a schematic toggle. It still honours raycaster.firstHitOnly. */}
        <mesh ref={mesh} geometry={fit.geometry} material={fit.material} raycast={acceleratedRaycast} onClick={onWall} onPointerMove={stop} />
        {p.showArteries && (
          <Bvh firstHitOnly key={fit.id}>                                    {/* keyed: rebuilds when a re-fit changes the tubes */}
            {ARTERIES.map((seg) => (
              <Artery key={seg.name} seg={seg} placed={fit.arteries[seg.name]} radiusScale={tune.tubeScale}
                minor={tune.minorBranches === "selected" && seg.radius < 0.02}
                probs={probs} selected={selected} hovered={hovered} occluder={mesh} onSelect={onSelect} onHover={onHover} />))}
          </Bvh>)}
        {pins && fit.pins && <AnatomyPins occluders={[mesh]} positions={fit.pins} />}      {/* Step 12, optional */}
      </group>
      {HEART_DEBUG.overlay && <HeartDebugOverlay fit={fit} />}
    </group>);
}
```

**f) `Artery`: three optional props.** The procedural output does not change.
```tsx
interface ArteryProps { /* existing */ placed?: PlacedVessel; radiusScale?: number; minor?: boolean }
function Artery({ seg, placed, radiusScale = 1, minor = false, ...rest }: ArteryProps) {
  const { geometry, labelPos, tip, tipRadius } = useMemo(() => {
    const curve = placed?.curve ?? vesselCurve(seg);
    const base = vesselRadius(seg), radius = radiusScale === 1 ? base : (t: number) => base(t) * radiusScale;
    return { geometry: taperedTube(curve, radius, 180, 14),
      labelPos: placed ? placed.labelPos.clone() : (() => { const p = curve.getPoint(0.3); return p.add(new THREE.Vector3(p.x, 0, p.z).normalize().multiplyScalar(0.18)); })(),
      tip: curve.getPoint(1), tipRadius: radius(1) };
  }, [seg, placed, radiusScale]);
  // ...all existing hooks unchanged...
  if (minor && !isSelected) return null;                                 // after the hooks
  // ...existing JSX unchanged...
}
```

**g) DOM additions.**
- **Overlay text.** In GLB mode: `` `Loading anatomical model… ${progress.toFixed(0)}%` ``. Otherwise: "Building 3D heart model…".
- **Toggle.** Shown only when `heart.available`: `{glb ? "Realistic" : "Schematic"}`, calling `heart.choose(glb ? "procedural" : "glb")`. The title should say it switches between the specimen mesh and the schematic heart.
- **Credit.** Bottom-right, shown when `glb && ready`. Text and links are in section 6.
- **Anatomy button.** Hide it in GLB mode if Step 12 is cut.
- **Debug panel.** `{HEART_DEBUG.overlay && fit && <HeartDebugPanel fit={fit} calib={calib} tune={tune} />}`.

### Step 11. Debug and calibration tools: `heartModel/HeartDebug.tsx` (1.25 h)

**URL switches** (no dat or leva needed):

| Switch | Effect |
|---|---|
| `?heart=procedural`, `?heart=glb` | Force the source |
| `?heartdebug` | Show the overlays and the diagnostics panel |
| `?calibrate` | DEV only. Landmark picking; heartbeat paused |
| `twist=<deg>` | Rotate all paths about the long axis |
| `tube=<scale>` | Tube radius scale |
| `embed=<0..1>` | Centreline height above the surface, in radii |
| `av=<0..1>` | Auto-guess only: AV height fraction |
| `cc=<0..1>` | Clearcoat |
| `rough=<scale>` | Roughness scale |
| `bloom=<threshold>` | Bloom threshold in GLB mode |
| `ao=<intensity>` | N8AO intensity in GLB mode |
| `minor=always` | Always show minor branches |
| `offset=x,y,z` | Shift the model in the scene |

Shift-click in `?heartdebug` mode logs the (u,v) of the clicked point.

- **`HeartDebugOverlay`**, drawn in the heart frame. Give every element `raycast={() => null}` and `depthTest={false}`.
  - `<axesHelper args={[1.2]} position={[0, APEX_Y, 0]} />`.
  - A dashed long-axis `<Line>` from `(0, APEX_Y, 0)` to `(0, BASE_Y + 0.8, 0)`.
  - A procedural ghost: `buildMyocardium(36, 64)` as a wireframe, opacity 0.15.
  - A **yellow AV ring**: `projector.project(AV, v)` for 97 values of v.
  - **Cyan groove lines**: `project(u, LAD_GROOVE_V)` and `project(u, PDA_GROOVE_V)` for u from 0.05 to 1.
  - Landmark spheres with small `Html` tags, positioned at `P(raw).applyMatrix4(fit.registration.matrix)`.
- **`HeartDebugPanel`** (DOM, top-right). It shows:
  - the landmark source (config, calibration or auto);
  - the scale;
  - the warp anchors in degrees;
  - the AV u per anchor;
  - the gates (axis-inside, radius ratio), shown red when failed;
  - hitRatio and outlier count per trunk;
  - `diag.ms` timings;
  - texture sizes;
  - the mesh table;
  - the warnings.
- **`useCalibration(enabled)`**.
  - Saved picks from localStorage are read **only** in calibrate mode, so a stale pick never overrides the committed config.
  - `active` is memoised by a JSON key: `useMemo(() => complete ? picks : HEART_MODEL.landmarks, [JSON.stringify(...)])`.
  - Steps:
    1. apex, "tip of the LV";
    2. ladMid, "on the engraved LAD halfway down the anterior groove";
    3. avLeft, "AV groove on the lateral wall, seen from the left";
    4. avRight, "AV groove at the right border";
    5. avCrux (optional), "posterior crux";
    6. pdaMid (optional), "PDA halfway down";
    7. leftOstium (optional);
    8. rightOstium (optional).
  - Controls: Undo, Skip (optional steps), Clear, "Tubes on/off" (so tubes do not swallow clicks on the grooves), and Copy JSON. Copy JSON uses `navigator.clipboard.writeText` (localhost counts as a secure origin), prints the JSON to the console, and also shows it in a read-only textarea.
  - Once the 4 required picks exist, the live fit uses them. In calibrate mode `strict` is false: failed gates show red instead of falling back.

### Step 12. Optional: anatomy pins in GLB mode (40 min)
- `anatomyLabels.ts`: give the ventricular labels a `uv: [u, v, lift]` field next to the existing position, so the values are unchanged.
- `AnatomyPins`: add an optional `positions?: Record<string, THREE.Vector3 | null>` prop and skip null entries.
- In `fitHeartModel`, compute `pins`:
  - Labels with `uv`: `projector.project(u, v)` plus the normal times the lift.
  - Other labels: `projector.snap(position, n, 0.35)` plus `n * 0.06`, or null if nothing is within reach.
- Check the great-vessel pins by eye and drop any that land on the wrong vessel.

### Step 13. Documentation (20 min)
- README: add a Credits section (section 6), and replace "procedural, no external mesh" in the feature list with "realistic CC BY specimen mesh with a procedural fallback".
- HANDOFF §7b: record that the earlier decision against a GLB is reversed, and why: CC BY, our own overlays in (u,v), and the fallback.
- `public/models/README.txt`: how to regenerate `heart.glb`.

## 3. Visual tuning once the file arrives (timebox 1.5 h)

| # | Check (and debug aid) | Knob | Expected range |
|---|---|---|---|
| 1 | `inspect` output: one mesh, ~22.6k triangles, textures no larger than 2048, no `KHR_draco`. The console table at load lists the meshes | `hideMeshes` if there is a ground plane or separate coronary meshes | – |
| 2 | `?heartdebug` with the auto-guess: apex at the bottom, ghost wireframe roughly inside the ventricles, yellow AV ring near the crease, cyan LAD line on the anterior surface | `av` (0.45-0.6), `twist` (180 if the model faces −Z) | – |
| 3 | `?calibrate`: click the 4 required landmarks (+ avCrux, pdaMid; ostia only if visible) | – | About 5 min |
| 3a | From the anterior view, the LAD tube lies in the engraved anterior groove | Re-click ladMid | – |
| 3b | From the inferior view (select RCA), the PDA lies in the posterior groove | Click avCrux or pdaMid | – |
| 3c | From the left view (select LCX), the LCX lies in the AV crease | Re-click avLeft (height) | – |
| 3d | The RCA follows the right AV groove | Re-click avRight | – |
| 3e | Copy JSON, paste into `HEART_MODEL.landmarks` and `.tune`, reload without parameters; the panel must say "config" | – | – |
| 4 | Tubes sit in the grooves and cover the dark engraved line, without floating or being buried | `embed` | 0.35-0.55 |
| | | `tube` | 1.0-1.35 |
| | | `twist` | Should stay 0 once calibrated |
| 5 | Textbook branches that run over smooth sculpt look like extra vessels | `minorBranches` | Keep "selected" unless it looks better always on |
| 6 | Territories mode with the three-vessel preset: tint stops at the AV groove; nothing on auricles or aortic root; boundaries roughly along the IV grooves | Height mask `0.97/1.05` | – |
| | | Shell test `0.02/0.07` (territory.ts) | – |
| 7 | Tint strength: red reads red, fat at the base not muddy | `0.72 - 0.3*fatty` | – |
| | | `cvRecolour` gains `0.3 + 1.6·sqrt(lum)` | – |
| 8 | Wet look without a plastic look | `cc` | 0.4-0.8 |
| | | `clearcoatRoughness` | 0.15-0.3 |
| | | `rough` | 0.8-1.1 |
| | | `aoMapIntensity` | 0.5-1 |
| | | `sheen` | 0-0.4 |
| | | `ao` (N8AO; baked AO already darkens the grooves) | 1.2-2.0 |
| 9 | Bloom: select a **red** artery and confirm it blooms at its pulse peak; wet highlights must not glow. Red emissive has luminance of only about 0.77 at peak | `bloom` | 0.85-1.0 |
| 10 | Placement on a narrow window: framing and shadow contact | `offset` | – |
| | | `cameraScale` clamp in `placeInScene` | – |
| 11 | Labels: all three visible and hidden when behind the heart; none stuck inside the left auricle | `clearLabel` lift | 0.18 / 0.12 |
| 12 | Clicks: anterior wall selects LAD, lateral wall LCX, inferior wall and RV RCA; atria and aorta do nothing. Shift-click logs (u,v) | Mask threshold | 0.4 |
| 13 | Performance on the Intel laptop (Chrome Rendering, FPS meter), HD on, full screen, at least 55 fps | `dpr={[1,1.5]}`, N8AO `quality="low"` | – |

## 4. Verification before committing
- `npm run build` passes (`tsc -b` strict). Watch for the `vite-env.d.ts` reference and the `three-mesh-bvh` type augmentation of `computeBoundsTree` and `disposeBoundsTree`.
- **Fallback.**
  - Rename `heart.glb`: the procedural heart appears, with one console warning.
  - `?heart=procedural` shows the procedural heart.
  - Point a candidate URL at a corrupt file: the boundary falls back.
  - Throttle the network to "Slow 3G": after 25 s the timeout falls back.
  - Toggle Realistic, Schematic, Realistic: no errors, no stuck overlay, picking still fast. This would have broken if the GLB mesh sat inside drei `<Bvh>`.
- **Both modes:** Arteries and Territories cross-fade smoothly; selection pulse and bloom; camera focus per vessel; heartbeat on and off; label occlusion while rotating; Reset.
- **Production preview:** `npm run build && npm run preview` with the GLB present. Confirm the credit line shows and links work.

## 5. Cuts, what not to cut, and the decision gate

**If short on time, cut in this order (saves roughly 4 h):**
1. The useProgress percentage.
2. Pins in GLB mode (Step 12): hide the Anatomy button there.
3. IV-groove debug lines and the shift-click logger.
4. Ostium landmarks: keep the surface stubs.
5. The `clearLabel` push-out: use a fixed 0.25 lift instead.
6. The shell ray test in the territory mask: use the height mask only.
7. The emissive pulse on the selected territory, and the `uSel`/`uSelColor` cross-fade: switch instantly.
8. Automatic placement and camera scale: use the `offset` URL parameter and constants.
9. WebP conversion: keep the resize to 2048 at minimum, because 4K maps are about 340 MB of VRAM on the integrated GPU.

**Never cut:**
- the probe with `text/html` rejection, the boundary inside the Canvas, and the timeout fallback;
- the CC BY credit in the UI, the README and the licence file;
- landmark registration with committed JSON;
- the first-exit projector;
- the `aTerr` shader tint and click-to-territory;
- label occlusion against the GLB;
- `compileAsync` before the overlay hides;
- the procedural path kept as it is.

**Decision gate, end of Fri 10 Oct:** if the trunks still do not sit plausibly in the engraved grooves after calibration, switch the default preference to `"procedural"` and label the toggle "Realistic (beta)". It is a one-line change in `useHeartSource`; everything else already supports it.

**Estimate:** about 10.5 h of implementation plus 1.5 h of tuning; about 7 h with the cuts.

## 6. Attribution text

**UI credit** (bottom-right, GLB mode only, links open with `target="_blank" rel="noopener noreferrer"`):
`3D heart: "Realistic Human Heart" by neshallads · CC BY 4.0 · modified`
- The title links to https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089
- "neshallads" links to https://sketchfab.com/neshallads
- "CC BY 4.0" links to https://creativecommons.org/licenses/by/4.0/

**README "Credits"** (also used as `public/models/CREDITS.md`):
> **3D heart model:** "Realistic Human Heart" by neshallads (https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089), licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
>
> **Changes:**
> - Converted to a single binary glTF (`frontend/public/models/heart.glb`), with textures resized to 2048 px and re-encoded as WebP.
> - At runtime CardioVision re-orients and scales the mesh, tints it by predicted perfusion risk, and draws its own coronary-artery overlays, labels and animation on top. These are not part of the original work.
>
> The author does not endorse this project. Licence text: `frontend/public/models/LICENSE-heart.txt`. If the asset is missing, the app falls back to its own procedural heart.

**Demo video and Devpost:** use the same one-line credit plus both links, and keep the UI credit visible in the video frame.

## 7. Judge-flagged problems and where each is fixed

| Problem | Fix |
|---|---|
| Outside-in rays ride over the auricles and PA root (autofit) | First-exit ray from inside the axis (Step 6) |
| One global azimuth cannot fit LAD and PDA together (autofit); per-vessel du/dv moves the whole RCA+PDA tree (pragmatic) | Circular angle warp through the LAD, right-border and crux anchors, plus a uScale(v) warp of AV height (Step 5) |
| Default pose assumed HEART_ROTATION; the model is upright (pragmatic, fidelity) | Auto-guess uses the native glTF pose; 4-click calibration is authoritative |
| Bounding-box scale included the arch (pragmatic) | Scale comes from AV-groove height above the apex |
| Per-sample closest-point snaps jump between surfaces and kink the tube (pragmatic) | Ray projection, radial outlier rejection, Laplacian smoothing; closest point only as a fallback |
| Quality gate was vacuous (fidelity) | Axis-inside exit test and radius-ratio test, with winding taken from vertex normals; strict mode throws and falls back |
| `disposeBoundsTree` missing; Bvh mount and StrictMode problems | Methods attached; GLB mesh outside drei `<Bvh>`; arteries in `<Bvh key={fit.id}>` |
| `crypto.randomUUID` | `THREE.MathUtils.generateUUID` |
| `uv1` dropped | Kept, along with `color` when the material uses vertex colours |
| Metalness copied | Forced to 0, no metalnessMap |
| `copy()` into a physical material | Built field by field |
| Draco and the gstatic decoder | `useGLTF(url, false, true)`; never compress with Draco |
| No `compileAsync` (pragmatic); the root FirstFrame would reveal an empty canvas | `compileAsync` before the overlay hides; FirstFrame moved into the procedural branch |
| Tone-mapping switch and changes to shared Artery/tube code (fidelity) | Not done: ACES kept; Artery only gains optional props with today's defaults |
| Ridge snapping enabled by default (fidelity) | Not used: the coronaries are engraved, and the experiment showed snapping makes alignment worse |
| Fit time under-estimated | Runs once behind the overlay; landmarks committed in config; StrictMode double run is dev only |