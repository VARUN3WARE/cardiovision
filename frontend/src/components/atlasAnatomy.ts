/**
 * BodyParts3D atlas heart and chest (built by frontend/scripts/build-bp3d.mjs from the official
 * BodyParts3D v4.0 meshes; every coronary branch is a named FMA structure).
 *
 * Frame: glTF metres, +X patient's left, +Y superior, +Z anterior (anatomical position), so the
 * default camera on +Z gives the standard anterior view. Each GLB node is one part (one draw call);
 * its extras name the part's layer, the model output it shows (vessel) and its branches, and the
 * per-vertex `_branch` attribute indexes into that branch list.
 */
import * as THREE from "three";
import { acceleratedRaycast } from "three-mesh-bvh";
import { VESSELS, VESSEL_IDS, type Vessel } from "../anatomy/registry";
import { whenIdle } from "../lab/idle";

export const ATLAS_HEART_URL = "/models/bp3d_heart.glb";
export const ATLAS_THORAX_URL = "/models/bp3d_thorax.glb";
export const ATLAS_CREDIT = {
  text: "BodyParts3D, © The Database Center for Life Science, CC BY 4.0",
  url: "https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html",
};

/** World units per metre when displayed: the heart is then about as large as the other models. */
export const ATLAS_SCALE = 19;

export type AtlasLayer =
  | "myocardium" | "atrium" | "artery" | "pulmonary" | "vein" | "coronary" | "cardiac_vein"
  | "skin" | "bone" | "cartilage" | "diaphragm" | "airway";

export interface AtlasBranch {
  fj: string; // BodyParts3D element file
  fma: string; // FMA concept id
  name: string; // FMA English name
}

export interface AtlasPart {
  key: string;
  layer: AtlasLayer;
  label: string;
  fma: string;
  /** Model output this part shows; "lm" (left main) is not labelled in the dataset. */
  vessel?: Vessel | "lm";
  branches: AtlasBranch[];
  geometry: THREE.BufferGeometry; // metres, centred on the heart
}

export interface PreparedAtlasHeart {
  parts: AtlasPart[];
  center: THREE.Vector3; // heart centre in the GLB frame (metres)
  /** Ventricular wall, per vertex: one territory weight per registry artery (VESSEL_IDS order, sum 1). */
  territory: Float32Array;
  /** Ventricular wall, per vertex: epicardial fat amount 0..1 (fat fills the coronary grooves). */
  fat: Float32Array;
  /** Camera direction (unit) facing each vessel's territory. */
  views: Record<Vessel, THREE.Vector3>;
  /** Label anchors on each vessel's main trunk, lifted off the surface (metres, centred). */
  labels: Record<Vessel, THREE.Vector3>;
}

const N = VESSEL_IDS.length;
// Softness of territory boundaries (mm): weights ~ exp(-(d - dmin) / TAU).
const TERRITORY_TAU_MM = 2.2;

/** Copy every mesh out of the glTF scene as plain float geometry in one frame (dequantised). */
function bakeParts(scene: THREE.Object3D, center: THREE.Vector3 | null) {
  scene.updateMatrixWorld(true);
  const parts: AtlasPart[] = [];
  scene.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const src = mesh.geometry;
    const g = new THREE.BufferGeometry();
    g.setIndex(src.index ? Array.from(src.index.array) : null);
    for (const name of Object.keys(src.attributes)) {
      const a = src.getAttribute(name);
      const arr = new Float32Array(a.count * a.itemSize);
      for (let i = 0; i < a.count; i++) for (let k = 0; k < a.itemSize; k++) arr[i * a.itemSize + k] = a.getComponent(i, k);
      g.setAttribute(name, new THREE.BufferAttribute(arr, a.itemSize));
    }
    g.applyMatrix4(mesh.matrixWorld);
    const x = (mesh.userData ?? {}) as Partial<AtlasPart>;
    parts.push({
      key: x.key ?? mesh.name,
      layer: (x.layer ?? "myocardium") as AtlasLayer,
      label: x.label ?? mesh.name,
      fma: x.fma ?? "",
      vessel: x.vessel,
      branches: x.branches ?? [],
      geometry: g,
    });
  });
  if (center) for (const p of parts) p.geometry.translate(-center.x, -center.y, -center.z);
  for (const p of parts) {
    p.geometry.computeBoundingBox();
    p.geometry.computeBoundingSphere();
  }
  return parts;
}

/** The glTF loader puts node extras on the parent object of a mesh in some exports; merge both. */
function liftExtras(scene: THREE.Object3D) {
  scene.traverse((o) => {
    if ((o as THREE.Mesh).isMesh && !o.userData.key && o.parent?.userData?.key) o.userData = { ...o.parent.userData };
  });
}

export function prepareAtlasHeart(scene: THREE.Object3D): PreparedAtlasHeart {
  liftExtras(scene);
  const raw = bakeParts(scene, null);
  const byKey = Object.fromEntries(raw.map((p) => [p.key, p]));
  const wall = byKey.ventricles;
  if (!wall) throw new Error("atlas heart has no ventricular wall");

  // Pivot: centre of the ventricles + atria (the great vessels would pull it upwards).
  const box = new THREE.Box3();
  for (const k of ["ventricles", "atria"]) if (byKey[k]) box.union(byKey[k].geometry.boundingBox!);
  const center = box.getCenter(new THREE.Vector3());
  for (const p of raw) {
    p.geometry.translate(-center.x, -center.y, -center.z);
    p.geometry.computeBoundingBox();
    p.geometry.computeBoundingSphere();
  }

  // Territories from the baked distance (0.1 mm units) to each registry artery's tree.
  const dist = VESSELS.map((v) => {
    const a = wall.geometry.getAttribute(`_d_${v.atlasPart}`);
    if (!a) throw new Error(`atlas heart has no distance attribute for ${v.atlasPart}`);
    return a;
  });
  const groove = wall.geometry.getAttribute("_d_groove");
  const n = wall.geometry.getAttribute("position").count;
  const territory = new Float32Array(n * N);
  const fat = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = dist.map((a) => a.getX(i) / 10); // mm
    const dmin = Math.min(...d);
    const w = d.map((x) => Math.exp(-(x - dmin) / TERRITORY_TAU_MM));
    const s = w.reduce((a, b) => a + b, 0);
    territory.set(w.map((x) => x / s), i * N);
    // Epicardial fat fills the grooves the main vessels run in, thinning out within a few mm.
    fat[i] = groove ? 1 - smoothstep(2, 7, groove.getX(i) / 10) : 0;
  }

  const parts = raw.filter((p) => p.geometry.getAttribute("position").count > 0);
  return {
    parts,
    center,
    territory,
    fat,
    views: vesselViews(wall.geometry, territory),
    labels: labelAnchors(byKey),
  };
}

export function prepareThorax(scene: THREE.Object3D, center: THREE.Vector3): AtlasPart[] {
  liftExtras(scene);
  return bakeParts(scene, center);
}

function smoothstep(e0: number, e1: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** Camera directions that face the centroid of each vessel's territory on the outer wall. */
function vesselViews(geometry: THREE.BufferGeometry, territory: Float32Array) {
  const pos = geometry.getAttribute("position");
  const nrm = geometry.getAttribute("normal");
  const sums = VESSEL_IDS.map(() => new THREE.Vector3());
  const p = new THREE.Vector3();
  const nv = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    nv.fromBufferAttribute(nrm, i);
    if (nv.dot(p) <= 0) continue; // outer (epicardial) surface only
    VESSEL_IDS.forEach((_, k) => {
      if (territory[i * N + k] > 0.6) sums[k].add(p);
    });
  }
  return Object.fromEntries(VESSEL_IDS.map((t, k) => {
    const dir = sums[k].normalize();
    // Face the territory horizontally, from a gently elevated (or, for the inferior wall, low) viewpoint.
    dir.y = Math.min(0.3, Math.max(-0.35, dir.y * 0.5 + 0.1));
    return [t, dir.normalize()];
  })) as Record<Vessel, THREE.Vector3>;
}

/** Where to pin each vessel's label: on its proximal trunk, nudged outward from the heart. */
function labelAnchors(byKey: Record<string, AtlasPart>): Record<Vessel, THREE.Vector3> {
  const lm = byKey.lm?.geometry.boundingSphere?.center ?? new THREE.Vector3();
  const anchor = (key: string, trunkFma: string | null, along: number) => {
    const part = byKey[key];
    const pos = part.geometry.getAttribute("position");
    const br = part.geometry.getAttribute("_branch");
    // Trunk branch: by FMA id, else the branch whose vertices come closest to the left main.
    let trunk = part.branches.findIndex((b) => b.fma === trunkFma);
    const p = new THREE.Vector3();
    if (trunk < 0) {
      let best = Infinity;
      for (let i = 0; i < pos.count; i++) {
        const d = p.fromBufferAttribute(pos, i).distanceToSquared(lm);
        if (d < best) {
          best = d;
          trunk = br.getX(i);
        }
      }
    }
    // Order the trunk's vertices by distance from the vessel's origin and take one part-way along.
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < pos.count; i++) if (br.getX(i) === trunk) pts.push(new THREE.Vector3().fromBufferAttribute(pos, i));
    const origin = pts.reduce((a, b) => (a.distanceToSquared(lm) <= b.distanceToSquared(lm) ? a : b));
    pts.sort((a, b) => a.distanceToSquared(origin) - b.distanceToSquared(origin));
    const at = pts[Math.floor(pts.length * along)].clone();
    return at.add(at.clone().normalize().multiplyScalar(0.008));
  };
  return Object.fromEntries(
    VESSELS.map((v) => [v.id, anchor(v.atlasPart, v.atlasLabel.trunkFma, v.atlasLabel.along)]),
  ) as Record<Vessel, THREE.Vector3>;
}

// ---------------------------------------------------------------------------------------------------
// Ischaemia lab (teaching sandbox). Computed on first use of the lab, not at load:
//  - distance from the coronary ostium along every coronary tree (drives the flow pulses);
//  - per artery, the SANDBOX narrowing marker on its proximal trunk and which vertices carry the blood
//    that has passed through it;
//  - the ventricular-wall vertex under every vessel vertex (territory weights for hypokinesis).
// ---------------------------------------------------------------------------------------------------

/**
 * Where the sandbox narrowing sits: mm beyond the start of each artery's trunk, in its proximal segment (the
 * atlas LAD gives off its first branch ~31 mm along). The proximal trunks run under the atrial appendages and
 * the great arteries, so the lab view shows those see-through.
 */
const LAB_MARKER_MM: Record<Vessel, number> = { lad: 10, lcx: 12, rca: 14 };
const BRIDGE_MM = 0.6; // branches of one tree whose surfaces come this close are joined
const OSTIUM_MM = 2; // vertices this close to the ostium start the flow at distance 0
const FEED_MM = 2; // a tree this close to the left main is fed by it (the LAD and LCX)

export interface LabMarker {
  position: THREE.Vector3; // on the trunk's axis (metres, centred atlas frame)
  tangent: THREE.Vector3; // unit, downstream
  radius: number; // local vessel radius (metres)
  distance: number; // from the ostium, mm
  /** Per vertex of the artery's tree: 1 where the blood flows through the narrowing (the whole trunk and the branches beyond the marker). */
  restricted: Float32Array;
  /** Per vertex of the artery's tree: 0..1 weight of the waist drawn at the narrowing. */
  pinch: Float32Array;
  /** Camera direction (unit) from which the marker is in sight, as close to the artery's territory view as possible. */
  view: THREE.Vector3;
}

export interface AtlasLabAnatomy {
  /** Per coronary part key: distance from its coronary ostium along the tree (mm), per vertex. */
  ostium: Record<string, Float32Array>;
  /** Per coronary part key and the cardiac veins: index of the nearest ventricular-wall vertex. */
  wallIndex: Record<string, Uint32Array>;
  markers: Record<Vessel, LabMarker>;
}

const labCache = new WeakMap<PreparedAtlasHeart, AtlasLabAnatomy>();
const labJobs = new WeakMap<PreparedAtlasHeart, Generator<void, AtlasLabAnatomy>>();

/** Lab anatomy for a prepared atlas heart (cached per heart; finishes any work started in idle time). */
export function prepareLabAnatomy(atlas: PreparedAtlasHeart): AtlasLabAnatomy {
  for (;;) {
    const hit = labCache.get(atlas);
    if (hit) return hit;
    stepLabAnatomy(atlas);
  }
}

/**
 * Prepares the lab anatomy a step at a time (each a few tens of ms at most) while the browser is idle, so the
 * first time the lab opens it is ready. Returns a cancel function.
 */
export function prepareLabAnatomyWhenIdle(atlas: PreparedAtlasHeart): () => void {
  let cancel = () => {};
  let cancelled = false;
  const work = (deadline: { timeRemaining: () => number }) => {
    if (cancelled) return;
    try {
      do {
        if (labCache.has(atlas)) return;
        stepLabAnatomy(atlas);
      } while (deadline.timeRemaining() > 6);
    } catch (e) {
      console.warn("Lab anatomy could not be prepared in advance", e);
      return; // opening the lab retries (and reports the error there)
    }
    // Idle periods first; on a busy main thread a step still runs every ~0.3 s.
    if (!labCache.has(atlas)) cancel = whenIdle(work, 300);
  };
  cancel = whenIdle(work, 1500);
  return () => {
    cancelled = true;
    cancel();
  };
}

/** Runs one step of the heart's lab-anatomy job; the result goes to the cache when the last step is done. */
function stepLabAnatomy(atlas: PreparedAtlasHeart) {
  let job = labJobs.get(atlas);
  if (!job) labJobs.set(atlas, (job = labAnatomySteps(atlas)));
  let r: IteratorResult<void, AtlasLabAnatomy>;
  try {
    r = job.next();
  } catch (e) {
    labJobs.delete(atlas); // a failed job is not resumed: the next call starts again (and fails visibly)
    throw e;
  }
  if (r.done) {
    labCache.set(atlas, r.value);
    labJobs.delete(atlas);
  }
}

function* labAnatomySteps(atlas: PreparedAtlasHeart): Generator<void, AtlasLabAnatomy> {
  const byKey = Object.fromEntries(atlas.parts.map((p) => [p.key, p]));
  const inMm = new Map<string, Float32Array>();
  const mm = (p: AtlasPart) => {
    let P = inMm.get(p.key);
    if (!P) inMm.set(p.key, (P = toMm(p.geometry)));
    return P;
  };
  const aorta = byKey.aorta ? mm(byKey.aorta) : null;
  const lm = byKey.lm;
  const lmMm = lm ? mm(lm) : null;
  const ostium: Record<string, Float32Array> = {};
  const paths: Record<string, TreePaths> = {};
  yield;

  // The left main starts at the aorta; trees touching it (LAD, LCX) start where they leave it.
  let lmPaths: TreePaths | null = null;
  if (lm && lmMm) {
    lmPaths = treePaths(lm.geometry, lmMm, ostiumSeeds(lm, lmMm, aorta, null));
    ostium.lm = lmPaths.dist;
    yield;
  }
  for (const v of VESSELS) {
    const part = byKey[v.atlasPart];
    if (!part) continue;
    const P = mm(part);
    const fed = lmPaths && lmMm ? feedSeeds(P, lmMm, lmPaths.dist) : null;
    const seeds = fed ?? ostiumSeeds(part, P, aorta, v.atlasLabel.trunkFma);
    yield;
    paths[v.id] = treePaths(part.geometry, P, seeds);
    ostium[v.atlasPart] = paths[v.id].dist;
    yield;
  }

  // Only the ventricular wall can hide a marker: the lab shows the atria and great arteries see-through.
  const wallMesh = byKey.ventricles ? new THREE.Mesh(byKey.ventricles.geometry, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide })) : null;
  if (wallMesh?.geometry.boundsTree) wallMesh.raycast = acceleratedRaycast;
  const markers = Object.fromEntries(VESSELS.filter((v) => paths[v.id]).map((v) => {
    const part = byKey[v.atlasPart];
    const m = sandboxMarker(part, mm(part), paths[v.id], v.atlasLabel.trunkFma, LAB_MARKER_MM[v.id]);
    if (wallMesh) m.view.copy(markerView(m, atlas.views[v.id], wallMesh));
    return [v.id, m];
  })) as Record<Vessel, LabMarker>;
  (wallMesh?.material as THREE.Material | undefined)?.dispose();
  yield;

  const wall = byKey.ventricles;
  const wallIndex: Record<string, Uint32Array> = {};
  const grid = pointGrid(mm(wall), 4);
  for (const p of atlas.parts) {
    if (p.layer === "coronary" || p.layer === "cardiac_vein") {
      yield;
      wallIndex[p.key] = nearestIndices(mm(p), grid);
    }
  }
  return { ostium, wallIndex, markers };
}

interface TreePaths {
  dist: Float32Array; // mm from the ostium
  parent: Int32Array; // previous vertex on the shortest path (-1 at a seed)
  order: Int32Array; // vertices in the order they were reached
}

function toMm(g: THREE.BufferGeometry) {
  const pos = g.getAttribute("position");
  const out = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    out[3 * i] = pos.getX(i) * 1000;
    out[3 * i + 1] = pos.getY(i) * 1000;
    out[3 * i + 2] = pos.getZ(i) * 1000;
  }
  return out;
}

const dist2 = (A: Float32Array, a: number, B: Float32Array, b: number) =>
  (A[3 * a] - B[3 * b]) ** 2 + (A[3 * a + 1] - B[3 * b + 1]) ** 2 + (A[3 * a + 2] - B[3 * b + 2]) ** 2;

/** Seeds at the ostium: the trunk vertex nearest the aorta, and every vertex within OSTIUM_MM of it. */
function ostiumSeeds(part: AtlasPart, P: Float32Array, aorta: Float32Array | null, trunkFma: string | null): [number, number][] {
  const br = part.geometry.getAttribute("_branch");
  const trunk = part.branches.findIndex((b) => b.fma === trunkFma);
  const n = P.length / 3;
  let o = 0;
  let best = Infinity;
  for (let i = 0; i < n; i++) {
    if (trunk >= 0 && Math.round(br.getX(i)) !== trunk) continue;
    let d = Infinity;
    if (aorta) for (let j = 0; j < aorta.length / 3; j++) d = Math.min(d, dist2(P, i, aorta, j));
    else d = -P[3 * i + 1]; // no aorta in the asset: start at the highest point
    if (d < best) {
      best = d;
      o = i;
    }
  }
  const seeds: [number, number][] = [];
  for (let i = 0; i < n; i++) if (dist2(P, i, P, o) <= OSTIUM_MM * OSTIUM_MM) seeds.push([i, 0]);
  return seeds;
}

/** Seeds where a tree leaves the left main (null if it does not touch it): the left main's distance plus the gap. */
function feedSeeds(P: Float32Array, L: Float32Array, lmDist: Float32Array): [number, number][] | null {
  const seeds: [number, number][] = [];
  let closest: [number, number] = [-1, Infinity];
  for (let i = 0; i < P.length / 3; i++) {
    let d = Infinity;
    let k = -1;
    for (let j = 0; j < L.length / 3; j++) {
      const e = dist2(P, i, L, j);
      if (e < d) {
        d = e;
        k = j;
      }
    }
    const gap = Math.sqrt(d);
    if (gap < closest[1]) closest = [i, gap];
    if (gap <= 1) seeds.push([i, lmDist[k] + gap]);
  }
  if (closest[1] > FEED_MM) return null;
  return seeds.length ? seeds : null;
}

/**
 * Shortest paths over the tree's surface mesh from the seeds. Branches are separate meshes in the atlas,
 * so touching branches are joined (BRIDGE_MM) and any piece still unreached is joined to the nearest
 * reached vertex (the atlas has small gaps between some circumflex segments).
 */
function treePaths(g: THREE.BufferGeometry, P: Float32Array, seeds: [number, number][]): TreePaths {
  const n = P.length / 3;
  const br = g.getAttribute("_branch");
  const adj: number[][] = Array.from({ length: n }, () => []);
  const link = (a: number, b: number) => {
    if (a === b) return;
    const l = Math.sqrt(dist2(P, a, P, b));
    adj[a].push(b, l);
    adj[b].push(a, l);
  };
  const idx = g.index!.array;
  for (let t = 0; t < idx.length; t += 3) {
    link(idx[t], idx[t + 1]);
    link(idx[t + 1], idx[t + 2]);
    link(idx[t + 2], idx[t]);
  }
  const grid = pointGrid(P, BRIDGE_MM);
  for (let a = 0; a < n; a++) {
    grid.near(P, a, BRIDGE_MM, (b) => {
      if (b > a && Math.round(br.getX(a)) !== Math.round(br.getX(b))) link(a, b);
    });
  }

  // Float64 while searching: a heap key must compare equal to the distance stored for its vertex.
  const dist = new Float64Array(n).fill(Infinity);
  const parent = new Int32Array(n).fill(-1);
  const done = new Uint8Array(n);
  const order: number[] = [];
  const heap = new MinHeap();
  for (const [v, d] of seeds) {
    if (d < dist[v]) {
      dist[v] = d;
      heap.push(d, v);
    }
  }
  const run = () => {
    while (heap.size) {
      const [d, v] = heap.pop();
      if (done[v] || d > dist[v]) continue;
      done[v] = 1;
      order.push(v);
      const nb = adj[v];
      for (let k = 0; k < nb.length; k += 2) {
        const u = nb[k];
        const nd = d + nb[k + 1];
        if (nd < dist[u]) {
          dist[u] = nd;
          parent[u] = v;
          heap.push(nd, u);
        }
      }
    }
  };
  run();
  if (!order.length && n) {
    dist[0] = 0;
    heap.push(0, 0);
    run();
  }
  while (order.length < n) {
    let best = Infinity;
    let bu = -1;
    let bv = -1;
    for (let u = 0; u < n; u++) {
      if (done[u]) continue;
      for (const v of order) {
        const d = dist2(P, u, P, v);
        if (d < best) {
          best = d;
          bu = u;
          bv = v;
        }
      }
    }
    dist[bu] = dist[bv] + Math.sqrt(best);
    parent[bu] = bv;
    heap.push(dist[bu], bu);
    run();
  }
  return { dist: Float32Array.from(dist), parent, order: Int32Array.from(order) };
}

/** The sandbox narrowing on the artery's proximal trunk and the vertices downstream of it. */
function sandboxMarker(part: AtlasPart, P: Float32Array, paths: TreePaths, trunkFma: string | null, offsetMm: number): LabMarker {
  const br = part.geometry.getAttribute("_branch");
  const n = P.length / 3;
  const { dist, parent, order } = paths;
  const branch = (i: number) => Math.round(br.getX(i));
  // Trunk: by FMA id, else the branch the flow enters first.
  let trunk = part.branches.findIndex((b) => b.fma === trunkFma);
  if (trunk < 0) trunk = branch(order[0]);
  let d0 = Infinity;
  let d1 = 0;
  for (let i = 0; i < n; i++) {
    if (branch(i) !== trunk) continue;
    d0 = Math.min(d0, dist[i]);
    d1 = Math.max(d1, dist[i]);
  }
  const dm = d0 + Math.min(offsetMm, 0.4 * (d1 - d0));

  // Axis point and direction from the trunk's vertices around the marker.
  const centroid = (lo: number, hi: number) => {
    const c = new THREE.Vector3();
    let k = 0;
    for (let i = 0; i < n; i++) {
      if (branch(i) !== trunk || dist[i] < lo || dist[i] > hi) continue;
      c.x += P[3 * i];
      c.y += P[3 * i + 1];
      c.z += P[3 * i + 2];
      k++;
    }
    return k ? c.divideScalar(k) : null;
  };
  const at = centroid(dm - 1.5, dm + 1.5) ?? centroid(dm - 4, dm + 4) ?? new THREE.Vector3();
  const before = centroid(dm - 5, dm - 1) ?? at;
  const after = centroid(dm + 1, dm + 5) ?? at;
  const tangent = after.clone().sub(before);
  if (tangent.lengthSq() < 1e-9) tangent.set(0, -1, 0);
  tangent.normalize();
  let r = 0;
  let k = 0;
  const q = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    if (branch(i) !== trunk || Math.abs(dist[i] - dm) > 2) continue;
    q.set(P[3 * i], P[3 * i + 1], P[3 * i + 2]).sub(at);
    q.addScaledVector(tangent, -q.dot(tangent));
    r += q.length();
    k++;
  }
  const radiusMm = k ? r / k : 1.5;

  // Restricted: the whole trunk (flow is conserved along it) and every vertex whose path crosses the marker.
  const crossed = new Uint8Array(n);
  for (const v of order) {
    const p = parent[v];
    if (p < 0) continue;
    crossed[v] = crossed[p] || (branch(v) === trunk && dist[p] < dm && dist[v] >= dm) ? 1 : 0;
  }
  const restricted = new Float32Array(n);
  const pinch = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const onTrunk = branch(i) === trunk;
    restricted[i] = onTrunk || crossed[i] ? 1 : 0;
    pinch[i] = onTrunk ? 1 - smoothstep(1.2, 3.2, Math.abs(dist[i] - dm)) : 0;
  }
  return {
    position: at.multiplyScalar(0.001),
    tangent,
    radius: radiusMm * 0.001,
    distance: dm,
    restricted,
    pinch,
    view: new THREE.Vector3(0, 0, 1),
  };
}

/**
 * The candidate view (territory-facing first) that best shows the marker past the ventricular wall: its
 * centre line must be clear, then the more of a 4 mm ring of sight lines clear the better (the trunk runs
 * in a groove, and the heartbeat or a slightly different camera must not tuck it away).
 */
function markerView(m: LabMarker, territory: THREE.Vector3, wall: THREE.Object3D): THREE.Vector3 {
  const front = new THREE.Vector3(0, 0.015, 1).normalize();
  const lift = (v: THREE.Vector3, y: number) => v.clone().setY(v.y + y).normalize();
  const candidates = [
    lift(territory.clone().add(front.clone().multiplyScalar(0.6)), 0.1),
    territory.clone(),
    lift(front, 0.12),
    lift(front.clone().add(new THREE.Vector3(-0.35, 0, 0)), 0.2),
    lift(territory.clone(), 0.35),
    lift(front.clone().add(new THREE.Vector3(0.45, 0, 0)), 0.3),
  ];
  const ray = new THREE.Raycaster();
  const far = 0.36; // about the camera's distance from the heart, metres
  const hits = (from: THREE.Vector3, dir: THREE.Vector3, stop: number, objects: THREE.Object3D[]) => {
    ray.set(from.clone().addScaledVector(dir, far), dir.clone().negate());
    ray.far = far - stop;
    return ray.intersectObjects(objects, false).length > 0;
  };
  const score = (dir: THREE.Vector3) => {
    const u = new THREE.Vector3().crossVectors(dir, Math.abs(dir.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0)).normalize();
    const w = new THREE.Vector3().crossVectors(dir, u);
    let clear = hits(m.position, dir, 1.5 * m.radius, [wall]) ? 0 : 9;
    for (let k = 0; k < 8; k++) {
      const p = m.position.clone();
      p.addScaledVector(u, 0.004 * Math.cos((k * Math.PI) / 4)).addScaledVector(w, 0.004 * Math.sin((k * Math.PI) / 4));
      if (!hits(p, dir, 2.5 * m.radius, [wall])) clear++;
    }
    return clear;
  };
  let best = candidates[0];
  let bestScore = -1;
  for (const dir of candidates) {
    const sc = score(dir);
    if (sc > bestScore) {
      best = dir;
      bestScore = sc;
    }
  }
  return best;
}

/** Uniform grid over points (mm) for neighbour queries. */
function pointGrid(P: Float32Array, cell: number) {
  const cells = new Map<number, number[]>();
  const key = (x: number, y: number, z: number) => (x + 512) + (y + 512) * 1024 + (z + 512) * 1048576;
  const at = (v: number) => Math.floor(v / cell);
  for (let i = 0; i < P.length / 3; i++) {
    const k = key(at(P[3 * i]), at(P[3 * i + 1]), at(P[3 * i + 2]));
    const list = cells.get(k);
    if (list) list.push(i);
    else cells.set(k, [i]);
  }
  return {
    cell,
    points: P,
    /** Calls f for every grid point within r (<= cell) of point i of Q. */
    near(Q: Float32Array, i: number, r: number, f: (j: number) => void) {
      const x = at(Q[3 * i]), y = at(Q[3 * i + 1]), z = at(Q[3 * i + 2]);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        for (const j of cells.get(key(x + dx, y + dy, z + dz)) ?? []) if (dist2(Q, i, P, j) <= r * r) f(j);
      }
    },
    /** Nearest grid point to point i of Q, searching outward shell by shell. */
    nearest(Q: Float32Array, i: number) {
      const x = at(Q[3 * i]), y = at(Q[3 * i + 1]), z = at(Q[3 * i + 2]);
      let best = Infinity;
      let bj = 0;
      for (let s = 0; s < 64; s++) {
        for (let dx = -s; dx <= s; dx++) for (let dy = -s; dy <= s; dy++) for (let dz = -s; dz <= s; dz++) {
          if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== s) continue;
          for (const j of cells.get(key(x + dx, y + dy, z + dz)) ?? []) {
            const d = dist2(Q, i, P, j);
            if (d < best) {
              best = d;
              bj = j;
            }
          }
        }
        if (best <= (s * cell) ** 2) break;
      }
      return bj;
    },
  };
}

function nearestIndices(Q: Float32Array, grid: ReturnType<typeof pointGrid>) {
  const out = new Uint32Array(Q.length / 3);
  for (let i = 0; i < out.length; i++) out[i] = grid.nearest(Q, i);
  return out;
}

/** Binary min-heap of (key, value) pairs. */
class MinHeap {
  private k: number[] = [];
  private v: number[] = [];
  get size() {
    return this.k.length;
  }
  push(key: number, value: number) {
    const { k, v } = this;
    let i = k.length;
    k.push(key);
    v.push(value);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= k[i]) break;
      [k[p], k[i]] = [k[i], k[p]];
      [v[p], v[i]] = [v[i], v[p]];
      i = p;
    }
  }
  pop(): [number, number] {
    const { k, v } = this;
    const top: [number, number] = [k[0], v[0]];
    const lk = k.pop()!;
    const lv = v.pop()!;
    if (k.length) {
      k[0] = lk;
      v[0] = lv;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < k.length && k[l] < k[m]) m = l;
        if (r < k.length && k[r] < k[m]) m = r;
        if (m === i) break;
        [k[m], k[i]] = [k[i], k[m]];
        [v[m], v[i]] = [v[i], v[m]];
        i = m;
      }
    }
    return top;
  }
}

/** HEAD probe: the dev server answers unknown paths with index.html, so a hit must not be HTML. */
export async function atlasAvailable(): Promise<boolean> {
  try {
    const res = await fetch(ATLAS_HEART_URL, { method: "HEAD" });
    const type = res.headers.get("content-type") ?? "";
    return res.ok && !type.includes("text/html");
  } catch {
    return false;
  }
}
