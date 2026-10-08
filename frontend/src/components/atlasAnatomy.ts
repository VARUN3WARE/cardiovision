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
import type { Vessel } from "./heartAnatomy";

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
  /** Ventricular wall, per vertex: territory weights for lad, lcx, rca (sum 1). */
  territory: Float32Array;
  /** Ventricular wall, per vertex: epicardial fat amount 0..1 (fat fills the coronary grooves). */
  fat: Float32Array;
  /** Camera direction (unit) facing each vessel's territory. */
  views: Record<Vessel, THREE.Vector3>;
  /** Label anchors on each vessel's main trunk, lifted off the surface (metres, centred). */
  labels: Record<Vessel, THREE.Vector3>;
}

const TARGETS: Vessel[] = ["lad", "lcx", "rca"];
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
    for (const name of ["position", "normal", "_branch", "_vdist"]) {
      const a = src.getAttribute(name);
      if (!a) continue;
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

  // Territories from the baked distance (0.1 mm units) to each scored coronary tree.
  const vd = wall.geometry.getAttribute("_vdist");
  const n = wall.geometry.getAttribute("position").count;
  const territory = new Float32Array(n * 3);
  const fat = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = [vd.getX(i) / 10, vd.getY(i) / 10, vd.getZ(i) / 10]; // mm to the LAD, LCX, RCA trees
    const dmin = Math.min(d[0], d[1], d[2]);
    const w = d.map((x) => Math.exp(-(x - dmin) / TERRITORY_TAU_MM));
    const s = w[0] + w[1] + w[2];
    territory.set([w[0] / s, w[1] / s, w[2] / s], i * 3);
    // Epicardial fat fills the grooves the main vessels run in, thinning out within a few mm.
    fat[i] = 1 - smoothstep(2, 7, vd.getW(i) / 10);
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
  const sums = TARGETS.map(() => new THREE.Vector3());
  const p = new THREE.Vector3();
  const nv = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    nv.fromBufferAttribute(nrm, i);
    if (nv.dot(p) <= 0) continue; // outer (epicardial) surface only
    TARGETS.forEach((_, k) => {
      if (territory[i * 3 + k] > 0.6) sums[k].add(p);
    });
  }
  return Object.fromEntries(TARGETS.map((t, k) => {
    const dir = sums[k].normalize();
    // Face the territory horizontally, from a gently elevated (or, for the inferior wall, low) viewpoint.
    dir.y = Math.min(0.3, Math.max(-0.35, dir.y * 0.5 + 0.1));
    return [t, dir.normalize()];
  })) as Record<Vessel, THREE.Vector3>;
}

/** Where to pin each vessel's label: on its proximal trunk, nudged outward from the heart. */
function labelAnchors(byKey: Record<string, AtlasPart>): Record<Vessel, THREE.Vector3> {
  const lm = byKey.lm?.geometry.boundingSphere?.center ?? new THREE.Vector3();
  const anchor = (key: Vessel, trunkFma: string | null, along: number) => {
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
  return {
    lad: anchor("lad", "FMA74912", 0.45),
    lcx: anchor("lcx", null, 0.5),
    rca: anchor("rca", "FMA3802", 0.35),
  };
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
