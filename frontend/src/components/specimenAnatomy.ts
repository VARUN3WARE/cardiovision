/**
 * Realistic heart model ("Realistic Human Heart" by neshallads, CC BY 4.0) and the coronary tree
 * fitted onto it.
 *
 * Coordinates are in the model's own frame after its glTF node transforms are baked in:
 * +Y superior, +Z anterior, +X the patient's left (anatomical anterior view), about 1.6 units tall.
 * The coronary control points were picked on the model's surface along its sculpted grooves, so
 * the risk-coloured tubes sit in the real LAD / LCX / RCA grooves rather than next to them.
 * At runtime each path is resampled, snapped onto the surface (closest point via a BVH), lifted
 * along the surface normal and smoothed.
 */
import * as THREE from "three";
import { MeshBVH, type HitPointInfo } from "three-mesh-bvh";
import { VESSEL_IDS, type Vessel } from "../anatomy/registry";

export const SPECIMEN_URL = "/models/heart.glb";
export const SPECIMEN_CREDIT = {
  title: "Realistic Human Heart",
  author: "neshallads",
  license: "CC BY 4.0",
  url: "https://sketchfab.com/3d-models/realistic-human-heart-3f8072336ce94d18b3d0d055a1ece089",
  licenseUrl: "https://creativecommons.org/licenses/by/4.0/",
};

/** World units per model unit when displayed (the schematic heart is ~3.5 units tall). */
export const SPECIMEN_SCALE = 2.2;

type P3 = [number, number, number];

export interface SpecimenVessel {
  vessel: Vessel | "lm";
  name: string;
  radius: number; // proximal radius, model units
  taper: number; // distal / proximal radius
  points: P3[]; // surface points, proximal -> distal
  label?: boolean;
  labelT?: number; // where along the vessel to anchor its label
}

// Shared branch points, so branches start exactly on their parent vessel.
const BIFURCATION: P3 = [0.0865, 0.1576, 0.3371]; // left main -> LAD + LCX
const RCA_GROOVE_START: P3 = [-0.2703, 0.0778, 0.3942];
const CRUX: P3 = [-0.1662, 0.0403, -0.3333];

export const SPECIMEN_ARTERIES: SpecimenVessel[] = [
  {
    vessel: "lm", name: "Left Main", radius: 0.017, taper: 0.95,
    points: [[0.0096, 0.2784, 0.3941], [0.0529, 0.2348, 0.394], BIFURCATION],
  },
  {
    vessel: "lad", name: "LAD", radius: 0.016, taper: 0.4, label: true, labelT: 0.35,
    points: [
      BIFURCATION, [0.1054, 0.0897, 0.3476], [0.1208, 0.0185, 0.3967], [0.1307, -0.052, 0.4222],
      [0.1282, -0.1377, 0.4206], [0.099, -0.213, 0.422], [0.0686, -0.2698, 0.4238], [0.0502, -0.332, 0.4196],
      [0.0375, -0.4106, 0.4015], [0.0192, -0.4966, 0.3664], [0.0099, -0.5905, 0.3117], [0.0148, -0.6788, 0.2446],
      [0.0871, -0.7313, 0.1951],
    ],
  },
  {
    vessel: "lad", name: "LAD · D1", radius: 0.011, taper: 0.45,
    points: [
      [0.1208, 0.0185, 0.3967], [0.15, -0.0196, 0.4005], [0.19, -0.0301, 0.3804], [0.2315, -0.0461, 0.3574],
      [0.2752, -0.0787, 0.3278], [0.3117, -0.1285, 0.2991], [0.3427, -0.1841, 0.2601], [0.3687, -0.235, 0.215],
    ],
  },
  {
    vessel: "lad", name: "LAD · D2", radius: 0.009, taper: 0.5,
    points: [[0.1282, -0.1377, 0.4206], [0.1414, -0.1633, 0.4068], [0.1565, -0.2095, 0.3893], [0.1771, -0.2459, 0.3749], [0.2022, -0.2775, 0.3601]],
  },
  {
    vessel: "lcx", name: "LCX", radius: 0.015, taper: 0.5, label: true, labelT: 0.4,
    points: [
      BIFURCATION, [0.19, 0.1, 0.33], [0.3034, -0.008, 0.2948], [0.3675, -0.0064, 0.2006], [0.3928, -0.0069, 0.1103],
      [0.4041, -0.016, 0.028], [0.4023, -0.0339, -0.0407], [0.3985, -0.0277, -0.1182], [0.3737, -0.0117, -0.2014],
      [0.2741, 0.0076, -0.2995], [0.2137, -0.0178, -0.3277], [0.1554, 0.0078, -0.3421], [0.0946, 0.0322, -0.3516],
    ],
  },
  {
    vessel: "lcx", name: "LCX · OM1", radius: 0.011, taper: 0.45,
    points: [
      [0.4023, -0.0339, -0.0407], [0.4031, -0.1128, -0.0346], [0.418, -0.1725, -0.0206], [0.426, -0.229, -0.003],
      [0.4239, -0.2894, 0.0133], [0.4118, -0.3641, 0.0258], [0.391, -0.4451, 0.0361],
    ],
  },
  {
    vessel: "lcx", name: "LCX · OM2", radius: 0.009, taper: 0.5,
    points: [
      [0.1554, 0.0078, -0.3421], [0.1027, -0.0612, -0.3605], [0.1152, -0.131, -0.3594], [0.1356, -0.2046, -0.3513],
      [0.1534, -0.282, -0.3344], [0.1656, -0.3492, -0.3134], [0.1881, -0.4143, -0.2856],
    ],
  },
  {
    vessel: "rca", name: "RCA", radius: 0.016, taper: 0.7, label: true, labelT: 0.18,
    points: [
      [-0.1853, 0.2773, 0.3373], [-0.2225, 0.1833, 0.3638], RCA_GROOVE_START, [-0.433, 0.0075, 0.3313],
      [-0.5083, -0.0386, 0.2642], [-0.5448, -0.072, 0.1934], [-0.5607, -0.0965, 0.1191], [-0.5636, -0.1119, 0.041],
      [-0.5615, -0.1212, -0.0399], [-0.5507, -0.1277, -0.1255], [-0.506, -0.1331, -0.2186], [-0.4214, -0.1489, -0.2787],
      [-0.4006, -0.093, -0.2949], [-0.373, -0.0447, -0.3082], [-0.3349, -0.0073, -0.3216], [-0.2819, 0.0179, -0.33],
      [-0.2239, 0.0299, -0.3352], CRUX,
    ],
  },
  {
    vessel: "rca", name: "RCA · RV branch", radius: 0.01, taper: 0.5,
    points: [
      RCA_GROOVE_START, [-0.2947, 0.0379, 0.3935], [-0.3036, -0.0092, 0.4024], [-0.3153, -0.0629, 0.404],
      [-0.3281, -0.1161, 0.4007], [-0.3404, -0.1693, 0.3937], [-0.3519, -0.2121, 0.3848],
    ],
  },
  {
    vessel: "rca", name: "RCA · PDA", radius: 0.012, taper: 0.45,
    points: [
      CRUX, [-0.1357, 0.0167, -0.3393], [-0.1437, -0.0496, -0.355], [-0.1536, -0.1088, -0.3604],
      [-0.1687, -0.1635, -0.355], [-0.1855, -0.2154, -0.3444], [-0.1992, -0.268, -0.3324], [-0.2083, -0.3255, -0.3164],
      [-0.213, -0.3846, -0.2955], [-0.2103, -0.4427, -0.2701], [-0.2034, -0.4997, -0.2381],
    ],
  },
];

/** Atrioventricular groove points (right, posterior and left), used to mask the ventricles. */
const AV_GROOVE: P3[] = [
  RCA_GROOVE_START, [-0.433, 0.0075, 0.3313], [-0.5083, -0.0386, 0.2642], [-0.5448, -0.072, 0.1934],
  [-0.5607, -0.0965, 0.1191], [-0.5636, -0.1119, 0.041], [-0.5615, -0.1212, -0.0399], [-0.5507, -0.1277, -0.1255],
  [-0.506, -0.1331, -0.2186], [-0.4214, -0.1489, -0.2787], [-0.3349, -0.0073, -0.3216], [-0.2239, 0.0299, -0.3352],
  CRUX, [-0.0334, 0.0783, -0.3491], [0.0946, 0.0322, -0.3516], [0.2137, -0.0178, -0.3277], [0.2741, 0.0076, -0.2995],
  [0.3737, -0.0117, -0.2014], [0.3985, -0.0277, -0.1182], [0.4023, -0.0339, -0.0407], [0.4041, -0.016, 0.028],
  [0.3928, -0.0069, 0.1103], [0.3675, -0.0064, 0.2006], [0.3034, -0.008, 0.2948], [0.1663, -0.0046, 0.3864],
];

/** Anatomy-pin positions on this model (ids match anatomyLabels.ts). */
export const SPECIMEN_PINS: Record<string, P3> = {
  aorta: [-0.1306, 0.4067, 0.3346],
  arch: [0.1077, 0.6467, 0.0663],
  pt: [0.2836, 0.367, 0.2348],
  svc: [-0.2883, 0.6385, 0.1986],
  raa: [-0.432, 0.1938, 0.2843],
  laa: [0.3705, 0.1667, 0.0864],
  rv: [-0.2371, -0.2794, 0.4268],
  lv: [0.3112, -0.3847, 0.2625],
  apex: [0.0871, -0.7313, 0.1951],
  ivg: [0.0556, -0.2442, 0.4366],
};

export interface PreparedSpecimen {
  geometry: THREE.BufferGeometry; // baked into the model frame, with a BVH as geometry.boundsTree
  material: THREE.Material;
  center: THREE.Vector3; // bounding-box centre (display pivot)
  vessels: { seg: SpecimenVessel; curve: THREE.CatmullRomCurve3; radiusAt: (t: number) => number; labelPos: THREE.Vector3 }[];
  /** Per vertex (stride TERRITORY_STRIDE): territory weight per registry artery, then a ventricle mask (0..1). */
  territory: Float32Array;
  /** Camera direction (unit, display frame) that faces each vessel's territory. */
  views: Record<Vessel, THREE.Vector3>;
  /** Anatomy-pin positions lifted just off the surface (so label occlusion doesn't hide them). */
  pins: Record<string, [number, number, number]>;
}

/** Bake the glTF mesh into one geometry in the model frame and fit the coronary tree onto it. */
export function prepareSpecimen(scene: THREE.Object3D): PreparedSpecimen {
  scene.updateMatrixWorld(true);
  let source: THREE.Mesh | null = null;
  scene.traverse((o) => {
    if (!source && (o as THREE.Mesh).isMesh) source = o as THREE.Mesh;
  });
  if (!source) throw new Error("heart model contains no mesh");
  const mesh = source as THREE.Mesh;
  const geometry = mesh.geometry.clone();
  geometry.applyMatrix4(mesh.matrixWorld);
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  const bvh = new MeshBVH(geometry);
  geometry.boundsTree = bvh;

  const material = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material).clone();
  if ((material as THREE.MeshStandardMaterial).isMeshStandardMaterial) {
    const m = material as THREE.MeshStandardMaterial;
    m.metalness = 0; // the asset leaves metallicFactor at its default of 1; tissue is not metal
    // The texture's roughness (0.13-0.23) plus its bumpy normal map glitters under bloom;
    // scale it up slightly so the surface stays wet-looking without sparkles.
    m.roughness = 1.45;
    m.envMapIntensity = 0.9;
  }

  const center = geometry.boundingBox!.getCenter(new THREE.Vector3());
  const vessels = SPECIMEN_ARTERIES.map((seg) => {
    const radiusAt = (t: number) => seg.radius * (1 + (seg.taper - 1) * Math.pow(t, 0.9));
    const curve = snapToSurface(bvh, geometry, seg.points, radiusAt);
    const at = curve.getPoint(seg.labelT ?? 0.3);
    const out = at.clone().sub(center).setY(0).normalize().multiplyScalar(0.07);
    return { seg, curve, radiusAt, labelPos: at.add(out) };
  });
  const territory = computeTerritories(geometry, vessels);
  const pins = Object.fromEntries(Object.entries(SPECIMEN_PINS).map(([id, p]) => {
    const point = new THREE.Vector3(...p);
    bvh.closestPointToPoint(point, _hit);
    const lifted = _hit.point.clone().addScaledVector(normalAt(geometry, _hit.faceIndex, _hit.point), 0.03);
    return [id, lifted.toArray() as [number, number, number]];
  }));
  return { geometry, material, center, vessels, territory, views: vesselViews(geometry, territory, center), pins };
}

const _hit: HitPointInfo = { point: new THREE.Vector3(), distance: 0, faceIndex: 0 };
const _tri = new THREE.Triangle();
const _bary = new THREE.Vector3();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();

/** Interpolated vertex normal at a point on face `faceIndex`. */
function normalAt(geometry: THREE.BufferGeometry, faceIndex: number, point: THREE.Vector3, out = new THREE.Vector3()) {
  const index = geometry.index!;
  const pos = geometry.getAttribute("position");
  const nrm = geometry.getAttribute("normal");
  const ia = index.getX(faceIndex * 3), ib = index.getX(faceIndex * 3 + 1), ic = index.getX(faceIndex * 3 + 2);
  _tri.set(_a.fromBufferAttribute(pos, ia), _b.fromBufferAttribute(pos, ib), _c.fromBufferAttribute(pos, ic));
  _tri.getBarycoord(point, _bary);
  out.set(0, 0, 0)
    .addScaledVector(_a.fromBufferAttribute(nrm, ia), _bary.x)
    .addScaledVector(_b.fromBufferAttribute(nrm, ib), _bary.y)
    .addScaledVector(_c.fromBufferAttribute(nrm, ic), _bary.z);
  return out.normalize();
}

/** Resample a control polyline, snap every sample onto the surface, lift it, and smooth. */
function snapToSurface(bvh: MeshBVH, geometry: THREE.BufferGeometry, points: P3[], radiusAt: (t: number) => number) {
  const control = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)), false, "centripetal");
  const n = Math.max(24, Math.ceil(control.getLength() / 0.006));
  const samples = control.getSpacedPoints(n);
  const normal = new THREE.Vector3();
  let snapped = samples.map((p, i) => {
    bvh.closestPointToPoint(p, _hit);
    normalAt(geometry, _hit.faceIndex, _hit.point, normal);
    // Sit ~60% proud of the surface: reads as a vessel lying in its groove.
    return _hit.point.clone().addScaledVector(normal, radiusAt(i / n) * 0.45);
  });
  // A few passes of Laplacian smoothing remove kinks from surface relief (ends stay fixed).
  for (let pass = 0; pass < 3; pass++) {
    snapped = snapped.map((p, i) =>
      i === 0 || i === snapped.length - 1 ? p : p.clone().multiplyScalar(2).add(snapped[i - 1]).add(snapped[i + 1]).multiplyScalar(0.25));
  }
  return new THREE.CatmullRomCurve3(snapped, false, "centripetal");
}

const TERRITORY_SIGMA = 0.07; // model units; softness of the boundaries between territories
const N = VESSEL_IDS.length;
/** Per vertex: one territory weight per registry artery, then the ventricle mask. */
export const TERRITORY_STRIDE = N + 1;

/**
 * Perfusion territories by proximity: each ventricular vertex belongs mostly to the nearest
 * artery tree (soft-min over squared distance), masked to the ventricles below the AV groove.
 */
function computeTerritories(geometry: THREE.BufferGeometry, vessels: PreparedSpecimen["vessels"]): Float32Array {
  const samples = Object.fromEntries(VESSEL_IDS.map((v) => [v, [] as THREE.Vector3[]])) as Record<Vessel, THREE.Vector3[]>;
  for (const v of vessels) {
    if (v.seg.vessel === "lm") continue;
    for (const p of v.curve.getSpacedPoints(Math.ceil(v.curve.getLength() / 0.02))) samples[v.seg.vessel].push(p);
  }
  const groove = grooveHeight();
  const pos = geometry.getAttribute("position");
  const out = new Float32Array(pos.count * TERRITORY_STRIDE);
  const p = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    const d2 = VESSEL_IDS.map((t) => samples[t].reduce((m, s) => Math.min(m, s.distanceToSquared(p)), Infinity));
    const dmin = Math.min(...d2);
    const w = d2.map((d) => Math.exp(-(d - dmin) / (TERRITORY_SIGMA * TERRITORY_SIGMA)));
    const sum = w.reduce((a, b) => a + b, 0);
    const gy = groove(p);
    const mask = smoothstep(gy + 0.035, gy - 0.025, p.y);
    out.set([...w.map((x) => x / sum), mask], i * TERRITORY_STRIDE);
  }
  return out;
}

/** Height of the AV groove as a function of azimuth around the heart's long axis (periodic). */
function grooveHeight() {
  const cx = AV_GROOVE.reduce((s, p) => s + p[0], 0) / AV_GROOVE.length;
  const cz = AV_GROOVE.reduce((s, p) => s + p[2], 0) / AV_GROOVE.length;
  const ring = AV_GROOVE.map((p) => ({ a: Math.atan2(p[2] - cz, p[0] - cx), y: p[1] })).sort((m, n) => m.a - n.a);
  return (p: THREE.Vector3) => {
    const a = Math.atan2(p.z - cz, p.x - cx);
    let hi = ring.findIndex((r) => r.a >= a);
    if (hi === -1) hi = 0;
    const lo = (hi - 1 + ring.length) % ring.length;
    const span = (ring[hi].a - ring[lo].a + 2 * Math.PI) % (2 * Math.PI) || 2 * Math.PI;
    const t = ((a - ring[lo].a + 2 * Math.PI) % (2 * Math.PI)) / span;
    return ring[lo].y + (ring[hi].y - ring[lo].y) * t;
  };
}

function smoothstep(e0: number, e1: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** Camera directions that face the centroid of each vessel's (ventricular) territory. */
function vesselViews(geometry: THREE.BufferGeometry, territory: Float32Array, center: THREE.Vector3) {
  const pos = geometry.getAttribute("position");
  const sums = VESSEL_IDS.map(() => new THREE.Vector3());
  const counts = VESSEL_IDS.map(() => 0);
  const p = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    if (territory[i * TERRITORY_STRIDE + N] < 0.5) continue;
    p.fromBufferAttribute(pos, i);
    VESSEL_IDS.forEach((_, k) => {
      if (territory[i * TERRITORY_STRIDE + k] > 0.6) {
        sums[k].add(p);
        counts[k]++;
      }
    });
  }
  return Object.fromEntries(VESSEL_IDS.map((t, k) => {
    const dir = sums[k].divideScalar(Math.max(1, counts[k])).sub(center);
    dir.normalize();
    // Face the territory horizontally, from a gently elevated viewpoint.
    dir.y = Math.min(0.3, Math.max(-0.2, dir.y * 0.3 + 0.15));
    return [t, dir.normalize()];
  })) as Record<Vessel, THREE.Vector3>;
}

/** HEAD probe for the model file. The dev server answers unknown paths with index.html, so a
 * successful response must not be HTML. */
export async function specimenAvailable(): Promise<boolean> {
  try {
    const res = await fetch(SPECIMEN_URL, { method: "HEAD" });
    const type = res.headers.get("content-type") ?? "";
    return res.ok && !type.includes("text/html");
  } catch {
    return false;
  }
}
