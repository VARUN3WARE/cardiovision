/**
 * Procedural anatomical heart.
 *
 * The ventricular mass is an analytic surface P(u, v):
 *   u in [0, 1]  long axis, 0 = apex, 1 = base (atrioventricular groove)
 *   v (radians)  around the long axis, in the heart's local frame:
 *                0 = +X (patient's left, lateral LV wall)
 *                pi/2 = +Z (anterior / sternocostal surface)
 *                pi = -X (right border, RV)
 *                3pi/2 = -Z (posterior / diaphragmatic surface)
 *
 * Because the surface is analytic, coronary vessels are defined as paths in
 * (u, v) and projected exactly onto the epicardium, following the real grooves:
 * LAD in the anterior interventricular groove, LCX in the left AV groove,
 * RCA in the right AV groove continuing into the posterior descending artery.
 *
 * Realism cues taken from cadaveric specimens and anatomy atlases:
 *  - lobular yellow epicardial fat filling the AV and interventricular grooves,
 *    with the coronaries partly embedded in it;
 *  - cardiac veins running alongside the arteries (great / middle / small cardiac
 *    veins, coronary sinus);
 *  - tapering, slightly tortuous arteries with diagonal / marginal / RV branches;
 *  - oblique myocardial fibre texture under a wet (clearcoat + sheen) surface;
 *  - crinkled atrial appendages, aortic root sinuses and arch branches, cut vessel ends.
 */
import * as THREE from "three";

export type Vessel = "lad" | "lcx" | "rca";

const APEX_Y = -1.15;
const BASE_Y = 0.5;
const R_MAX = 0.95;
export const CAP = 0.14; // u-range of the domed base cap above the AV groove
const FAT_THICKNESS = 0.045;

// ---------------------------------------------------------------------------
// Noise (improved Perlin, seeded) — deterministic so arteries and fat line up.
// ---------------------------------------------------------------------------

const PERM = new Uint8Array(512);
{
  const p = Array.from({ length: 256 }, (_, i) => i);
  let s = 1337;
  for (let i = 255; i > 0; i--) {
    s = (s * 16807) % 2147483647;
    const j = s % (i + 1);
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
}
const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
function grad(h: number, x: number, y: number, z: number) {
  const k = h & 15;
  const u = k < 8 ? x : y;
  const v = k < 4 ? y : k === 12 || k === 14 ? x : z;
  return ((k & 1) ? -u : u) + ((k & 2) ? -v : v);
}

/** Perlin noise in roughly [-1, 1]. */
export function noise3(x: number, y: number, z: number) {
  const X = Math.floor(x) & 255, Y = Math.floor(y) & 255, Z = Math.floor(z) & 255;
  x -= Math.floor(x); y -= Math.floor(y); z -= Math.floor(z);
  const u = fade(x), v = fade(y), w = fade(z);
  const A = PERM[X] + Y, AA = PERM[A] + Z, AB = PERM[A + 1] + Z;
  const B = PERM[X + 1] + Y, BA = PERM[B] + Z, BB = PERM[B + 1] + Z;
  return lerp(
    lerp(lerp(grad(PERM[AA], x, y, z), grad(PERM[BA], x - 1, y, z), u),
      lerp(grad(PERM[AB], x, y - 1, z), grad(PERM[BB], x - 1, y - 1, z), u), v),
    lerp(lerp(grad(PERM[AA + 1], x, y, z - 1), grad(PERM[BA + 1], x - 1, y, z - 1), u),
      lerp(grad(PERM[AB + 1], x, y - 1, z - 1), grad(PERM[BB + 1], x - 1, y - 1, z - 1), u), v),
    w,
  );
}

export function fbm(x: number, y: number, z: number, octaves = 4) {
  let amp = 0.5, freq = 1, sum = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise3(x * freq, y * freq, z * freq);
    amp *= 0.5;
    freq *= 2.03;
  }
  return sum;
}

function wrapAngle(a: number) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

function gauss(d: number, w: number) {
  return Math.exp(-(d * d) / (2 * w * w));
}

function smoothstep(a: number, b: number, x: number) {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------
// Ventricular surface.
// ---------------------------------------------------------------------------

/** Point on the bare epicardial surface (no fat). u may exceed 1 (base cap). */
export function surfacePoint(u: number, v: number, out = new THREE.Vector3()) {
  const uc = Math.min(u, 1);
  // LV is a blunt cone; slightly fuller profile toward the base.
  let r = R_MAX * Math.pow(Math.sin(Math.min(1, uc * 1.2) * Math.PI / 2), 0.68);
  let y = APEX_Y + (BASE_Y - APEX_Y) * uc;
  if (u > 1) {
    const t = Math.min((u - 1) / CAP, 1);
    r *= Math.sqrt(Math.max(0, 1 - t * t));
    y += t * 0.16;
  }
  // Right ventricle wraps the LV anteriorly / to the right, ending in a sharper acute margin.
  const rvBand = gauss(wrapAngle(v - 2.25), 0.72);
  const rv = 0.17 * rvBand * Math.min(1, uc * 1.6);
  const acuteMargin = 0.05 * gauss(wrapAngle(v - 3.35), 0.18) * Math.min(1, uc * 2);
  // Interventricular grooves where LAD and PDA run.
  const groove = -0.035 * gauss(wrapAngle(v - 1.2), 0.11) - 0.035 * gauss(wrapAngle(v - 4.55), 0.13);
  // RV outflow tract (conus) bulges up toward the pulmonary trunk.
  const conus = 0.06 * gauss(wrapAngle(v - 1.75), 0.3) * smoothstep(0.75, 1.0, uc);
  r *= 1 + rv + acuteMargin + groove + conus;
  let z = r * Math.sin(v);
  if (z < 0) z *= 0.8; // flatter diaphragmatic surface
  let x = r * Math.cos(v);
  // Apex sits slightly to the LV side, not dead centre.
  x += 0.08 * (1 - uc) * (1 - uc);
  // Gentle large-scale asymmetry so it doesn't read as a lathe object.
  const s = 1 + 0.025 * fbm(x * 1.2 + 3.1, y * 1.2, z * 1.2, 2);
  out.set(x * s, y, z * s);
  return out;
}

const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();

/** Outward unit normal by finite differences. */
export function surfaceNormal(u: number, v: number, out = new THREE.Vector3()) {
  const e = 1e-3;
  const uu = Math.max(u, 0.02);
  surfacePoint(uu + e, v, _a).sub(surfacePoint(uu - e, v, _c));
  surfacePoint(uu, v + e, _b).sub(surfacePoint(uu, v - e, _c));
  out.crossVectors(_b, _a).normalize();
  // Guard orientation: normal must point away from the long axis.
  surfacePoint(uu, v, _c);
  _c.y = 0;
  if (out.dot(_c) < 0 && _c.lengthSq() > 1e-6) out.negate();
  return out;
}

/** Myocardial perfusion territory weights (LAD / LCX / RCA), summing to 1. */
export function territoryWeights(u: number, v: number): Record<Vessel, number> {
  const apex = Math.max(0, 0.3 - u) * 4; // apex is classically LAD territory
  const lad = gauss(wrapAngle(v - 1.25), 0.5) + apex;
  const lcx = gauss(wrapAngle(v + 0.15), 0.55);
  const rca = gauss(wrapAngle(v - 3.6), 0.85);
  const s = lad + lcx + rca + 1e-6;
  return { lad: lad / s, lcx: lcx / s, rca: rca / s };
}

// ---------------------------------------------------------------------------
// Coronary arteries and cardiac veins.
// ---------------------------------------------------------------------------

export interface VesselPath {
  /** "vein" paths are anatomical context only (not clickable, not risk-coloured). */
  vessel: Vessel | "lm" | "vein";
  name: string;
  radius: number;
  /** Distal radius as a fraction of the proximal radius. */
  taper?: number;
  /** Optional 3D lead-in points (e.g. from the aortic ostium) before the surface path. */
  lead?: THREE.Vector3[];
  /** Control points on the epicardium as [u, v]. */
  path: [number, number][];
  /** Anchor the vessel's risk label on this path (main trunks only). */
  label?: boolean;
  /** Tortuosity amplitude in (u, v) space. */
  wiggle?: number;
  /** Width of the epicardial fat bed laid along this vessel (0 = none). */
  fat?: number;
}

const LEFT_OSTIUM = new THREE.Vector3(0.02, 0.68, 0.26);
const RIGHT_OSTIUM = new THREE.Vector3(-0.32, 0.66, 0.26);

const AV = 0.985; // AV groove sits just below the base cap

export const ARTERIES: VesselPath[] = [
  {
    vessel: "lm", name: "Left Main", radius: 0.05, taper: 0.95,
    lead: [LEFT_OSTIUM, new THREE.Vector3(0.12, 0.64, 0.55)],
    path: [[1.02, 1.3]],
  },
  // LAD and branches
  {
    vessel: "lad", name: "LAD", radius: 0.046, taper: 0.35, label: true, wiggle: 0.012, fat: 0.13,
    path: [[1.02, 1.3], [0.88, 1.24], [0.7, 1.2], [0.5, 1.15], [0.3, 1.08], [0.14, 0.95], [0.04, 0.4], [0.06, -0.6]],
  },
  { vessel: "lad", name: "LAD · D1", radius: 0.03, taper: 0.35, wiggle: 0.02, fat: 0.035, path: [[0.82, 1.22], [0.68, 0.88], [0.52, 0.6], [0.36, 0.45]] },
  { vessel: "lad", name: "LAD · D2", radius: 0.026, taper: 0.35, wiggle: 0.02, fat: 0.025, path: [[0.58, 1.16], [0.46, 0.88], [0.32, 0.66], [0.22, 0.58]] },
  { vessel: "lad", name: "LAD · D3", radius: 0.018, taper: 0.4, wiggle: 0.02, path: [[0.36, 1.1], [0.27, 0.9], [0.18, 0.78]] },
  { vessel: "lad", name: "LAD · RV branch", radius: 0.015, taper: 0.4, wiggle: 0.02, path: [[0.7, 1.21], [0.62, 1.45], [0.55, 1.7]] },
  // LCX and branches
  {
    vessel: "lcx", name: "LCX", radius: 0.042, taper: 0.45, label: true, wiggle: 0.008, fat: 0.1,
    path: [[1.02, 1.3], [AV, 1.0], [AV, 0.5], [AV, 0.0], [AV, -0.5], [AV, -0.95]],
  },
  { vessel: "lcx", name: "LCX · OM1", radius: 0.03, taper: 0.35, wiggle: 0.02, fat: 0.05, path: [[AV, 0.28], [0.8, 0.18], [0.6, 0.12], [0.4, 0.15], [0.28, 0.2]] },
  { vessel: "lcx", name: "LCX · OM2", radius: 0.025, taper: 0.35, wiggle: 0.02, fat: 0.04, path: [[AV, -0.42], [0.78, -0.5], [0.55, -0.48], [0.4, -0.4]] },
  { vessel: "lcx", name: "LCX · OM3", radius: 0.018, taper: 0.4, wiggle: 0.02, path: [[AV, -0.85], [0.8, -0.95], [0.62, -0.95]] },
  // RCA and branches
  {
    vessel: "rca", name: "RCA", radius: 0.046, taper: 0.6, label: true, wiggle: 0.008, fat: 0.12,
    lead: [RIGHT_OSTIUM, new THREE.Vector3(-0.45, 0.62, 0.64)],
    path: [[AV, 2.1], [AV, 2.6], [AV, 3.15], [AV, 3.7], [AV, 4.2], [AV, 4.55]],
  },
  { vessel: "rca", name: "RCA · Conus branch", radius: 0.014, taper: 0.5, wiggle: 0.02, path: [[AV, 2.15], [0.95, 1.95], [0.92, 1.75]] },
  { vessel: "rca", name: "RCA · RV branch", radius: 0.018, taper: 0.4, wiggle: 0.025, path: [[AV, 2.55], [0.8, 2.45], [0.6, 2.3], [0.45, 2.2]] },
  { vessel: "rca", name: "RCA · Acute marginal", radius: 0.026, taper: 0.35, wiggle: 0.02, fat: 0.05, path: [[AV, 3.0], [0.78, 3.05], [0.55, 3.12], [0.32, 3.22]] },
  { vessel: "rca", name: "RCA · PDA", radius: 0.032, taper: 0.35, wiggle: 0.015, fat: 0.1, path: [[AV, 4.55], [0.8, 4.55], [0.55, 4.58], [0.3, 4.6], [0.12, 4.72]] },
  { vessel: "rca", name: "RCA · Posterolateral", radius: 0.022, taper: 0.4, wiggle: 0.02, path: [[AV, 4.55], [0.96, 4.9], [0.85, 5.15], [0.7, 5.3]] },
];

export const VEINS: VesselPath[] = [
  // Great cardiac vein: up the anterior IV groove beside the LAD, then round the left AV groove.
  {
    vessel: "vein", name: "Great cardiac vein", radius: 0.018, taper: 2.2, wiggle: 0.01,
    path: [[0.15, 1.08], [0.35, 1.2], [0.6, 1.28], [0.85, 1.33], [0.975, 1.18], [0.975, 0.6], [0.975, 0.0], [0.975, -0.7], [0.975, -1.2]],
  },
  // Coronary sinus in the posterior AV groove, draining into the right atrium.
  { vessel: "vein", name: "Coronary sinus", radius: 0.04, taper: 1.3, path: [[0.975, -1.2], [0.975, -1.6], [0.98, -2.0], [0.99, -2.4]] },
  {
    vessel: "vein", name: "Middle cardiac vein", radius: 0.014, taper: 2.0, wiggle: 0.012,
    path: [[0.1, 4.45], [0.35, 4.45], [0.6, 4.47], [0.85, 4.48], [0.975, 4.45]],
  },
  { vessel: "vein", name: "Small cardiac vein", radius: 0.012, taper: 1.4, path: [[0.97, 3.2], [0.968, 3.6], [0.97, 3.95]] },
  { vessel: "vein", name: "Left marginal vein", radius: 0.01, taper: 2.0, wiggle: 0.02, path: [[0.35, 0.3], [0.6, 0.25], [0.8, 0.32], [0.975, 0.38]] },
  { vessel: "vein", name: "Anterior cardiac vein", radius: 0.01, taper: 1.4, wiggle: 0.02, path: [[0.55, 2.55], [0.75, 2.65], [0.95, 2.75]] },
];

// ---------------------------------------------------------------------------
// Epicardial fat.
// ---------------------------------------------------------------------------

interface FatSample { x: number; y: number; z: number; w2: number }
let fatSamples: FatSample[] | null = null;

function getFatSamples() {
  if (fatSamples) return fatSamples;
  const samples: FatSample[] = [];
  const add = (p: THREE.Vector3, w: number) => samples.push({ x: p.x, y: p.y, z: p.z, w2: w * w });
  // Continuous fat ring in the AV groove.
  for (let i = 0; i < 160; i++) add(surfacePoint(0.99, (i / 160) * Math.PI * 2), 0.17);
  for (const seg of ARTERIES) {
    if (!seg.fat || seg.path.length < 2) continue;
    const param = new THREE.CatmullRomCurve3(seg.path.map(([u, v]) => new THREE.Vector3(u, v, 0)));
    for (const s of param.getPoints(40)) add(surfacePoint(s.x, s.y), seg.fat);
  }
  fatSamples = samples;
  return samples;
}

/** Fat coverage in [0, 1] at a point of the bare surface (u, v given for regional terms). */
export function fatMask(p: THREE.Vector3, u: number, v: number) {
  let best = 0;
  for (const s of getFatSamples()) {
    const dx = p.x - s.x, dy = p.y - s.y, dz = p.z - s.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > 9 * s.w2) continue;
    const g = Math.exp(-d2 / s.w2);
    if (g > best) best = g;
  }
  // Patchy fat over the RV free wall and the base; very little over the apex.
  best += 0.28 * gauss(wrapAngle(v - 2.5), 0.45) * smoothstep(0.35, 0.8, u);
  best += 0.5 * smoothstep(0.98, 1.06, u);
  const edge = 0.28 * fbm(p.x * 3.5 + 11, p.y * 3.5, p.z * 3.5, 3);
  return smoothstep(0.32, 0.62, best + edge) * smoothstep(0.08, 0.25, u);
}

/** Height of the fat layer (lobular) at a surface point. */
function fatHeight(p: THREE.Vector3, mask: number, lobular = true) {
  if (mask <= 0) return 0;
  const lobules = lobular ? 0.65 + 0.55 * Math.abs(noise3(p.x * 14, p.y * 14, p.z * 14)) : 0.9;
  return FAT_THICKNESS * mask * lobules;
}

export function buildMyocardium(segU = 170, segV = 240) {
  const geo = new THREE.BufferGeometry();
  const count = (segU + 1) * (segV + 1);
  const pos = new Float32Array(count * 3);
  const uvs = new Float32Array(count * 2);
  const params = new Float32Array(count * 2);
  const fat = new Float32Array(count);
  const shade = new Float32Array(count);
  const idx: number[] = [];
  const uMax = 1 + CAP;
  const p = new THREE.Vector3();
  const n = new THREE.Vector3();
  let k = 0;
  for (let i = 0; i <= segU; i++) {
    const u = (i / segU) * uMax;
    for (let j = 0; j <= segV; j++) {
      const v = (j / segV) * Math.PI * 2;
      surfacePoint(u, v, p);
      surfaceNormal(u, v, n);
      const mask = fatMask(p, u, v);
      // Fine muscle relief everywhere, lobular fat on top where present.
      const relief = 0.004 * noise3(p.x * 9, p.y * 9, p.z * 9);
      p.addScaledVector(n, fatHeight(p, mask) + relief);
      pos[k * 3] = p.x; pos[k * 3 + 1] = p.y; pos[k * 3 + 2] = p.z;
      uvs[k * 2] = v / (Math.PI * 2); uvs[k * 2 + 1] = u / uMax; // texture space (normal map)
      params[k * 2] = u; params[k * 2 + 1] = v; // anatomical space (territories)
      fat[k] = mask;
      shade[k] = 0.5 + 0.5 * fbm(p.x * 2.5, p.y * 2.5 + 7, p.z * 2.5, 3);
      k++;
    }
  }
  const row = segV + 1;
  for (let i = 0; i < segU; i++) {
    for (let j = 0; j < segV; j++) {
      const a = i * row + j;
      const b = a + row;
      idx.push(a, a + 1, b, b, a + 1, b + 1);
    }
  }
  geo.setIndex(idx);
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geo.setAttribute("param", new THREE.BufferAttribute(params, 2));
  geo.setAttribute("fat", new THREE.BufferAttribute(fat, 1));
  geo.setAttribute("shade", new THREE.BufferAttribute(shade, 1));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3).fill(1), 3));
  geo.computeVertexNormals();
  // Weld normals across the v = 0 / 2pi seam so it doesn't show as a line.
  const nrm = geo.getAttribute("normal") as THREE.BufferAttribute;
  for (let i = 0; i <= segU; i++) {
    const a = i * row, b = a + segV;
    const nx = nrm.getX(a) + nrm.getX(b), ny = nrm.getY(a) + nrm.getY(b), nz = nrm.getZ(a) + nrm.getZ(b);
    const len = Math.hypot(nx, ny, nz) || 1;
    nrm.setXYZ(a, nx / len, ny / len, nz / len);
    nrm.setXYZ(b, nx / len, ny / len, nz / len);
  }
  return geo;
}

/**
 * Sample a vessel onto the epicardium. Arteries sit partly embedded in the fat bed;
 * veins sit a little deeper.
 */
export function vesselCurve(seg: VesselPath): THREE.CatmullRomCurve3 {
  const pts: THREE.Vector3[] = seg.lead ? seg.lead.map((p) => p.clone()) : [];
  const n = new THREE.Vector3();
  const embed = seg.vessel === "vein" ? 0.45 : 0.62;
  const place = (u: number, v: number) => {
    const p = surfacePoint(u, v);
    const lift = fatHeight(p, fatMask(p, u, v), false) * 0.8 + seg.radius * embed;
    return p.add(surfaceNormal(u, v, n).multiplyScalar(lift));
  };
  if (seg.path.length === 1) {
    pts.push(place(...seg.path[0]));
  } else {
    const param = new THREE.CatmullRomCurve3(seg.path.map(([u, v]) => new THREE.Vector3(u, v, 0)));
    const count = Math.max(32, seg.path.length * 14);
    const samples = param.getSpacedPoints(count);
    const amp = seg.wiggle ?? 0;
    const phase = seg.name.length * 1.7;
    // Roughly two gentle bends per unit of path length, never a zig-zag on short branches.
    const bends = Math.max(1.2, param.getLength() * 2.2);
    samples.forEach((s, i) => {
      const t = i / count;
      if (amp) {
        // Tortuosity: smooth wandering perpendicular to the path in (u, v), growing distally.
        const tan = param.getTangentAt(t);
        const w = amp * 1.8 * noise3(t * bends + phase, phase * 0.37, 0.5) * (0.3 + 0.7 * t);
        s.x += -tan.y * w;
        s.y += tan.x * w;
      }
      pts.push(place(s.x, s.y));
    });
  }
  return new THREE.CatmullRomCurve3(pts, false, "centripetal");
}

/** Tube whose radius follows radiusAt(t) — used for tapering arteries and the aortic root. */
export function taperedTube(
  curve: THREE.Curve<THREE.Vector3>,
  radiusAt: (t: number) => number,
  tubular = 160,
  radial = 16,
) {
  const frames = curve.computeFrenetFrames(tubular, false);
  const pos: number[] = [];
  const nrm: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  const P = new THREE.Vector3();
  const N = new THREE.Vector3();
  for (let i = 0; i <= tubular; i++) {
    const t = i / tubular;
    curve.getPointAt(t, P);
    const r = radiusAt(t);
    for (let j = 0; j <= radial; j++) {
      const a = (j / radial) * Math.PI * 2;
      N.copy(frames.normals[i]).multiplyScalar(-Math.cos(a))
        .addScaledVector(frames.binormals[i], Math.sin(a)).normalize();
      pos.push(P.x + r * N.x, P.y + r * N.y, P.z + r * N.z);
      nrm.push(N.x, N.y, N.z);
      uv.push(t * 4, j / radial);
    }
  }
  for (let i = 0; i < tubular; i++) {
    for (let j = 0; j < radial; j++) {
      const a = i * (radial + 1) + j;
      const b = a + radial + 1;
      idx.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setIndex(idx);
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("normal", new THREE.Float32BufferAttribute(nrm, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  return geo;
}

export function vesselRadius(seg: VesselPath) {
  const taper = seg.taper ?? 0.5;
  return (t: number) => seg.radius * (1 + (taper - 1) * Math.pow(t, 0.9));
}

// ---------------------------------------------------------------------------
// Great vessels and atria, in the same local frame.
// ---------------------------------------------------------------------------

type P3 = readonly [number, number, number];
export interface GreatVessel {
  points: readonly P3[];
  radius: number;
  /** Radius multiplier along t (e.g. aortic root sinuses). */
  profile?: (t: number) => number;
  deoxygenated?: boolean;
}

export const GREAT_VESSELS: Record<string, GreatVessel> = {
  aorta: {
    radius: 0.19,
    profile: (t) => 1 + 0.22 * Math.exp(-(((t - 0.03) / 0.05) ** 2)) - 0.12 * smoothstep(0.5, 1, t),
    points: [
      [-0.15, 0.55, 0.1], [-0.25, 1.1, 0.15], [-0.28, 1.6, 0.1],
      [-0.05, 1.95, -0.05], [0.35, 1.9, -0.35], [0.55, 1.5, -0.55],
      [0.58, 0.9, -0.62], [0.58, 0.2, -0.66],
    ],
  },
  brachiocephalic: { radius: 0.075, points: [[-0.2, 1.8, 0.05], [-0.3, 2.15, 0.08], [-0.38, 2.45, 0.1]] },
  leftCarotid: { radius: 0.05, points: [[0.04, 1.92, -0.12], [0.06, 2.2, -0.14], [0.07, 2.5, -0.15]] },
  leftSubclavian: { radius: 0.06, points: [[0.25, 1.88, -0.27], [0.36, 2.15, -0.32], [0.48, 2.42, -0.36]] },
  pulmonary: {
    radius: 0.16, deoxygenated: true,
    profile: (t) => 1.15 - 0.15 * t,
    points: [[-0.12, 0.55, 0.62], [-0.02, 1.0, 0.55], [0.15, 1.35, 0.32], [0.3, 1.5, 0.05]],
  },
  leftPA: { radius: 0.11, deoxygenated: true, points: [[0.28, 1.5, 0.08], [0.6, 1.5, -0.1], [0.9, 1.4, -0.25]] },
  rightPA: { radius: 0.11, deoxygenated: true, points: [[0.28, 1.5, 0.05], [0.0, 1.45, -0.28], [-0.45, 1.38, -0.4], [-0.85, 1.3, -0.45]] },
  svc: { radius: 0.13, deoxygenated: true, points: [[-0.78, 0.9, -0.05], [-0.8, 1.4, -0.1], [-0.8, 1.85, -0.12]] },
  // Short cut stumps, as on a dissected specimen.
  ivc: { radius: 0.13, deoxygenated: true, points: [[-0.7, 0.55, -0.15], [-0.72, 0.32, -0.3], [-0.73, 0.12, -0.38]] },
  // Four pulmonary veins (oxygenated) draining into the posterior left atrium.
  rspv: { radius: 0.065, points: [[-0.1, 0.86, -0.6], [-0.28, 0.92, -0.74], [-0.42, 0.95, -0.82]] },
  ripv: { radius: 0.065, points: [[-0.1, 0.64, -0.6], [-0.28, 0.6, -0.74], [-0.42, 0.57, -0.82]] },
  lspv: { radius: 0.065, points: [[0.52, 0.86, -0.6], [0.68, 0.92, -0.74], [0.82, 0.95, -0.82]] },
  lipv: { radius: 0.065, points: [[0.52, 0.64, -0.6], [0.68, 0.6, -0.74], [0.82, 0.57, -0.82]] },
};

export interface Chamber {
  name: string;
  center: P3;
  radii: P3;
  rotation?: P3;
  /** Surface crinkle amplitude (atrial appendages are wrinkled). */
  crinkle: number;
}

export const ATRIA: Chamber[] = [
  { name: "LA", center: [0.2, 0.72, -0.42], radii: [0.6, 0.3, 0.36], crinkle: 0.015 },
  { name: "RA", center: [-0.68, 0.66, 0.0], radii: [0.4, 0.38, 0.45], crinkle: 0.02 },
  // Right auricle: a broad triangular flap overlapping the aortic root.
  { name: "RAA", center: [-0.45, 0.8, 0.42], radii: [0.26, 0.13, 0.2], rotation: [0.3, 0.5, -0.3], crinkle: 0.22 },
  // Left auricle: a narrow, finger-like, crenated appendage curling round the pulmonary trunk.
  { name: "LAA", center: [0.62, 0.74, 0.28], radii: [0.24, 0.1, 0.15], rotation: [0.2, -0.7, -0.35], crinkle: 0.3 },
];

/** Unit sphere with fbm displacement, to be scaled by the chamber radii. */
export function buildChamber(crinkle: number, seed: number) {
  const geo = new THREE.SphereGeometry(1, 72, 54);
  const pos = geo.getAttribute("position") as THREE.BufferAttribute;
  const p = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    const big = 0.06 * fbm(p.x * 1.5 + seed, p.y * 1.5, p.z * 1.5, 2);
    // Crenations: ridged noise gives the scalloped edge of the appendages.
    const ridged = crinkle * (0.5 - Math.abs(noise3(p.x * 4 + seed, p.y * 4, p.z * 4)));
    p.multiplyScalar(1 + big + ridged);
    pos.setXYZ(i, p.x, p.y, p.z);
  }
  geo.computeVertexNormals();
  return geo;
}

// ---------------------------------------------------------------------------
// Surface detail textures (generated once, no image downloads).
// ---------------------------------------------------------------------------

function heightToNormalMap(size: number, heightAt: (x: number, y: number) => number, strength: number, wrapY: boolean) {
  const h = new Float32Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) h[y * size + x] = heightAt(x / size, y / size);
  const data = new Uint8Array(size * size * 4);
  const row = (y: number) => (wrapY ? (y + size) % size : Math.min(size - 1, Math.max(0, y)));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const l = h[y * size + ((x - 1 + size) % size)], r = h[y * size + ((x + 1) % size)];
      const d = h[row(y - 1) * size + x], u = h[row(y + 1) * size + x];
      const nx = (l - r) * strength, ny = (d - u) * strength;
      const len = Math.hypot(nx, ny, 1);
      const o = (y * size + x) * 4;
      data[o] = ((nx / len) * 0.5 + 0.5) * 255;
      data[o + 1] = ((ny / len) * 0.5 + 0.5) * 255;
      data[o + 2] = ((1 / len) * 0.5 + 0.5) * 255;
      data[o + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = wrapY ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Normal map for the myocardium: oblique, slightly wavy muscle fibres plus fine grain.
 * x maps around the heart (v) and is periodic, so there is no texture seam.
 */
export function makeMuscleNormalMap(size = 512) {
  return heightToNormalMap(size, (fx, fy) => {
    const a = fx * Math.PI * 2;
    const warp = fbm(Math.cos(a) * 3, Math.sin(a) * 3, fy * 8, 3); // periodic: sampled on a cylinder
    // Fine oblique fibres whose strength fades in and out, so they never read as stripes.
    const fibres = Math.sin(2 * Math.PI * (64 * fx + 80 * fy) + 9 * warp);
    const fibreMask = 0.5 + 0.5 * noise3(Math.cos(a) * 5 + 3, Math.sin(a) * 5, fy * 14);
    const grain = fbm(Math.cos(a) * 14, Math.sin(a) * 14, fy * 40, 3);
    return 0.25 * fibres * fibreMask + 0.75 * grain;
  }, 1.2, false);
}

/** Subtle tileable bump for vessels and atria (wet tissue, not plastic). */
export function makeTissueNormalMap(size = 256) {
  return heightToNormalMap(size, (fx, fy) => {
    const a = fx * Math.PI * 2, b = fy * Math.PI * 2; // torus sampling: tiles in both axes
    return noise3(Math.cos(a) * 3, Math.sin(a) * 3 + Math.cos(b) * 3, Math.sin(b) * 3) +
      0.5 * noise3(Math.cos(a) * 9 + 5, Math.sin(a) * 9, Math.cos(b) * 9 + Math.sin(b) * 9);
  }, 1.5, true);
}

// ---------------------------------------------------------------------------
// Colours.
// ---------------------------------------------------------------------------

const GREEN = new THREE.Color("#22c55e");
const AMBER = new THREE.Color("#f59e0b"); // amber, not yellow, so mid-risk vessels don't vanish into fat
const RED = new THREE.Color("#ef2b2b");
export const NEUTRAL_ARTERY = new THREE.Color("#e3a196");
export const VEIN_COLOR = new THREE.Color("#4b3f73");
export const MUSCLE_DARK = new THREE.Color("#5a1e1b");
export const MUSCLE_LIGHT = new THREE.Color("#8c3a32");
export const FAT_DARK = new THREE.Color("#c99a55");
export const FAT_LIGHT = new THREE.Color("#efd7a2");

/** Probability [0,1] -> green / amber / red. */
export function riskColor(p: number, out = new THREE.Color()) {
  return p < 0.5 ? out.copy(GREEN).lerp(AMBER, p * 2) : out.copy(AMBER).lerp(RED, (p - 0.5) * 2);
}
