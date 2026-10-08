// Build the BodyParts3D atlas assets (public/models/bp3d_heart.glb, bp3d_thorax.glb) from the OBJ meshes
// fetched by scripts/bp3d/fetch_bp3d.py. Run with `npm run assets:bp3d`.
//
// Per element file: convert to the glTF frame, weld, drop small loose fragments, simplify (meshoptimizer),
// recompute smooth normals. Files are then merged into one mesh per part (one draw call each); a per-vertex
// `_BRANCH` index names the element (FMA concept + BodyParts3D file id) so a click resolves to the exact
// branch. The ventricular wall carries one `_D_<PART>` attribute per scored coronary tree (distance in 0.1 mm
// to its nearest vertex), from which the app derives illustrative perfusion territories, and `_D_GROOVE`
// (distance to the nearest main groove vessel, where epicardial fat lies).
//
// BodyParts3D, (c) The Database Center for Life Science, licensed under CC Attribution 4.0 International.

import { readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Document, NodeIO } from "@gltf-transform/core";
import { EXTMeshoptCompression, KHRMeshQuantization } from "@gltf-transform/extensions";
import { meshopt } from "@gltf-transform/functions";
import { MeshoptEncoder, MeshoptSimplifier } from "meshoptimizer";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const PARTS = JSON.parse(readFileSync(resolve(ROOT, "scripts/bp3d/parts.json"), "utf8"));
const CACHE = resolve(ROOT, "scripts/bp3d/cache");
const OUT = resolve(ROOT, "frontend/public/models");

// Fraction of triangles kept per layer, and the largest simplification error allowed (relative to the
// element's own size). Thin vessels keep more of their geometry than broad surfaces.
const LOD = {
  myocardium: { ratio: 0.6, error: 0.004 },
  atrium: { ratio: 0.42, error: 0.006 },
  cavity: { ratio: 1, error: 0 },
  artery: { ratio: 1, error: 0.003 },
  pulmonary: { ratio: 0.4, error: 0.006 },
  vein: { ratio: 1, error: 0.003 },
  coronary: { ratio: 0.62, error: 0.004 },
  cardiac_vein: { ratio: 0.4, error: 0.006 },
  skin: { ratio: 0.3, error: 0.004 },
  bone: { ratio: 0.17, error: 0.01 },
  cartilage: { ratio: 0.3, error: 0.01 },
  diaphragm: { ratio: 0.12, error: 0.008 },
  airway: { ratio: 0.3, error: 0.008 },
};
// Layers that are only used at build time (the LV/RV cavities) are not written to the GLB.
const BUILD_ONLY = new Set(["cavity"]);
// Skin is cropped to the chest (BodyParts3D z, mm above the soles) before simplifying.
const SKIN_Z = [1000, 1450];
const MIN_FRAGMENT = 0.02; // drop connected pieces with < 2% of an element's vertices
// Vessels that run in the atrioventricular and interventricular grooves, which epicardial fat fills:
// LAD trunk, circumflex, RCA trunk, posterior interventricular branch, left main, coronary sinus,
// great / middle cardiac veins and the anterior interventricular vein.
const GROOVE_FMA = new Set(["FMA74912", "FMA3895", "FMA3802", "FMA3840", "FMA3855", "FMA4706", "FMA4707", "FMA4713", "FMA66403"]);

/** OBJ (v / f only) -> positions in the glTF frame, metres: (x, z, -y) * 0.001. */
function readObj(fj) {
  const pos = [];
  const idx = [];
  for (const line of readFileSync(resolve(CACHE, `${fj}.obj`), "utf8").split("\n")) {
    if (line.startsWith("v ")) {
      const [, x, y, z] = line.trim().split(/\s+/).map(Number);
      pos.push(x * 0.001, z * 0.001, -y * 0.001);
    } else if (line.startsWith("f ")) {
      const v = line.trim().split(/\s+/).slice(1).map((s) => parseInt(s, 10) - 1);
      for (let k = 1; k + 1 < v.length; k++) idx.push(v[0], v[k], v[k + 1]);
    }
  }
  return { pos: new Float32Array(pos), idx: new Uint32Array(idx) };
}

/** Merge vertices at identical positions and drop degenerate triangles. */
function weld({ pos, idx }) {
  const map = new Map();
  const remap = new Uint32Array(pos.length / 3);
  const out = [];
  for (let i = 0; i < remap.length; i++) {
    const key = `${Math.round(pos[3 * i] * 1e6)},${Math.round(pos[3 * i + 1] * 1e6)},${Math.round(pos[3 * i + 2] * 1e6)}`;
    let j = map.get(key);
    if (j === undefined) {
      j = out.length / 3;
      map.set(key, j);
      out.push(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    }
    remap[i] = j;
  }
  const tri = [];
  for (let t = 0; t < idx.length; t += 3) {
    const a = remap[idx[t]], b = remap[idx[t + 1]], c = remap[idx[t + 2]];
    if (a !== b && b !== c && a !== c) tri.push(a, b, c);
  }
  return { pos: new Float32Array(out), idx: new Uint32Array(tri) };
}

/** Keep only triangles whose vertices all satisfy keep(vertexIndex). */
function filterTris(mesh, keep) {
  const tri = [];
  for (let t = 0; t < mesh.idx.length; t += 3) {
    if (keep(mesh.idx[t]) && keep(mesh.idx[t + 1]) && keep(mesh.idx[t + 2])) tri.push(mesh.idx[t], mesh.idx[t + 1], mesh.idx[t + 2]);
  }
  return compact({ pos: mesh.pos, idx: new Uint32Array(tri) });
}

/** Drop unreferenced vertices. */
function compact({ pos, idx }) {
  const remap = new Int32Array(pos.length / 3).fill(-1);
  const out = [];
  const tri = new Uint32Array(idx.length);
  for (let t = 0; t < idx.length; t++) {
    const v = idx[t];
    if (remap[v] < 0) {
      remap[v] = out.length / 3;
      out.push(pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]);
    }
    tri[t] = remap[v];
  }
  return { pos: new Float32Array(out), idx: tri };
}

/** Remove loose fragments (connected components with < MIN_FRAGMENT of the vertices). */
function dropFragments(mesh) {
  const n = mesh.pos.length / 3;
  const parent = new Int32Array(n).map((_, i) => i);
  const find = (a) => {
    while (parent[a] !== a) a = parent[a] = parent[parent[a]];
    return a;
  };
  for (let t = 0; t < mesh.idx.length; t += 3) {
    const a = find(mesh.idx[t]), b = find(mesh.idx[t + 1]), c = find(mesh.idx[t + 2]);
    parent[b] = a;
    parent[find(c)] = a;
  }
  const size = new Map();
  for (let i = 0; i < n; i++) size.set(find(i), (size.get(find(i)) ?? 0) + 1);
  const keep = new Set([...size].filter(([, s]) => s >= MIN_FRAGMENT * n).map(([r]) => r));
  return { mesh: filterTris(mesh, (v) => keep.has(find(v))), pieces: size.size, kept: keep.size };
}

function simplify(mesh, { ratio, error }) {
  if (ratio >= 1 && !error) return mesh;
  const target = Math.max(3, Math.floor((mesh.idx.length / 3) * ratio) * 3);
  const [idx] = MeshoptSimplifier.simplify(mesh.idx, mesh.pos, 3, target, error || 1, ["LockBorder"]);
  return compact({ pos: mesh.pos, idx });
}

/** Area-weighted smooth vertex normals. */
function normals({ pos, idx }) {
  const nrm = new Float32Array(pos.length);
  for (let t = 0; t < idx.length; t += 3) {
    const [a, b, c] = [idx[t] * 3, idx[t + 1] * 3, idx[t + 2] * 3];
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const v of [a, b, c]) {
      nrm[v] += nx;
      nrm[v + 1] += ny;
      nrm[v + 2] += nz;
    }
  }
  for (let i = 0; i < nrm.length; i += 3) {
    const l = Math.hypot(nrm[i], nrm[i + 1], nrm[i + 2]) || 1;
    nrm[i] /= l;
    nrm[i + 1] /= l;
    nrm[i + 2] /= l;
  }
  return nrm;
}

/** Nearest-vertex distance from every point of `pos` to the point cloud `cloud`, via a uniform grid. */
function nearestDistance(pos, cloud, cell = 0.004) {
  const grid = new Map();
  const key = (i, j, k) => `${i},${j},${k}`;
  for (let p = 0; p < cloud.length; p += 3) {
    const k = key(Math.floor(cloud[p] / cell), Math.floor(cloud[p + 1] / cell), Math.floor(cloud[p + 2] / cell));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(p);
  }
  const out = new Float32Array(pos.length / 3);
  for (let v = 0; v < out.length; v++) {
    const [x, y, z] = [pos[3 * v], pos[3 * v + 1], pos[3 * v + 2]];
    const [ci, cj, ck] = [Math.floor(x / cell), Math.floor(y / cell), Math.floor(z / cell)];
    let best = Infinity;
    // Grow the search shell until a hit is found and the shell is farther than the best distance.
    for (let r = 0; r < 40; r++) {
      for (let i = -r; i <= r; i++) for (let j = -r; j <= r; j++) for (let k = -r; k <= r; k++) {
        if (Math.max(Math.abs(i), Math.abs(j), Math.abs(k)) !== r) continue;
        for (const p of grid.get(key(ci + i, cj + j, ck + k)) ?? []) {
          const d = (cloud[p] - x) ** 2 + (cloud[p + 1] - y) ** 2 + (cloud[p + 2] - z) ** 2;
          if (d < best) best = d;
        }
      }
      if (best < Infinity && Math.sqrt(best) <= r * cell) break;
    }
    out[v] = Math.sqrt(best);
  }
  return out;
}

function processElement(fj, layer) {
  let mesh = weld(readObj(fj));
  if (layer === "skin") {
    mesh = filterTris(mesh, (v) => {
      const zmm = mesh.pos[3 * v + 1] * 1000;
      return zmm >= SKIN_Z[0] && zmm <= SKIN_Z[1];
    });
  }
  const before = mesh.idx.length / 3;
  const frag = dropFragments(mesh);
  mesh = simplify(frag.mesh, LOD[layer]);
  return { ...mesh, before, pieces: frag.pieces, kept: frag.kept };
}

async function buildGroup(name, group) {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const scene = doc.createScene(name);
  const built = {};
  const report = [];

  for (const part of group.parts) {
    const branches = [];
    const meshes = part.files.map((fj, b) => {
      const m = processElement(fj, part.layer);
      const el = PARTS.elements[fj];
      branches.push({ fj, fma: el.fma, name: el.name });
      return { ...m, branch: b };
    });
    // Merge the part's elements into one primitive with a per-vertex branch index.
    const nv = meshes.reduce((s, m) => s + m.pos.length / 3, 0);
    const pos = new Float32Array(nv * 3);
    const idx = new Uint32Array(meshes.reduce((s, m) => s + m.idx.length, 0));
    const branch = new Uint16Array(nv);
    let vo = 0;
    let io = 0;
    for (const m of meshes) {
      pos.set(m.pos, vo * 3);
      for (let t = 0; t < m.idx.length; t++) idx[io + t] = m.idx[t] + vo;
      branch.fill(m.branch, vo, vo + m.pos.length / 3);
      vo += m.pos.length / 3;
      io += m.idx.length;
    }
    built[part.key] = { part, pos, idx, branch, branches };
    report.push({
      part: part.key,
      files: part.files.length,
      trisIn: meshes.reduce((s, m) => s + m.before, 0),
      trisOut: idx.length / 3,
      fragmentsDropped: meshes.reduce((s, m) => s + m.pieces - m.kept, 0),
    });
  }

  // Ventricular wall: distance to each scored coronary tree (territory shading at runtime) and to the
  // groove vessels (epicardial fat). Scored trees are the coronary parts with a model output.
  if (built.ventricles) {
    const w = built.ventricles;
    const scored = group.parts.filter((p) => p.layer === "coronary" && p.vessel && p.vessel !== "lm").map((p) => p.key);
    const groove = [];
    for (const b of Object.values(built).filter((b) => ["coronary", "cardiac_vein"].includes(b.part.layer))) {
      for (let i = 0; i < b.branch.length; i++) {
        if (GROOVE_FMA.has(b.branches[b.branch[i]].fma)) groove.push(b.pos[3 * i], b.pos[3 * i + 1], b.pos[3 * i + 2]);
      }
    }
    const clouds = { ...Object.fromEntries(scored.map((k) => [k, built[k].pos])), groove: new Float32Array(groove) };
    w.dist = {};
    for (const [key, cloud] of Object.entries(clouds)) {
      const d = nearestDistance(w.pos, cloud);
      w.dist[key] = Uint16Array.from(d, (x) => Math.min(65535, Math.round(x * 1e4))); // 0.1 mm
    }
  }

  for (const b of Object.values(built)) {
    if (BUILD_ONLY.has(b.part.layer)) continue;
    const { part } = b;
    const prim = doc.createPrimitive()
      .setIndices(doc.createAccessor().setType("SCALAR").setArray(b.idx).setBuffer(buffer))
      .setAttribute("POSITION", doc.createAccessor().setType("VEC3").setArray(b.pos).setBuffer(buffer))
      .setAttribute("NORMAL", doc.createAccessor().setType("VEC3").setArray(normals(b)).setBuffer(buffer))
      .setAttribute("_BRANCH", doc.createAccessor().setType("SCALAR").setArray(b.branch).setBuffer(buffer))
      .setMaterial(doc.createMaterial(part.layer));
    for (const [key, arr] of Object.entries(b.dist ?? {})) {
      prim.setAttribute(`_D_${key.toUpperCase()}`, doc.createAccessor().setType("SCALAR").setArray(arr).setBuffer(buffer));
    }
    const mesh = doc.createMesh(part.key).addPrimitive(prim);
    const extras = { key: part.key, layer: part.layer, fma: part.fma, label: part.label, branches: b.branches };
    if (part.vessel) extras.vessel = part.vessel;
    scene.addChild(doc.createNode(part.key).setMesh(mesh).setExtras(extras));
  }
  doc.getRoot().getAsset().extras = {
    source: PARTS.source.archive,
    credit: PARTS.source.credit,
    licence: PARTS.source.licence,
    changes: "Converted to glTF frame (x, z, -y) in metres; welded; loose fragments removed; simplified with meshoptimizer; merged per part with a branch index; skin cropped to the chest; ventricular wall annotated with distance to each coronary tree.",
  };

  await MeshoptEncoder.ready;
  await MeshoptSimplifier.ready;
  doc.createExtension(EXTMeshoptCompression).setRequired(true).setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
  doc.createExtension(KHRMeshQuantization).setRequired(true);
  await doc.transform(meshopt({ encoder: MeshoptEncoder, level: "high", quantizationVolume: "scene" }));
  const file = resolve(OUT, group.output);
  await new NodeIO().registerExtensions([EXTMeshoptCompression, KHRMeshQuantization]).registerDependencies({ "meshopt.encoder": MeshoptEncoder }).write(file, doc);
  console.table(report);
  const tris = report.filter((r) => !BUILD_ONLY.has(group.parts.find((p) => p.key === r.part).layer)).reduce((s, r) => s + r.trisOut, 0);
  console.log(`${group.output}: ${tris} triangles, ${(statSync(file).size / 1e6).toFixed(2)} MB`);
}

await MeshoptSimplifier.ready;
const only = process.argv.slice(2);
for (const [name, group] of Object.entries(PARTS.groups)) {
  if (only.length && !only.includes(name)) continue;
  await buildGroup(name, group);
}
