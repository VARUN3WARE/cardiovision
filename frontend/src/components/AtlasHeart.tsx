import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { Html, useGLTF } from "@react-three/drei";
import {
  Suspense, createContext, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState,
  type MutableRefObject, type RefObject,
} from "react";
import * as THREE from "three";
import { MeshBVH, acceleratedRaycast } from "three-mesh-bvh";
import { VesselLabel, useVesselMaterial, type ArteryProbs } from "./ArteryTube";
import {
  ATLAS_HEART_URL, ATLAS_SCALE, ATLAS_THORAX_URL, prepareAtlasHeart, prepareLabAnatomy, prepareLabAnatomyWhenIdle, prepareThorax,
  type AtlasBranch, type AtlasLabAnatomy, type AtlasPart, type LabMarker, type PreparedAtlasHeart,
} from "./atlasAnatomy";
import { FLOW_SPACING_MM, LAB, patchMaterial } from "./atlasMaterials";
import { NEUTRAL_ARTERY, fbm, riskColor } from "./heartAnatomy";
import { RING_SIZE, TICK, heartClock, readTick, ventricularContraction, type BeatTiming, type TwinParams } from "../twin/twin";
import { ARTERY_PHYSIOLOGY, LEFT_MAIN_DSVR, phasicMean, type LabState } from "../twin/coronaryPhysiology";
import { useHeartMotion } from "../twin/useHeartMotion";
import { whenIdle } from "../lab/idle";
import { UNMODELLED, VESSEL, VESSELS, VESSEL_IDS, type Vessel } from "../anatomy/registry";
import { FlowAnchor, supportHull } from "../reasoning/anchors";

/** Which chest layers to draw around the heart. */
export interface ChestLayers {
  skin: boolean;
  skeleton: boolean;
  organs: boolean; // diaphragm and airway
}

const LAYER_GROUP: Record<string, keyof ChestLayers> = {
  skin: "skin", bone: "skeleton", cartilage: "skeleton", diaphragm: "organs", airway: "organs",
};

export interface AtlasPick {
  vessel: Vessel | "lm";
  part: string; // e.g. "Right coronary artery (RCA)"
  partFma: string;
  branch: AtlasBranch;
}

/** Ischaemia lab view: the sandbox narrowing on one artery and the physiology it produces. */
export interface AtlasLab {
  vessel: Vessel;
  state: LabState;
}

/** The part of the canvas the lab leaves free for the heart (px from the canvas's top-left): the marker's label stays in it. */
export interface LabFrame {
  left: number;
  right: number;
  top: number;
  /** How much the view is scaled down to fit there (the arteries' name tags shrink with it, to a point). */
  scale: number;
}

interface Props {
  probs: ArteryProbs | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  mode: "arteries" | "territories";
  xray: boolean;
  chest: ChestLayers | null;
  twin: TwinParams;
  timing: BeatTiming;
  /** Ischaemia lab (teaching sandbox): replaces the risk colours with flow pulses, the marker and the ischaemic territory. */
  lab?: AtlasLab | null;
  /** Camera direction per artery from which its sandbox marker is in sight (reported once the lab anatomy exists). */
  onLabViews?: (views: Record<Vessel, THREE.Vector3>) => void;
  /** Where the sandbox marker's label may go (it flips below the ring near the top, and shifts sideways). */
  labFrame?: LabFrame | null;
  /** The render pipeline (HD effects on or off): the lab's shaders are compiled ahead for each one used. */
  pipeline?: string;
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
  onPick: (pick: AtlasPick | null) => void;
  onReady: (views: Record<Vessel, THREE.Vector3>) => void;
}

const N = VESSEL_IDS.length;
// Lab: visual pulse speed (mm/s) at the reference demand; it scales with flow (the demand factor).
const FLOW_SPEED_MM_S = 16;
const LAB_ARTERY = new THREE.Color("#a5372e");
const LAB_MARKER_COLOR = "#f59e0b";
// Per-vertex lab attributes on the ventricular structures (all start at 0: no effect).
const LAB_ATTRIBUTES = ["aLabW", "aOstium", "aRestrict", "aPinch"] as const;
const label = (v: Vessel) => VESSEL[v].label;

const MUSCLE_DEEP = new THREE.Color("#5c1d1a");
const MUSCLE = new THREE.Color("#8e3a31");
const FAT_DEEP = new THREE.Color("#c4954f");
const FAT = new THREE.Color("#ecd39a");
// In X-ray mode tissue fades to a ghost that keeps its silhouette (no effect while opaque).
const XRAY_FRESNEL = { min: 0.12, power: 2.2 };

interface Tip {
  text: string;
  sub?: string;
  point: THREE.Vector3; // in the atlas group's local frame
}

/** Hover read-out: each heart structure names itself under the pointer. */
const TipContext = createContext<{ show: (e: ThreeEvent<PointerEvent>, text: string, sub?: string) => void; hide: () => void }>({
  show: () => {},
  hide: () => {},
});

function useTip(name: (e: ThreeEvent<PointerEvent>) => { text: string; sub?: string } | null) {
  const tip = useContext(TipContext);
  return {
    onPointerMove: (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();
      const n = name(e);
      if (n) tip.show(e, n.text, n.sub);
      else tip.hide();
    },
    onPointerOut: () => tip.hide(),
  };
}

/** Renders the hover read-out; owns its state so pointer moves don't re-render the heart. */
function HoverTip({ setter }: { setter: MutableRefObject<(t: Tip | null) => void> }) {
  const [tip, setTip] = useState<Tip | null>(null);
  useLayoutEffect(() => {
    setter.current = setTip;
  }, [setter]);
  if (!tip) return null;
  return (
    <Html position={tip.point} zIndexRange={[20, 10]} style={{ pointerEvents: "none" }}>
      <div className="ml-3 -mt-3 w-max max-w-[220px] rounded bg-slate-950/90 px-2 py-1 text-[11px] leading-tight text-white shadow">
        <div className="first-letter:uppercase">{tip.text}</div>
        {tip.sub && <div className="text-[10px] text-gray-400">{tip.sub}</div>}
      </div>
    </Html>
  );
}

const branchName = (part: AtlasPart, e: ThreeEvent<PointerEvent | MouseEvent>) => {
  const br = part.geometry.getAttribute("_branch");
  return e.face ? part.branches[Math.round(br.getX(e.face.a))] : undefined;
};

/** Children transform (scale / rotate) about `at` instead of the origin. */
function Pivoted({ at, innerRef, children }: { at: THREE.Vector3; innerRef: RefObject<THREE.Group>; children: React.ReactNode }) {
  return (
    <group position={at}>
      <group ref={innerRef}>
        <group position={[-at.x, -at.y, -at.z]}>{children}</group>
      </group>
    </group>
  );
}

/** Dense meshes get a BVH so pointer and label-occlusion raycasts stay cheap. */
function useBvh(parts: AtlasPart[]) {
  useMemo(() => {
    for (const p of parts) if (!p.geometry.boundsTree) p.geometry.boundsTree = new MeshBVH(p.geometry);
  }, [parts]);
}

/** BodyParts3D atlas heart: every coronary branch is a named FMA structure coloured by its vessel's model. */
export default function AtlasHeart(props: Props) {
  const { probs, selected, hovered, mode, xray, chest, twin, timing, lab = null, onLabViews, labFrame = null, pipeline = "", onSelect, onHover, onPick, onReady } = props;
  const gltf = useGLTF(ATLAS_HEART_URL, false, true); // meshopt-compressed; decoder is bundled, no CDN
  const atlas = useMemo(() => prepareAtlasHeart(gltf.scene), [gltf.scene]);
  useBvh(atlas.parts);
  useLabAttributes(atlas);
  const labOn = lab !== null;
  // Lab anatomy (ostium distances, markers, wall mapping): prepared in idle time once the heart is shown, so the
  // first open has no pause (prepareLabAnatomy finishes it at once if the lab opens first).
  useEffect(() => prepareLabAnatomyWhenIdle(atlas), [atlas]);
  const labAnatomy = useMemo(() => (labOn ? prepareLabAnatomy(atlas) : null), [atlas, labOn]);
  const labMaterials = useLabMaterials(atlas);
  // The lab's shader variants (flow pulses, see-through tissue, the marker band) are compiled in idle time too,
  // for the render pipeline in use (they differ with the HD effects on or off).
  const warmed = useRef(new Set<string>());
  const [warming, setWarming] = useState<string | null>(null);
  useEffect(() => {
    if (labOn || warmed.current.has(pipeline)) return;
    return whenIdle(() => setWarming(pipeline), 2500);
  }, [pipeline, labOn]);
  useEffect(() => {
    if (labAnatomy) onLabViews?.(Object.fromEntries(VESSEL_IDS.map((v) => [v, labAnatomy.markers[v].view])) as Record<Vessel, THREE.Vector3>);
  }, [labAnatomy, onLabViews]);
  useLabView(atlas, labAnatomy, lab);
  useFlowPulses(labMaterials, lab, timing);
  const part = (key: string) => atlas.parts.find((p) => p.key === key);
  const wallRef = useRef<THREE.Mesh>(null);
  const atriaRef = useRef<THREE.Mesh>(null);
  const vesselsRef = useRef<THREE.Group>(null);
  const vent = useRef<THREE.Group>(null);
  const atr = useRef<THREE.Group>(null);
  const outflow = useRef<THREE.Group>(null);
  const root = useRef<THREE.Group>(null);
  const tipSetter = useRef<(t: Tip | null) => void>(() => {});
  const tipApi = useMemo(() => ({
    show: (e: ThreeEvent<PointerEvent>, text: string, sub?: string) => {
      if (!root.current) return;
      tipSetter.current({ text, sub, point: root.current.worldToLocal(e.point.clone()) });
    },
    hide: () => tipSetter.current(null),
  }), []);

  useEffect(() => {
    onReady(atlas.views);
  }, [atlas, onReady]);

  // Pivots: each structure contracts about its own centre.
  const pivots = useMemo(() => {
    const centre = (keys: string[]) => {
      const box = new THREE.Box3();
      for (const k of keys) {
        const g = atlas.parts.find((p) => p.key === k)?.geometry;
        if (g?.boundingBox) box.union(g.boundingBox);
      }
      return box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
    };
    return { vent: centre(["ventricles"]), atr: centre(["atria"]), out: centre(["aorta", "pulmonary"]) };
  }, [atlas]);
  useLayoutEffect(() => {
    LAB.uLabPivot.value.copy(pivots.vent);
  }, [pivots]);
  // Cardiac mass (ventricles and atria, not the great vessels): the reasoning view's halo and tower clearance.
  const cardiac = useMemo(() => supportHull(
    ["ventricles", "atria"].flatMap((k) => {
      const g = atlas.parts.find((p) => p.key === k)?.geometry;
      return g ? [{ positions: g.getAttribute("position") }] : [];
    }),
  ), [atlas]);

  // Function twin: ventricles squeeze with the patient's EF, atria kick after the P wave, the great
  // arteries distend with ejection; in LBBB the ventricles rock (septum early, lateral wall late).
  // The lab shaders read the same squeeze to undo part of it in a hypokinetic territory.
  useHeartMotion(twin, timing, (m) => {
    const v = m.ventricular;
    const sx = 1 - m.squeeze * v, sy = 1 - 0.55 * m.squeeze * v;
    vent.current?.scale.set(sx, sy, sx);
    LAB.uLabScale.value.set(sx, sy, sx);
    if (vent.current) vent.current.rotation.z = 0.05 * m.rock;
    atr.current?.scale.setScalar(1 - 0.05 * m.atrial + 0.012 * v);
    outflow.current?.scale.setScalar(1 + m.aortic * v);
  });

  const wall = part("ventricles")!;
  // The lab shows the atria and great arteries see-through (the coronary origins and proximal trunks run under
  // them), so only the ventricles hide its labels.
  const seeThrough = xray || labOn;
  const occluders = xray ? undefined : (labOn ? [wallRef] : [wallRef, atriaRef, vesselsRef]) as RefObject<THREE.Object3D>[];
  const marker = lab && labAnatomy ? labAnatomy.markers[lab.vessel] : null;

  return (
    <TipContext.Provider value={tipApi}>
      <group ref={root} scale={ATLAS_SCALE}>
        <HoverTip setter={tipSetter} />
        <FlowAnchor id="heart" position={cardiac.centre} hull={cardiac.hull} />
        <Pivoted at={pivots.vent} innerRef={vent}>
          <Wall part={wall} atlas={atlas} meshRef={wallRef} probs={probs} selected={selected} focus={labOn ? null : selected ?? hovered} mode={labOn ? "arteries" : mode} xray={xray} lab={labOn} onSelect={onSelect} onPick={onPick} />
          {part("cardiac_veins") && <CardiacVeins part={part("cardiac_veins")!} />}
          {atlas.parts.filter((p) => p.layer === "coronary").map((p) => (
            <Coronary
              key={p.key}
              part={p}
              probs={probs}
              selected={selected}
              hovered={hovered}
              labMaterial={labOn ? labMaterials.byPart[p.key] : null}
              onSelect={onSelect}
              onHover={onHover}
              onPick={onPick}
            />
          ))}
          {VESSEL_IDS.map((v) => (
            <FlowAnchor key={v} id={v} position={atlas.labels[v]} />
          ))}
          {VESSEL_IDS.map((v) => {
            // In the lab the arteries carry flow, not the patient's estimates: plain name tags (click to move the narrowing).
            const p = probs && !labOn ? probs[v] : null;
            return (
              <VesselLabel
                key={v}
                vessel={v}
                name={label(v)}
                position={atlas.labels[v]}
                p={p}
                color={labOn ? LAB_ARTERY : p === null ? NEUTRAL_ARTERY : riskColor(p)}
                selected={labOn ? lab.vessel : selected}
                occluders={occluders}
                onSelect={onSelect}
                scale={labOn && labFrame ? Math.max(0.7, Math.min(1, labFrame.scale)) : 1}
              />
            );
          })}
          {lab && marker && (
            <SandboxMarker marker={marker} narrowing={lab.state.narrowing} vessel={lab.vessel} wall={wallRef} frame={labFrame} />
          )}
          {warming === pipeline && !labOn && (
            <LabWarmup
              atlas={atlas}
              flow={labMaterials}
              at={pivots.vent}
              onDone={() => {
                warmed.current.add(pipeline);
                setWarming(null);
              }}
            />
          )}
        </Pivoted>
        <Pivoted at={pivots.atr} innerRef={atr}>
          {part("atria") && <Tissue part={part("atria")!} meshRef={atriaRef} color="#7d302b" xray={seeThrough} />}
        </Pivoted>
        <Pivoted at={pivots.out} innerRef={outflow}>
          <group ref={vesselsRef}>
            {part("aorta") && <Tissue part={part("aorta")!} color="#d98c7c" xray={seeThrough} sheen />}
            {part("pulmonary") && <Tissue part={part("pulmonary")!} color="#c47a72" xray={seeThrough} sheen />}
            {part("venae_cavae") && <Tissue part={part("venae_cavae")!} color="#6e3442" xray={seeThrough} />}
          </group>
        </Pivoted>
        {chest && (
          <Suspense fallback={null}>
            <Thorax center={atlas.center} layers={chest} />
          </Suspense>
        )}
      </group>
    </TipContext.Provider>
  );
}

// ---------------------------------------------------------------------------------------------------
// Ischaemia lab (teaching sandbox, not this patient's arteries)
// ---------------------------------------------------------------------------------------------------

/** Zero-filled lab attributes on the wall, veins and coronary trees, present before their first draw. */
function useLabAttributes(atlas: PreparedAtlasHeart) {
  useMemo(() => {
    for (const p of atlas.parts) {
      if (p.key !== "ventricles" && p.layer !== "cardiac_vein" && p.layer !== "coronary") continue;
      const n = p.geometry.getAttribute("position").count;
      for (const name of LAB_ATTRIBUTES) {
        if (!p.geometry.getAttribute(name)) p.geometry.setAttribute(name, new THREE.BufferAttribute(new Float32Array(n), 1));
      }
    }
  }, [atlas]);
}

/** One flow-pulse material per coronary tree, grouped by phasic pattern (left and right coronary). */
function useLabMaterials(atlas: PreparedAtlasHeart) {
  const value = useMemo(() => {
    const byPart: Record<string, THREE.MeshPhysicalMaterial> = {};
    const groups = new Map<number, THREE.Material[]>();
    for (const p of atlas.parts.filter((x) => x.layer === "coronary")) {
      const m = patchMaterial(new THREE.MeshPhysicalMaterial({
        color: LAB_ARTERY, emissive: new THREE.Color("#3a0b07"), roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.15,
        side: THREE.DoubleSide,
      }), { lab: "flow" });
      byPart[p.key] = m;
      const v = VESSELS.find((x) => x.atlasPart === p.key);
      const dsvr = v ? ARTERY_PHYSIOLOGY[v.id].dsvr : LEFT_MAIN_DSVR;
      groups.set(dsvr, [...(groups.get(dsvr) ?? []), m]);
    }
    return { byPart, groups: [...groups].map(([dsvr, materials]) => ({ dsvr, materials })) };
  }, [atlas]);
  useEffect(() => () => Object.values(value.byPart).forEach((m) => m.dispose()), [value]);
  return value;
}

/** Writes the chosen artery's territory, restricted path and narrowing into the lab attributes and uniforms. */
function useLabView(atlas: PreparedAtlasHeart, anatomy: AtlasLabAnatomy | null, lab: AtlasLab | null) {
  const vessel = lab?.vessel ?? null;
  useLayoutEffect(() => {
    if (!anatomy || !vessel) return;
    const k = VESSEL_IDS.indexOf(vessel);
    const T = atlas.territory;
    const marker = anatomy.markers[vessel];
    for (const p of atlas.parts) {
      const g = p.geometry;
      const w = g.getAttribute("aLabW") as THREE.BufferAttribute | undefined;
      if (!w) continue;
      const map = anatomy.wallIndex[p.key];
      for (let i = 0; i < w.count; i++) w.setX(i, T[(map ? map[i] : i) * N + k]);
      w.needsUpdate = true;
      if (p.layer !== "coronary") continue;
      const mine = VESSELS.find((v) => v.atlasPart === p.key)?.id === vessel;
      fill(g, "aOstium", anatomy.ostium[p.key]);
      fill(g, "aRestrict", mine ? marker.restricted : null);
      fill(g, "aPinch", mine ? marker.pinch : null);
    }
    LAB.uMarkerPos.value.copy(marker.position);
    LAB.uMarkerDir.value.copy(marker.tangent);
  }, [atlas, anatomy, vessel]);

  const state = lab?.state ?? null;
  useLayoutEffect(() => {
    if (!state) return;
    LAB.uLabHypo.value = 1 - state.wallThickening;
    LAB.uLabTint.value = Math.min(0.9, 1.6 * state.ischaemia); // dark blue, scaled by the ischaemia fraction
    LAB.uLabOutline.value = 0.32;
    LAB.uFlowRel.value = state.relFlow;
    LAB.uFlowGain.value = 2.6;
    LAB.uPinch.value = state.narrowing;
  }, [state]);
  // Lab off (or the heart unmounts): back to the patient's own twin.
  useEffect(() => {
    if (!state) return;
    return () => {
      LAB.uLabHypo.value = 0;
      LAB.uLabTint.value = 0;
      LAB.uLabOutline.value = 0;
      LAB.uFlowGain.value = 0;
      LAB.uPinch.value = 0;
    };
  }, [state === null]); // eslint-disable-line react-hooks/exhaustive-deps
}

function fill(g: THREE.BufferGeometry, name: string, src: Float32Array | null | undefined) {
  const a = g.getAttribute(name) as THREE.BufferAttribute;
  const arr = a.array as Float32Array;
  if (src && src.length === arr.length) arr.set(src);
  else arr.fill(0);
  a.needsUpdate = true;
}

/**
 * Coronary flow pulses: each tree's phase advances with the shared heart clock, weighted towards diastole
 * by its diastolic-to-systolic velocity ratio, at a speed proportional to flow (the demand factor).
 * Integrated over the clock's 2 ms ticks, so the pulses surge in step with relaxation. No allocations.
 */
function useFlowPulses(mats: ReturnType<typeof useLabMaterials>, lab: AtlasLab | null, timing: BeatTiming) {
  const means = useMemo(() => mats.groups.map((g) => phasicMean(g.dsvr, timing)), [mats, timing]);
  const phases = useMemo(() => new Float64Array(mats.groups.length), [mats]);
  const live = useRef({ lab, timing, means });
  live.current = { lab, timing, means };
  const last = useRef(heartClock.count);
  useFrame(() => {
    const { lab: l, timing: tm, means: mean } = live.current;
    let from = last.current;
    const to = heartClock.count;
    last.current = to;
    if (!l) return;
    if (to - from > RING_SIZE) from = to - RING_SIZE;
    const step = FLOW_SPEED_MM_S * l.state.demand * TICK;
    const wrap = FLOW_SPACING_MM * 997;
    for (let k = from; k < to; k++) {
      const v = ventricularContraction(readTick(k), tm);
      for (let g = 0; g < phases.length; g++) {
        const dsvr = mats.groups[g].dsvr;
        phases[g] = (phases[g] + (step * (dsvr - (dsvr - 1) * v)) / mean[g]) % wrap;
      }
    }
    for (let g = 0; g < phases.length; g++) {
      const list = mats.groups[g].materials;
      for (let i = 0; i < list.length; i++) (list[i].userData.uPhase as { value: number }).value = phases[g];
    }
  });
}

/**
 * The SANDBOX narrowing: an amber ring on the chosen artery's proximal trunk (the vessel waist inside it is
 * drawn by the shader) with a label that says what it is. It follows a hypokinetic wall like the vessel.
 *
 * Band, ring and label show and hide together: hidden only while the ventricular wall stands well in front of
 * the marker (more than 3 vessel radii: the groove the trunk runs in, the heartbeat or a frame's lag never
 * count) for a few frames in a row, so it never flickers and never shows as a bare amber band. The label sits
 * above the ring, or below it near the top of the free view, and shifts sideways to stay in that view.
 */
function SandboxMarker({ marker, narrowing, vessel, wall, frame }: {
  marker: LabMarker;
  narrowing: number;
  vessel: Vessel;
  /** The ventricular wall: in the lab (atria and great arteries see-through) the only thing that can hide the marker. */
  wall: RefObject<THREE.Mesh>;
  frame: LabFrame | null;
}) {
  const group = useRef<THREE.Group>(null);
  const band = useRef<THREE.Mesh>(null);
  const box = useRef<HTMLDivElement>(null);
  const leader = useRef<HTMLDivElement>(null);
  const tag = useRef<HTMLDivElement | null>(null);
  const tagWidth = useRef(0);
  const live = useRef(frame);
  live.current = frame;
  const vis = useRef({ shown: true, run: 0, below: false, shift: 0 });
  const quaternion = useMemo(
    () => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), marker.tangent),
    [marker],
  );
  const local = useMemo(() => new THREE.Vector3(), []);
  // The label's width (for the sideways shift), measured after layout rather than read every frame.
  const observer = useMemo(() => (typeof ResizeObserver === "undefined" ? null : new ResizeObserver((entries) => {
    for (const e of entries) tagWidth.current = (e.target as HTMLElement).offsetWidth;
  })), []);
  useEffect(() => () => observer?.disconnect(), [observer]);
  const tagRef = (el: HTMLDivElement | null) => {
    if (tag.current && tag.current !== el) observer?.unobserve(tag.current);
    tag.current = el;
    if (el) {
      tagWidth.current = el.offsetWidth;
      observer?.observe(el);
    }
  };

  useFrame(({ camera, size }) => {
    const g = group.current;
    if (!g) return;
    // Same local scale as the shader at the marker (its territory weight is that of the trunk there, ~1).
    const S = LAB.uLabScale.value;
    const c = Math.min(1, Math.max(0, LAB.uLabHypo.value));
    local.set(1 - (1 - S.x) * (1 - c), 1 - (1 - S.y) * (1 - c), 1 - (1 - S.z) * (1 - c)).divide(S);
    g.position.copy(marker.position).sub(LAB.uLabPivot.value).multiply(local).add(LAB.uLabPivot.value);

    g.updateWorldMatrix(true, false);
    _markerPos.setFromMatrixPosition(g.matrixWorld);
    const w = wall.current;
    let blocked = false;
    if (w) {
      w.updateWorldMatrix(true, false); // this frame's squeeze, like the marker
      _markerDir.copy(_markerPos).sub(camera.position);
      const dist = _markerDir.length();
      const tolerance = 3 * marker.radius * _markerScale.setFromMatrixScale(g.matrixWorld).x;
      _markerRay.set(camera.position, _markerDir.divideScalar(dist));
      _markerRay.far = Math.max(0, dist - tolerance);
      blocked = _markerRay.intersectObject(w, false).length > 0; // no hits in X-ray (the wall is not pickable)
    }
    const v = vis.current;
    if (blocked === v.shown) {
      // The other state must persist (hide after 6 frames, show again after 3) before anything changes.
      v.run++;
      if (v.run >= (v.shown ? 6 : 3)) {
        v.shown = !v.shown;
        v.run = 0;
        if (band.current) band.current.visible = v.shown;
        if (box.current) box.current.style.visibility = v.shown ? "" : "hidden";
      }
    } else v.run = 0;

    const f = live.current;
    if (!f || !tag.current || !leader.current) return;
    _markerNdc.copy(_markerPos).project(camera);
    const x = ((_markerNdc.x + 1) / 2) * size.width;
    const y = ((1 - _markerNdc.y) / 2) * size.height;
    // Ring radius 14 px, leader 16 px, label about 20 px: flip below when the label would reach the top.
    const below = v.below ? y - 66 < f.top : y - 56 < f.top;
    const half = tagWidth.current / 2;
    const lo = f.left + 6 - (x - half), hi = f.right - 6 - (x + half);
    const shift = Math.round(lo > 0 ? lo : hi < 0 ? Math.max(hi, lo) : 0);
    if (below !== v.below) {
      v.below = below;
      leader.current.style.top = below ? "100%" : "";
      leader.current.style.bottom = below ? "auto" : "";
      tag.current.style.top = below ? "calc(100% + 16px)" : "";
      tag.current.style.bottom = below ? "auto" : "";
    }
    if (shift !== v.shift) {
      v.shift = shift;
      tag.current.style.transform = `translateX(calc(-50% + ${shift}px))`;
    }
  });
  const r = marker.radius;
  return (
    <group ref={group} position={marker.position}>
      {/* A band round the vessel (the waist inside it is drawn by the vessel shader)... */}
      <mesh ref={band} quaternion={quaternion} raycast={() => null} renderOrder={5}>
        <torusGeometry args={[r * 2.4, r * 0.38, 10, 40]} />
        <meshBasicMaterial color={LAB_MARKER_COLOR} toneMapped={false} />
      </mesh>
      {/* ...and a screen-space ring with its label, so it reads at any zoom. */}
      <Html center zIndexRange={[16, 12]} style={{ pointerEvents: "none" }}>
        <div ref={box} className="relative h-7 w-7">
          <div className="absolute inset-0 rounded-full border-2 border-amber-400 shadow-[0_0_10px_rgba(245,158,11,0.85)]" />
          <div ref={leader} className="absolute bottom-full left-1/2 h-4 w-0 -translate-x-1/2 border-l-[1.5px] border-amber-400" />
          <div
            ref={tagRef}
            className="absolute bottom-[calc(100%+16px)] left-1/2 whitespace-nowrap rounded border border-amber-400 bg-amber-950/90 px-1.5 py-0.5 text-[10px] font-semibold leading-tight text-amber-100 shadow"
            style={{ transform: "translateX(-50%)" }}
          >
            SANDBOX narrowing {Math.round(narrowing * 100)}% <span className="font-normal text-amber-300">· {label(vessel)}</span>
          </div>
        </div>
      </Html>
    </group>
  );
}
const _markerPos = new THREE.Vector3();
const _markerDir = new THREE.Vector3();
const _markerNdc = new THREE.Vector3();
const _markerScale = new THREE.Vector3();
const _markerRay = Object.assign(new THREE.Raycaster(), { firstHitOnly: true });

/**
 * Draws the lab's shader variants once, tiny and inside the heart, so they are compiled while the user is not
 * waiting (the first lab open then has nothing to compile): the flow-pulse material, the see-through tissue
 * the lab shows the atria and great arteries with, and the marker band. Unmounts itself after two frames; its
 * own materials stay alive (disposing them could release the shared programs).
 */
function LabWarmup({ atlas, flow, at, onDone }: {
  atlas: PreparedAtlasHeart;
  flow: ReturnType<typeof useLabMaterials>;
  at: THREE.Vector3;
  onDone: () => void;
}) {
  const frames = useRef(0);
  const items = useMemo(() => {
    const coronary = atlas.parts.find((p) => p.layer === "coronary");
    const atria = atlas.parts.find((p) => p.key === "atria");
    const out: { geometry: THREE.BufferGeometry; material: THREE.Material }[] = [];
    if (coronary && flow.byPart[coronary.key]) out.push({ geometry: coronary.geometry, material: flow.byPart[coronary.key] });
    if (atria) out.push({ geometry: atria.geometry, material: warmupMaterials().tissue });
    out.push({ geometry: warmupMaterials().bandGeometry, material: warmupMaterials().band });
    return out;
  }, [atlas, flow]);
  useFrame(() => {
    frames.current += 1;
    if (frames.current === 3) onDone();
  });
  return (
    <group position={at} scale={1e-4}>
      {items.map((it, i) => (
        <mesh key={i} geometry={it.geometry} material={it.material} frustumCulled={false} raycast={() => null} />
      ))}
    </group>
  );
}

let warmup: { tissue: THREE.Material; band: THREE.Material; bandGeometry: THREE.BufferGeometry } | null = null;
function warmupMaterials() {
  if (!warmup) {
    const tissue = tissueMaterial("#7d302b", false);
    tissue.transparent = true; // as useXray sets it in the lab (and in X-ray mode)
    tissue.opacity = 0.4;
    tissue.depthWrite = false;
    warmup = {
      tissue,
      band: new THREE.MeshBasicMaterial({ color: LAB_MARKER_COLOR, toneMapped: false }),
      bandGeometry: new THREE.TorusGeometry(1, 0.2, 6, 12),
    };
  }
  return warmup;
}

interface WallProps {
  part: AtlasPart;
  atlas: PreparedAtlasHeart;
  meshRef: RefObject<THREE.Mesh>;
  probs: ArteryProbs | null;
  selected: Vessel | null;
  /** Artery whose territory is highlighted: the selected one, else the hovered one. */
  focus: Vessel | null;
  mode: "arteries" | "territories";
  xray: boolean;
  /** Ischaemia lab on: clicking moves the sandbox narrowing to the artery that supplies this wall. */
  lab: boolean;
  onSelect: (v: Vessel | null) => void;
  onPick: (pick: AtlasPick | null) => void;
}

/** Ventricular myocardium: muscle with epicardial fat in the grooves; tinted by territory. */
function Wall({ part, atlas, meshRef, probs, selected, focus, mode, xray, lab, onSelect, onPick }: WallProps) {
  const { geometry } = part;
  const base = useMemo(() => {
    const pos = geometry.getAttribute("position");
    const out = new Float32Array(pos.count * 3);
    const c = new THREE.Color();
    const f = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i) * 90, y = pos.getY(i) * 90, z = pos.getZ(i) * 90;
      const mottle = 0.5 + 0.5 * fbm(x, y, z, 3);
      c.copy(MUSCLE_DEEP).lerp(MUSCLE, 0.35 + 0.65 * mottle);
      // Fat lobules: patchy rather than a smooth band.
      const lobules = 0.5 + 0.5 * fbm(x * 3.1 + 7, y * 3.1, z * 3.1, 2);
      f.copy(FAT_DEEP).lerp(FAT, lobules);
      const fat = Math.min(1, atlas.fat[i] * (0.75 + 0.6 * lobules));
      c.lerp(f, 0.9 * fat);
      out.set([c.r, c.g, c.b], i * 3);
    }
    return out;
  }, [geometry, atlas]);

  useLayoutEffect(() => {
    if (!geometry.getAttribute("color")) geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(base.length), 3));
    const color = geometry.getAttribute("color") as THREE.BufferAttribute;
    const T = atlas.territory;
    const c = new THREE.Color();
    const tint = new THREE.Color();
    const tmp = new THREE.Color();
    for (let i = 0; i < color.count; i++) {
      c.setRGB(base[i * 3], base[i * 3 + 1], base[i * 3 + 2]);
      const fat = atlas.fat[i];
      if (probs && mode === "territories") {
        tint.setRGB(0, 0, 0);
        VESSEL_IDS.forEach((t, k) => tint.add(riskColor(probs[t], tmp).multiplyScalar(T[i * N + k])));
        c.lerp(tint, 0.74 - 0.2 * fat);
      } else if (focus) {
        tint.copy(probs ? riskColor(probs[focus], tmp) : NEUTRAL_ARTERY);
        c.lerp(tint, 0.5 * Math.max(0, (T[i * N + VESSEL_IDS.indexOf(focus)] - 0.35) / 0.65));
      }
      color.setXYZ(i, c.r, c.g, c.b);
    }
    color.needsUpdate = true;
  }, [geometry, base, atlas, probs, focus, mode]);

  const material = useMemo(() => patchMaterial(new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    roughness: 0.5,
    clearcoat: 0.7,
    clearcoatRoughness: 0.28,
    sheen: 0.8,
    sheenRoughness: 0.5,
    sheenColor: new THREE.Color("#ff5a48"),
    envMapIntensity: 0.85,
  }), { bump: { freq: 260, strength: 0.9 }, fresnel: XRAY_FRESNEL, lab: "wall" }), []);
  useXray(material, xray, 0.55);

  const supplier = (face: THREE.Face) => {
    const T = atlas.territory;
    return VESSEL_IDS[VESSEL_IDS.reduce((best, _, k) => (T[face.a * N + k] > T[face.a * N + best] ? k : best), 0)];
  };
  const onClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (!e.face) return;
    const v = supplier(e.face);
    onPick(null);
    onSelect(lab || v !== selected ? v : null);
  };
  const tip = useTip((e) => (e.face
    ? { text: "Ventricular wall", sub: `Usually supplied by the ${label(supplier(e.face))} · ${lab ? "click to put the sandbox narrowing on it" : "click to inspect"}` }
    : null));

  return (
    <BvhMesh
      meshRef={meshRef}
      geometry={geometry}
      material={material}
      pickable={!xray}
      onClick={onClick}
      {...tip}
    />
  );
}

/** Material of the atria and great vessels (also built, see-through, by LabWarmup to compile that variant early). */
function tissueMaterial(color: string, sheen: boolean) {
  return patchMaterial(new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.45,
    clearcoat: 0.8,
    clearcoatRoughness: 0.25,
    sheen: sheen ? 0.6 : 0.8,
    sheenRoughness: 0.5,
    sheenColor: new THREE.Color(sheen ? "#ffb0a0" : "#ff6650"),
    side: THREE.DoubleSide,
  }), { bump: { freq: 200, strength: 0.6 }, fresnel: XRAY_FRESNEL });
}

/** Atria and great vessels: opaque tissue that swallows clicks (so vessels behind can't be picked through it). */
function Tissue({ part, color, xray, sheen, meshRef }: {
  part: AtlasPart; color: string; xray: boolean; sheen?: boolean; meshRef?: RefObject<THREE.Mesh>;
}) {
  const material = useMemo(() => tissueMaterial(color, !!sheen), [color, sheen]);
  useXray(material, xray, 0.4);
  const tip = useTip((e) => {
    const b = branchName(part, e);
    return b ? { text: b.name, sub: b.fma } : null;
  });
  return (
    <BvhMesh
      meshRef={meshRef}
      geometry={part.geometry}
      material={material}
      pickable={!xray}
      onClick={(e) => e.stopPropagation()}
      {...tip}
    />
  );
}

function CardiacVeins({ part }: { part: AtlasPart }) {
  // Lab-patched so the veins follow a hypokinetic territory with the wall (no effect otherwise).
  const material = useMemo(() => patchMaterial(new THREE.MeshPhysicalMaterial({
    color: "#4a2638", roughness: 0.32, clearcoat: 0.9, clearcoatRoughness: 0.2, side: THREE.DoubleSide,
  }), { lab: "vessel" }), []);
  // Anatomical context: names itself on hover, but clicks pass through to the wall underneath.
  const tip = useTip((e) => {
    const b = branchName(part, e);
    return b ? { text: b.name, sub: `${b.fma} · cardiac vein, not modelled` } : null;
  });
  return <BvhMesh geometry={part.geometry} material={material} pickable {...tip} />;
}

interface CoronaryProps {
  part: AtlasPart;
  probs: ArteryProbs | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  /** Ischaemia lab: flow-pulse material used instead of the risk colour. */
  labMaterial: THREE.Material | null;
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
  onPick: (pick: AtlasPick | null) => void;
}

/** One coronary tree (all its named branches in one mesh), coloured by its vessel's model. */
function Coronary({ part, probs, selected, hovered, labMaterial, onSelect, onHover, onPick }: CoronaryProps) {
  // Model output shown by this tree (registry); trees without one (the left main) stay neutral.
  const vessel = VESSELS.find((v) => v.atlasPart === part.key)?.id ?? null;
  const unmodelled = UNMODELLED.find((u) => u.atlasPart === part.key);
  const { material } = useVesselMaterial(vessel, probs, selected, hovered);

  const branchAt = (e: ThreeEvent<PointerEvent | MouseEvent>) => branchName(part, e);
  const tip = useTip((e) => {
    const b = branchAt(e);
    if (!b) return null;
    if (labMaterial) {
      return { text: b.name, sub: vessel ? `${b.fma} · lab: click to put the sandbox narrowing on the ${label(vessel)}` : `${b.fma} · feeds the LAD and LCX` };
    }
    return {
      text: b.name,
      sub: vessel ? `${b.fma} · coloured by the ${label(vessel)} model` : `${b.fma} · not labelled in the dataset`,
    };
  });

  return (
    <BvhMesh
      geometry={part.geometry}
      material={labMaterial ?? material}
      pickable
      onClick={(e) => {
        e.stopPropagation();
        const branch = branchAt(e);
        if (!vessel) {
          // Left main: shown for anatomy, but the dataset has no left-main label to colour it by.
          if (branch && unmodelled) onPick({ vessel: unmodelled.id, part: part.label, partFma: part.fma, branch });
          return;
        }
        const deselect = !labMaterial && vessel === selected;
        onPick(deselect || !branch || labMaterial ? null : { vessel, part: part.label, partFma: part.fma, branch });
        onSelect(deselect ? null : vessel);
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        if (vessel) onHover(vessel);
      }}
      onPointerMove={tip.onPointerMove}
      onPointerOut={() => {
        onHover(null);
        tip.onPointerOut();
      }}
    />
  );
}

/** A mesh using the geometry's BVH for raycasts; pickable=false lets pointer events pass through. */
function BvhMesh({ meshRef, geometry, material, pickable, ...handlers }: {
  meshRef?: RefObject<THREE.Mesh>;
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  pickable: boolean;
  onClick?: (e: ThreeEvent<MouseEvent>) => void;
  onPointerMove?: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOver?: (e: ThreeEvent<PointerEvent>) => void;
  onPointerOut?: (e: ThreeEvent<PointerEvent>) => void;
}) {
  const local = useRef<THREE.Mesh>(null);
  const ref = meshRef ?? local;
  useLayoutEffect(() => {
    if (ref.current) ref.current.raycast = pickable ? acceleratedRaycast : () => null;
  }, [ref, pickable]);
  return <mesh ref={ref} geometry={geometry} material={material} {...handlers} />;
}

/** X-ray: make opaque tissue faint and stop it hiding the vessels behind it. */
function useXray(material: THREE.Material, on: boolean, opacity: number) {
  useEffect(() => {
    material.transparent = on;
    material.opacity = on ? opacity : 1;
    material.depthWrite = !on;
    material.needsUpdate = true;
  }, [material, on, opacity]);
}

/** Chest context: ghosted skin, skeleton, costal cartilage, diaphragm and airway, with a cutaway onto the heart. */
function Thorax({ center, layers }: { center: THREE.Vector3; layers: ChestLayers }) {
  const gltf = useGLTF(ATLAS_THORAX_URL, false, true);
  const parts = useMemo(() => prepareThorax(gltf.scene, center), [gltf.scene, center]);
  // Lambert shading for the chest: it covers much of the screen, so it must stay cheap on integrated GPUs.
  const materials = useMemo(() => ({
    skin: patchMaterial(new THREE.MeshLambertMaterial({
      color: "#9fb6d0", transparent: true, opacity: 0.55, depthWrite: false, side: THREE.FrontSide,
    }), { cutaway: true, fresnel: { min: 0.04, power: 2.6 } }),
    bone: patchMaterial(new THREE.MeshLambertMaterial({ color: "#e6dcc6" }), { cutaway: true }),
    cartilage: patchMaterial(new THREE.MeshLambertMaterial({ color: "#c9d6dc", transparent: true, opacity: 0.85 }), { cutaway: true }),
    diaphragm: patchMaterial(new THREE.MeshLambertMaterial({
      color: "#a3473d", transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide,
    }), { cutaway: true }),
    airway: patchMaterial(new THREE.MeshLambertMaterial({ color: "#d9b8ae" }), { cutaway: true }),
  }), []);
  return (
    <group>
      {parts.filter((p) => layers[LAYER_GROUP[p.layer] ?? "skeleton"]).map((p) => {
        const m = materials[p.layer as keyof typeof materials] ?? materials.bone;
        // Skin draws last (it is the outermost translucent layer); nothing in the chest takes clicks.
        return <mesh key={p.key} geometry={p.geometry} material={m} renderOrder={p.layer === "skin" ? 3 : p.layer === "diaphragm" ? 2 : 0} raycast={() => null} />;
      })}
    </group>
  );
}

useGLTF.preload(ATLAS_HEART_URL, false, true);
