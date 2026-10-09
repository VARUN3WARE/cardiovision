import type { ThreeEvent } from "@react-three/fiber";
import { Html, useGLTF } from "@react-three/drei";
import {
  Suspense, createContext, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState,
  type MutableRefObject, type RefObject,
} from "react";
import * as THREE from "three";
import { MeshBVH, acceleratedRaycast } from "three-mesh-bvh";
import { VesselLabel, useVesselMaterial, type ArteryProbs } from "./ArteryTube";
import {
  ATLAS_HEART_URL, ATLAS_SCALE, ATLAS_THORAX_URL, prepareAtlasHeart, prepareThorax,
  type AtlasBranch, type AtlasPart, type PreparedAtlasHeart,
} from "./atlasAnatomy";
import { patchMaterial } from "./atlasMaterials";
import { NEUTRAL_ARTERY, fbm, riskColor } from "./heartAnatomy";
import type { BeatTiming, TwinParams } from "../twin/twin";
import { useHeartMotion } from "../twin/useHeartMotion";
import { UNMODELLED, VESSEL, VESSELS, VESSEL_IDS, type Vessel } from "../anatomy/registry";

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

interface Props {
  probs: ArteryProbs | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  mode: "arteries" | "territories";
  xray: boolean;
  chest: ChestLayers | null;
  twin: TwinParams;
  timing: BeatTiming;
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
  onPick: (pick: AtlasPick | null) => void;
  onReady: (views: Record<Vessel, THREE.Vector3>) => void;
}

const N = VESSEL_IDS.length;
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
  const { probs, selected, hovered, mode, xray, chest, twin, timing, onSelect, onHover, onPick, onReady } = props;
  const gltf = useGLTF(ATLAS_HEART_URL, false, true); // meshopt-compressed; decoder is bundled, no CDN
  const atlas = useMemo(() => prepareAtlasHeart(gltf.scene), [gltf.scene]);
  useBvh(atlas.parts);
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

  // Function twin: ventricles squeeze with the patient's EF, atria kick after the P wave, the great
  // arteries distend with ejection; in LBBB the ventricles rock (septum early, lateral wall late).
  useHeartMotion(twin, timing, (m) => {
    const v = m.ventricular;
    vent.current?.scale.set(1 - m.squeeze * v, 1 - 0.55 * m.squeeze * v, 1 - m.squeeze * v);
    if (vent.current) vent.current.rotation.z = 0.05 * m.rock;
    atr.current?.scale.setScalar(1 - 0.05 * m.atrial + 0.012 * v);
    outflow.current?.scale.setScalar(1 + m.aortic * v);
  });

  const wall = part("ventricles")!;
  const occluders = xray ? undefined : [wallRef, atriaRef, vesselsRef] as RefObject<THREE.Object3D>[];

  return (
    <TipContext.Provider value={tipApi}>
      <group ref={root} scale={ATLAS_SCALE}>
        <HoverTip setter={tipSetter} />
        <Pivoted at={pivots.vent} innerRef={vent}>
          <Wall part={wall} atlas={atlas} meshRef={wallRef} probs={probs} selected={selected} focus={selected ?? hovered} mode={mode} xray={xray} onSelect={onSelect} onPick={onPick} />
          {part("cardiac_veins") && <CardiacVeins part={part("cardiac_veins")!} />}
          {atlas.parts.filter((p) => p.layer === "coronary").map((p) => (
            <Coronary key={p.key} part={p} probs={probs} selected={selected} hovered={hovered} onSelect={onSelect} onHover={onHover} onPick={onPick} />
          ))}
          {VESSEL_IDS.map((v) => {
            const p = probs ? probs[v] : null;
            return (
              <VesselLabel
                key={v}
                vessel={v}
                name={label(v)}
                position={atlas.labels[v]}
                p={p}
                color={p === null ? NEUTRAL_ARTERY : riskColor(p)}
                selected={selected}
                occluders={occluders}
                onSelect={onSelect}
              />
            );
          })}
        </Pivoted>
        <Pivoted at={pivots.atr} innerRef={atr}>
          {part("atria") && <Tissue part={part("atria")!} meshRef={atriaRef} color="#7d302b" xray={xray} />}
        </Pivoted>
        <Pivoted at={pivots.out} innerRef={outflow}>
          <group ref={vesselsRef}>
            {part("aorta") && <Tissue part={part("aorta")!} color="#d98c7c" xray={xray} sheen />}
            {part("pulmonary") && <Tissue part={part("pulmonary")!} color="#c47a72" xray={xray} sheen />}
            {part("venae_cavae") && <Tissue part={part("venae_cavae")!} color="#6e3442" xray={xray} />}
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
  onSelect: (v: Vessel | null) => void;
  onPick: (pick: AtlasPick | null) => void;
}

/** Ventricular myocardium: muscle with epicardial fat in the grooves; tinted by territory. */
function Wall({ part, atlas, meshRef, probs, selected, focus, mode, xray, onSelect, onPick }: WallProps) {
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
  }), { bump: { freq: 260, strength: 0.9 }, fresnel: XRAY_FRESNEL }), []);
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
    onSelect(v === selected ? null : v);
  };
  const tip = useTip((e) => (e.face ? { text: "Ventricular wall", sub: `Usually supplied by the ${label(supplier(e.face))} · click to inspect` } : null));

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

/** Atria and great vessels: opaque tissue that swallows clicks (so vessels behind can't be picked through it). */
function Tissue({ part, color, xray, sheen, meshRef }: {
  part: AtlasPart; color: string; xray: boolean; sheen?: boolean; meshRef?: RefObject<THREE.Mesh>;
}) {
  const material = useMemo(() => patchMaterial(new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.45,
    clearcoat: 0.8,
    clearcoatRoughness: 0.25,
    sheen: sheen ? 0.6 : 0.8,
    sheenRoughness: 0.5,
    sheenColor: new THREE.Color(sheen ? "#ffb0a0" : "#ff6650"),
    side: THREE.DoubleSide,
  }), { bump: { freq: 200, strength: 0.6 }, fresnel: XRAY_FRESNEL }), [color, sheen]);
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
  const material = useMemo(() => new THREE.MeshPhysicalMaterial({
    color: "#4a2638", roughness: 0.32, clearcoat: 0.9, clearcoatRoughness: 0.2, side: THREE.DoubleSide,
  }), []);
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
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
  onPick: (pick: AtlasPick | null) => void;
}

/** One coronary tree (all its named branches in one mesh), coloured by its vessel's model. */
function Coronary({ part, probs, selected, hovered, onSelect, onHover, onPick }: CoronaryProps) {
  // Model output shown by this tree (registry); trees without one (the left main) stay neutral.
  const vessel = VESSELS.find((v) => v.atlasPart === part.key)?.id ?? null;
  const unmodelled = UNMODELLED.find((u) => u.atlasPart === part.key);
  const { material } = useVesselMaterial(vessel, probs, selected, hovered);

  const branchAt = (e: ThreeEvent<PointerEvent | MouseEvent>) => branchName(part, e);
  const tip = useTip((e) => {
    const b = branchAt(e);
    if (!b) return null;
    return {
      text: b.name,
      sub: vessel ? `${b.fma} · coloured by the ${label(vessel)} model` : `${b.fma} · not labelled in the dataset`,
    };
  });

  return (
    <BvhMesh
      geometry={part.geometry}
      material={material}
      pickable
      onClick={(e) => {
        e.stopPropagation();
        const branch = branchAt(e);
        if (!vessel) {
          // Left main: shown for anatomy, but the dataset has no left-main label to colour it by.
          if (branch && unmodelled) onPick({ vessel: unmodelled.id, part: part.label, partFma: part.fma, branch });
          return;
        }
        const deselect = vessel === selected;
        onPick(deselect || !branch ? null : { vessel, part: part.label, partFma: part.fma, branch });
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
