import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { Suspense, useEffect, useLayoutEffect, useMemo, useRef, type RefObject } from "react";
import * as THREE from "three";
import { MeshBVH, acceleratedRaycast } from "three-mesh-bvh";
import { VesselLabel, type ArteryProbs } from "./ArteryTube";
import {
  ATLAS_HEART_URL, ATLAS_SCALE, ATLAS_THORAX_URL, prepareAtlasHeart, prepareThorax,
  type AtlasBranch, type AtlasPart, type PreparedAtlasHeart,
} from "./atlasAnatomy";
import { patchMaterial } from "./atlasMaterials";
import { NEUTRAL_ARTERY, fbm, riskColor, type Vessel } from "./heartAnatomy";

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
  chest: boolean;
  bpm: number;
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
  onPick: (pick: AtlasPick | null) => void;
  onReady: (views: Record<Vessel, THREE.Vector3>) => void;
}

const TARGETS: Vessel[] = ["lad", "lcx", "rca"];
const LABEL: Record<Vessel, string> = { lad: "LAD", lcx: "LCX", rca: "RCA" };

const MUSCLE_DEEP = new THREE.Color("#5c1d1a");
const MUSCLE = new THREE.Color("#8e3a31");
const FAT_DEEP = new THREE.Color("#c4954f");
const FAT = new THREE.Color("#ecd39a");
// In X-ray mode tissue fades to a ghost that keeps its silhouette (no effect while opaque).
const XRAY_FRESNEL = { min: 0.12, power: 2.2 };

/** Dense meshes get a BVH so pointer and label-occlusion raycasts stay cheap. */
function useBvh(parts: AtlasPart[]) {
  useMemo(() => {
    for (const p of parts) if (!p.geometry.boundsTree) p.geometry.boundsTree = new MeshBVH(p.geometry);
  }, [parts]);
}

/** BodyParts3D atlas heart: every coronary branch is a named FMA structure coloured by its vessel's model. */
export default function AtlasHeart(props: Props) {
  const { probs, selected, hovered, mode, xray, chest, bpm, onSelect, onHover, onPick, onReady } = props;
  const gltf = useGLTF(ATLAS_HEART_URL, false, true); // meshopt-compressed; decoder is bundled, no CDN
  const atlas = useMemo(() => prepareAtlasHeart(gltf.scene), [gltf.scene]);
  useBvh(atlas.parts);
  const part = (key: string) => atlas.parts.find((p) => p.key === key);
  const wallRef = useRef<THREE.Mesh>(null);
  const atriaRef = useRef<THREE.Mesh>(null);
  const vesselsRef = useRef<THREE.Group>(null);
  const beat = useRef<THREE.Group>(null);
  const phase = useRef(0);

  useEffect(() => {
    onReady(atlas.views);
  }, [atlas, onReady]);

  useFrame((_, dt) => {
    phase.current = (phase.current + (dt * bpm) / 60) % 1;
    const t = phase.current;
    const pulse = (c: number, w: number) => (bpm > 0 ? Math.exp(-(((t - c) / w) ** 2)) : 0);
    const v = pulse(0.16, 0.07) + 0.25 * pulse(0.34, 0.05);
    beat.current?.scale.set(1 - 0.022 * v, 1 - 0.012 * v, 1 - 0.022 * v);
  });

  const wall = part("ventricles")!;
  const occluders = xray ? undefined : [wallRef, atriaRef, vesselsRef] as RefObject<THREE.Object3D>[];

  return (
    <group scale={ATLAS_SCALE}>
      <group ref={beat}>
        <Wall part={wall} atlas={atlas} meshRef={wallRef} probs={probs} selected={selected} mode={mode} xray={xray} onSelect={onSelect} onPick={onPick} />
        {part("atria") && <Tissue part={part("atria")!} meshRef={atriaRef} color="#7d302b" xray={xray} />}
        <group ref={vesselsRef}>
          {part("aorta") && <Tissue part={part("aorta")!} color="#d98c7c" xray={xray} sheen />}
          {part("pulmonary") && <Tissue part={part("pulmonary")!} color="#c47a72" xray={xray} sheen />}
          {part("venae_cavae") && <Tissue part={part("venae_cavae")!} color="#6e3442" xray={xray} />}
        </group>
        {part("cardiac_veins") && <CardiacVeins part={part("cardiac_veins")!} />}
        {atlas.parts.filter((p) => p.layer === "coronary").map((p) => (
          <Coronary key={p.key} part={p} probs={probs} selected={selected} hovered={hovered} onSelect={onSelect} onHover={onHover} onPick={onPick} />
        ))}
        {TARGETS.map((v) => {
          const p = probs ? probs[v] : null;
          return (
            <VesselLabel
              key={v}
              vessel={v}
              name={LABEL[v]}
              position={atlas.labels[v]}
              p={p}
              color={p === null ? NEUTRAL_ARTERY : riskColor(p)}
              selected={selected}
              occluders={occluders}
              onSelect={onSelect}
            />
          );
        })}
      </group>
      {chest && (
        <Suspense fallback={null}>
          <Thorax center={atlas.center} />
        </Suspense>
      )}
    </group>
  );
}

interface WallProps {
  part: AtlasPart;
  atlas: PreparedAtlasHeart;
  meshRef: RefObject<THREE.Mesh>;
  probs: ArteryProbs | null;
  selected: Vessel | null;
  mode: "arteries" | "territories";
  xray: boolean;
  onSelect: (v: Vessel | null) => void;
  onPick: (pick: AtlasPick | null) => void;
}

/** Ventricular myocardium: muscle with epicardial fat in the grooves; tinted by territory. */
function Wall({ part, atlas, meshRef, probs, selected, mode, xray, onSelect, onPick }: WallProps) {
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
        TARGETS.forEach((t, k) => tint.add(riskColor(probs[t], tmp).multiplyScalar(T[i * 3 + k])));
        c.lerp(tint, 0.6 - 0.2 * fat);
      } else if (selected) {
        tint.copy(probs ? riskColor(probs[selected], tmp) : NEUTRAL_ARTERY);
        c.lerp(tint, 0.5 * Math.max(0, (T[i * 3 + TARGETS.indexOf(selected)] - 0.35) / 0.65));
      }
      color.setXYZ(i, c.r, c.g, c.b);
    }
    color.needsUpdate = true;
  }, [geometry, base, atlas, probs, selected, mode]);

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

  const onClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (!e.face) return;
    const T = atlas.territory;
    const i = e.face.a;
    const k = [0, 1, 2].reduce((a, b) => (T[i * 3 + a] >= T[i * 3 + b] ? a : b));
    onPick(null);
    onSelect(TARGETS[k] === selected ? null : TARGETS[k]);
  };

  return (
    <BvhMesh
      meshRef={meshRef}
      geometry={geometry}
      material={material}
      pickable={!xray}
      onClick={onClick}
      onPointerMove={(e) => e.stopPropagation()}
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
  return (
    <BvhMesh
      meshRef={meshRef}
      geometry={part.geometry}
      material={material}
      pickable={!xray}
      onClick={(e) => e.stopPropagation()}
      onPointerMove={(e) => e.stopPropagation()}
    />
  );
}

function CardiacVeins({ part }: { part: AtlasPart }) {
  const material = useMemo(() => new THREE.MeshPhysicalMaterial({
    color: "#4a2638", roughness: 0.32, clearcoat: 0.9, clearcoatRoughness: 0.2, side: THREE.DoubleSide,
  }), []);
  // Anatomical context only: no pointer events.
  return <mesh geometry={part.geometry} material={material} raycast={() => null} />;
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
  const vessel = part.vessel === "lm" || !part.vessel ? null : part.vessel;
  const p = probs && vessel ? probs[vessel] : null;
  const material = useMemo(() => new THREE.MeshPhysicalMaterial({
    color: NEUTRAL_ARTERY.clone(),
    emissive: NEUTRAL_ARTERY.clone(),
    roughness: 0.3,
    clearcoat: 1,
    clearcoatRoughness: 0.15,
    side: THREE.DoubleSide,
  }), []);
  const target = useMemo(() => (p === null ? NEUTRAL_ARTERY.clone() : riskColor(p)), [p]);
  const isSelected = vessel !== null && vessel === selected;
  const dimmed = selected !== null && !isSelected;

  useFrame(({ clock }, dt) => {
    // Colours glide to a new estimate instead of jumping.
    const k = 1 - Math.exp(-8 * dt);
    material.color.lerp(target, k);
    material.emissive.copy(material.color);
    material.emissiveIntensity = dimmed
      ? 0.04
      : isSelected
        ? 2.2 + 1.6 * Math.sin(clock.elapsedTime * 4)
        : vessel && vessel === hovered
          ? 1.0
          : p !== null && p >= 0.66
            ? 0.5
            : vessel ? 0.22 : 0.05;
    const opacity = dimmed ? 0.35 : 1;
    if (material.opacity !== opacity) {
      material.opacity = opacity;
      material.transparent = opacity < 1;
      material.needsUpdate = true;
    }
  });

  const branchAt = (e: ThreeEvent<PointerEvent | MouseEvent>) => {
    const br = part.geometry.getAttribute("_branch");
    return e.face ? part.branches[Math.round(br.getX(e.face.a))] : undefined;
  };

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
          if (branch) onPick({ vessel: "lm", part: part.label, partFma: part.fma, branch });
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
      onPointerOut={() => onHover(null)}
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
function Thorax({ center }: { center: THREE.Vector3 }) {
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
      {parts.map((p) => {
        const m = materials[p.layer as keyof typeof materials] ?? materials.bone;
        // Skin draws last (it is the outermost translucent layer); nothing in the chest takes clicks.
        return <mesh key={p.key} geometry={p.geometry} material={m} renderOrder={p.layer === "skin" ? 3 : p.layer === "diaphragm" ? 2 : 0} raycast={() => null} />;
      })}
    </group>
  );
}

useGLTF.preload(ATLAS_HEART_URL, false, true);
