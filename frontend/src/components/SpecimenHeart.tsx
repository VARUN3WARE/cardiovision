import { useFrame, type ThreeEvent } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { acceleratedRaycast } from "three-mesh-bvh";
import AnatomyPins from "./AnatomyPins";
import ArteryTube, { type ArteryProbs } from "./ArteryTube";
import { NEUTRAL_ARTERY, riskColor, type Vessel } from "./heartAnatomy";
import { SPECIMEN_SCALE, SPECIMEN_URL, TERRITORY_STRIDE as S, prepareSpecimen } from "./specimenAnatomy";
import { VESSEL_IDS } from "../anatomy/registry";

const N = VESSEL_IDS.length; // index of the ventricle mask within each vertex's territory record

interface Props {
  probs: ArteryProbs | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  mode: "arteries" | "territories";
  pins: boolean;
  bpm: number;
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
  /** Called once the model is prepared, with camera directions facing each vessel's territory. */
  onReady: (views: Record<Vessel, THREE.Vector3>) => void;
}


/** The textured, sculpted heart model with the risk-coloured coronary tree fitted onto it. */
export default function SpecimenHeart({ probs, selected, hovered, mode, pins, bpm, onSelect, onHover, onReady }: Props) {
  const gltf = useGLTF(SPECIMEN_URL, false, false); // the asset uses no Draco/Meshopt compression
  const prep = useMemo(() => prepareSpecimen(gltf.scene), [gltf.scene]);
  const meshRef = useRef<THREE.Mesh>(null);
  const focus = selected ?? hovered;
  const beat = useRef<THREE.Group>(null);
  const phase = useRef(0);

  useLayoutEffect(() => {
    // The geometry already carries a BVH; use it for pointer and label-occlusion raycasts.
    if (meshRef.current) meshRef.current.raycast = acceleratedRaycast;
  }, [prep]);

  useEffect(() => {
    onReady(prep.views);
  }, [prep, onReady]);

  // Translucent overlay sharing the model's surface: tints perfusion territories without hiding
  // the texture underneath. RGBA vertex colours (alpha = tint strength).
  const overlay = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setIndex(prep.geometry.index);
    g.setAttribute("position", prep.geometry.getAttribute("position"));
    g.setAttribute("normal", prep.geometry.getAttribute("normal"));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(prep.geometry.getAttribute("position").count * 4), 4));
    return g;
  }, [prep]);
  const [tinted, setTinted] = useState(false);

  useEffect(() => {
    const color = overlay.getAttribute("color") as THREE.BufferAttribute;
    const T = prep.territory;
    const c = new THREE.Color();
    const tmp = new THREE.Color();
    let any = false;
    for (let i = 0; i < color.count; i++) {
      const mask = T[i * S + N];
      let alpha = 0;
      if (mask > 0 && probs && mode === "territories") {
        c.setRGB(0, 0, 0);
        VESSEL_IDS.forEach((t, k) => c.add(riskColor(probs[t], tmp).multiplyScalar(T[i * S + k])));
        alpha = 0.55 * mask;
      } else if (mask > 0 && focus) {
        // Highlight the myocardium supplied by the selected (or hovered) artery.
        c.copy(probs ? riskColor(probs[focus], tmp) : NEUTRAL_ARTERY);
        alpha = 0.42 * mask * Math.max(0, (T[i * S + VESSEL_IDS.indexOf(focus)] - 0.35) / 0.65);
      }
      if (alpha > 0.01) any = true;
      color.setXYZW(i, c.r, c.g, c.b, alpha);
    }
    color.needsUpdate = true;
    setTinted(any);
  }, [overlay, prep, probs, focus, mode]);

  useFrame((_, dt) => {
    phase.current = (phase.current + (dt * bpm) / 60) % 1;
    const t = phase.current;
    const pulse = (c: number, w: number) => (bpm > 0 ? Math.exp(-(((t - c) / w) ** 2)) : 0);
    const v = pulse(0.16, 0.07) + 0.25 * pulse(0.34, 0.05);
    beat.current?.scale.set(1 - 0.022 * v, 1 - 0.012 * v, 1 - 0.022 * v);
  });

  // Clicking the ventricular wall selects the artery that perfuses it; other anatomy just
  // swallows the click so arteries behind it can't be picked through it.
  const onWallClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (!e.face) return;
    const T = prep.territory;
    const i = e.face.a;
    if (T[i * S + N] < 0.5) return;
    const k = VESSEL_IDS.reduce((best, _, j) => (T[i * S + j] > T[i * S + best] ? j : best), 0);
    onSelect(VESSEL_IDS[k] === selected ? null : VESSEL_IDS[k]);
  };

  const center = prep.center;
  return (
    <group scale={SPECIMEN_SCALE}>
      <group ref={beat}>
        <group position={[-center.x, -center.y, -center.z]}>
          <mesh
            ref={meshRef}
            geometry={prep.geometry}
            material={prep.material}
            onClick={onWallClick}
            onPointerMove={(e) => e.stopPropagation()}
          />
          <mesh geometry={overlay} visible={tinted} raycast={() => null} renderOrder={1}>
            <meshStandardMaterial
              vertexColors
              transparent
              depthWrite={false}
              polygonOffset
              polygonOffsetFactor={-2}
              polygonOffsetUnits={-2}
              roughness={0.35}
              metalness={0}
            />
          </mesh>
          {prep.vessels.map(({ seg, curve, radiusAt, labelPos }) => (
            <ArteryTube
              key={seg.name}
              vessel={seg.vessel === "lm" ? null : seg.vessel}
              name={seg.name}
              curve={curve}
              radiusAt={radiusAt}
              labelPos={seg.label ? labelPos : null}
              probs={probs}
              selected={selected}
              hovered={hovered}
              occluders={[meshRef]}
              onSelect={onSelect}
              onHover={onHover}
            />
          ))}
          {pins && <AnatomyPins occluders={[meshRef]} positions={prep.pins} />}
        </group>
      </group>
    </group>
  );
}
