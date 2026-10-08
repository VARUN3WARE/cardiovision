import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Bvh, ContactShadows, Environment, Lightformer, OrbitControls } from "@react-three/drei";
import { Bloom, EffectComposer, N8AO, ToneMapping, Vignette } from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";
import { Suspense, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import * as THREE from "three";
import AnatomyPins from "./AnatomyPins";
import ArteryTube, { type ArteryProbs } from "./ArteryTube";
import ErrorBoundary from "./ErrorBoundary";
import SpecimenHeart from "./SpecimenHeart";
import { SPECIMEN_CREDIT, specimenAvailable } from "./specimenAnatomy";
import {
  ARTERIES, ATRIA, FAT_DARK, FAT_LIGHT, GREAT_VESSELS, MUSCLE_DARK, MUSCLE_LIGHT, NEUTRAL_ARTERY,
  VEINS, VEIN_COLOR, buildChamber, buildMyocardium, makeMuscleNormalMap, makeTissueNormalMap,
  riskColor, surfacePoint, taperedTube, territoryWeights, vesselCurve, vesselRadius,
  type Chamber, type Vessel, type VesselPath,
} from "./heartAnatomy";

type HeartModel = "specimen" | "schematic";

// Anatomical orientation of the heart in the chest: apex points left, anterior and inferior.
const HEART_ROTATION = new THREE.Euler(-0.25, -0.35, 0.62);
const DEFAULT_VIEW = new THREE.Vector3(0, 0.015, 1);

/**
 * Camera direction facing each vessel's perfusion territory: territory centre on the surface (u, v)
 * plus a camera elevation (RCA/PDA run on the diaphragmatic surface, so view it from below).
 */
const VESSEL_VIEWS: Record<Vessel, THREE.Vector3> = Object.fromEntries(
  ([["lad", 0.5, 1.3, 0.2], ["lcx", 0.65, -0.1, 0.15], ["rca", 0.55, 4.1, -0.45]] as const).map(
    ([vessel, u, v, elevation]) => {
      const p = surfacePoint(u, v).applyEuler(HEART_ROTATION);
      p.y = 0;
      p.normalize().setY(elevation);
      return [vessel, p.normalize()];
    },
  ),
) as Record<Vessel, THREE.Vector3>;

// Generated once and shared (no image downloads).
let textures: { muscle: THREE.Texture; tissue: THREE.Texture } | null = null;
function getTextures() {
  textures ??= { muscle: makeMuscleNormalMap(), tissue: makeTissueNormalMap() };
  return textures;
}

export type { ArteryProbs, Vessel };

export const VESSEL_INFO: Record<Vessel, { name: string; territory: string }> = {
  lad: {
    name: "Left Anterior Descending",
    territory: "Anterior LV wall, anterior two-thirds of the septum, apex",
  },
  lcx: {
    name: "Left Circumflex",
    territory: "Lateral and posterolateral LV wall",
  },
  rca: {
    name: "Right Coronary Artery",
    territory: "Right ventricle, inferior LV wall, posterior septum (via PDA)",
  },
};

type ViewMode = "arteries" | "territories";

interface HeartSceneProps {
  probs: ArteryProbs | null;
  selected: Vessel | null;
  onSelect: (vessel: Vessel | null) => void;
  bpm: number;
}

export default function HeartScene({ probs, selected, onSelect, bpm }: HeartSceneProps) {
  const [mode, setMode] = useState<ViewMode>("arteries");
  const [beating, setBeating] = useState(true);
  const [fx, setFx] = useState(true);
  const [pins, setPins] = useState(false);
  const [hovered, setHovered] = useState<Vessel | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [framesReady, setFramesReady] = useState(false);
  // Realistic model when its file is present and loads; the schematic heart otherwise.
  const [model, setModel] = useState<HeartModel>("specimen");
  const [available, setAvailable] = useState<boolean | null>(null);
  const [specimenFailed, setSpecimenFailed] = useState(false);
  const [specimenViews, setSpecimenViews] = useState<Record<Vessel, THREE.Vector3> | null>(null);

  useEffect(() => {
    specimenAvailable().then(setAvailable);
  }, []);
  const useSpecimen = available === true && !specimenFailed && model === "specimen";
  const ready = framesReady && available !== null && (!useSpecimen || specimenViews !== null);

  useEffect(() => {
    document.body.style.cursor = hovered ? "pointer" : "auto";
    return () => {
      document.body.style.cursor = "auto";
    };
  }, [hovered]);

  return (
    <div className="relative h-full w-full overflow-hidden rounded">
      <Canvas
        camera={{ position: [0, 0.1, 6.8], fov: 40 }}
        dpr={[1, 2]}
        gl={{ antialias: !fx }}
        onPointerMissed={() => onSelect(null)}
      >
        <color attach="background" args={["#141a26"]} />
        <Lighting />

        {/* BVH keeps pointer and label-occlusion raycasts cheap on the dense surface mesh. */}
        <Bvh firstHitOnly>
          {available === null ? null : useSpecimen ? (
            <ErrorBoundary fallback={null} onError={() => setSpecimenFailed(true)}>
              <Suspense fallback={null}>
                <SpecimenHeart
                  probs={probs}
                  selected={selected}
                  hovered={hovered}
                  mode={mode}
                  pins={pins}
                  bpm={beating ? bpm : 0}
                  onSelect={onSelect}
                  onHover={setHovered}
                  onReady={setSpecimenViews}
                />
              </Suspense>
            </ErrorBoundary>
          ) : (
            <Heart
              probs={probs}
              selected={selected}
              hovered={hovered}
              mode={mode}
              pins={pins}
              bpm={beating ? bpm : 0}
              onSelect={onSelect}
              onHover={setHovered}
            />
          )}
        </Bvh>

        <ContactShadows position={[0, -1.85, 0]} opacity={0.6} scale={7} blur={2.6} far={3.5} resolution={512} />

        {fx && (
          <EffectComposer multisampling={4}>
            <N8AO aoRadius={0.3} distanceFalloff={0.8} intensity={2.2} quality="medium" halfRes />
            <Bloom mipmapBlur luminanceThreshold={0.9} luminanceSmoothing={0.2} intensity={0.8} radius={0.6} />
            <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
            <Vignette offset={0.3} darkness={0.7} />
          </EffectComposer>
        )}

        <FirstFrame onReady={() => setFramesReady(true)} />
        <FitCamera />
        <CameraRig focus={selected} resetKey={resetKey} views={useSpecimen && specimenViews ? specimenViews : VESSEL_VIEWS} />
        <OrbitControls enablePan={false} minDistance={3} maxDistance={12} makeDefault />
      </Canvas>

      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#141a26] text-xs text-gray-400">
          <span className="animate-pulse">Building 3D heart model…</span>
        </div>
      )}

      {/* View controls */}
      <div className="absolute left-3 top-3 flex flex-wrap gap-1 text-xs">
        {available && !specimenFailed && (
          <div className="mr-1 flex overflow-hidden rounded" role="radiogroup" aria-label="Heart model">
            {(["specimen", "schematic"] as const).map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={model === m}
                onClick={() => setModel(m)}
                title={m === "specimen" ? "Sculpted, textured heart model" : "Schematic heart with epicardial fat and cardiac veins"}
                className={`px-2 py-1 ${model === m ? "bg-slate-200 text-slate-900" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
              >
                {m === "specimen" ? "Realistic" : "Schematic"}
              </button>
            ))}
          </div>
        )}
        {(["arteries", "territories"] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded px-2 py-1 ${mode === m ? "bg-red-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
          >
            {m === "arteries" ? "Arteries" : "Territories"}
          </button>
        ))}
        <button
          onClick={() => setPins((p) => !p)}
          className={`rounded px-2 py-1 ${pins ? "bg-sky-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
          title="Show numbered anatomy annotations"
        >
          Anatomy
        </button>
        <button
          onClick={() => setBeating((b) => !b)}
          className="rounded bg-slate-900/80 px-2 py-1 text-gray-300 hover:bg-slate-700"
          title="Toggle heartbeat animation (rate = patient pulse)"
        >
          {beating ? `♥ ${bpm} bpm` : "♡ paused"}
        </button>
        <button
          onClick={() => setFx((f) => !f)}
          className="rounded bg-slate-900/80 px-2 py-1 text-gray-300 hover:bg-slate-700"
          title="Ambient occlusion, bloom and vignette. Turn off on slow GPUs."
        >
          {fx ? "HD on" : "HD off"}
        </button>
        <button
          onClick={() => {
            onSelect(null);
            setResetKey((k) => k + 1);
          }}
          className="rounded bg-slate-900/80 px-2 py-1 text-gray-300 hover:bg-slate-700"
        >
          Reset
        </button>
      </div>

      {/* Selected vessel info */}
      {selected && (
        <div className="absolute right-3 top-12 max-w-[240px] rounded bg-slate-900/90 p-3 text-xs">
          <div className="font-semibold text-white">
            {selected.toUpperCase()} · {VESSEL_INFO[selected].name}
          </div>
          <div className="mt-1 text-gray-400">Supplies: {VESSEL_INFO[selected].territory}</div>
          {probs && (
            <div className="mt-1 text-gray-200">
              Predicted stenosis probability: <b>{(probs[selected] * 100).toFixed(1)}%</b>
            </div>
          )}
        </div>
      )}

      {useSpecimen && (
        <a
          href={SPECIMEN_CREDIT.url}
          target="_blank"
          rel="noreferrer"
          className="absolute bottom-3 right-3 max-w-[220px] text-right text-[9px] leading-tight text-gray-500 hover:text-gray-300"
        >
          Heart model: “{SPECIMEN_CREDIT.title}” by {SPECIMEN_CREDIT.author}, {SPECIMEN_CREDIT.license}
        </a>
      )}

      {/* Legend */}
      <div className="absolute bottom-3 left-3 w-56 text-[10px] text-gray-400">
        <div className="mb-2 text-gray-500">Drag to rotate · scroll to zoom · click an artery or heart wall</div>
        <div className="mb-1">
          {probs ? "Predicted stenosis probability" : "Run a prediction to color the arteries"}
        </div>
        <div className="h-2 rounded bg-gradient-to-r from-green-500 via-amber-500 to-red-500" />
        <div className="mt-0.5 flex justify-between">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    </div>
  );
}

/** Studio lighting: soft boxes baked into a local environment map, plus a key light. */
function Lighting() {
  return (
    <>
      <ambientLight intensity={0.15} />
      <directionalLight position={[3, 4, 5]} intensity={1.4} color="#fff3ea" />
      <directionalLight position={[-4, 1, -3]} intensity={0.9} color="#ff8a78" />
      <Environment resolution={256} frames={1}>
        {/* key softbox, top front */}
        <Lightformer form="rect" intensity={2.5} position={[0, 5, 4]} scale={[8, 3, 1]} target={[0, 0, 0]} />
        {/* warm fill from the left */}
        <Lightformer form="rect" intensity={1.2} color="#ffd8c8" position={[-6, 1, 2]} scale={[3, 7, 1]} target={[0, 0, 0]} />
        {/* reddish rim from behind (reads as light scattering through tissue) */}
        <Lightformer form="rect" intensity={2.2} color="#ff7a68" position={[4, 2, -6]} scale={[5, 5, 1]} target={[0, 0, 0]} />
        {/* dim warm bounce from below */}
        <Lightformer form="circle" intensity={0.4} color="#ffb59a" position={[0, -6, 1]} scale={4} target={[0, 0, 0]} />
      </Environment>
    </>
  );
}

/** Reports once the first frame (and so shader compilation) has completed. */
function FirstFrame({ onReady }: { onReady: () => void }) {
  const frames = useRef(0);
  useFrame(() => {
    frames.current += 1;
    if (frames.current === 2) onReady();
  });
  return null;
}

/** Pull the camera back on narrow canvases so the whole heart stays in frame. */
function FitCamera() {
  const { camera, size } = useThree();
  useEffect(() => {
    if (!size.width || !size.height) return; // first layout pass can report 0
    const aspect = size.width / size.height;
    camera.position.setLength(6.8 * Math.min(1.4, Math.max(1, 0.8 / aspect)));
  }, [camera, size.width, size.height]);
  return null;
}

/** Swing the camera round to face the selected vessel's territory, then hand control back. */
function CameraRig({ focus, resetKey, views }: {
  focus: Vessel | null; resetKey: number; views: Record<Vessel, THREE.Vector3>;
}) {
  const { camera, controls } = useThree();
  const target = useRef<THREE.Vector3 | null>(null);

  useEffect(() => {
    if (focus) target.current = views[focus].clone();
  }, [focus, views]);
  useEffect(() => {
    if (resetKey) target.current = DEFAULT_VIEW.clone().normalize();
  }, [resetKey]);

  useFrame((_, dt) => {
    if (!target.current) return;
    const dist = camera.position.length();
    const dir = camera.position.clone().normalize();
    dir.lerp(target.current, Math.min(1, dt * 4)).normalize();
    camera.position.copy(dir.multiplyScalar(dist));
    (controls as unknown as { update?: () => void } | null)?.update?.();
    if (dir.angleTo(target.current) < 0.01) target.current = null;
  });
  return null;
}

interface HeartProps {
  probs: ArteryProbs | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  mode: ViewMode;
  pins: boolean;
  bpm: number;
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
}

function Heart({ probs, selected, hovered, mode, pins, bpm, onSelect, onHover }: HeartProps) {
  const ventricles = useRef<THREE.Group>(null);
  const atria = useRef<THREE.Group>(null);
  const outflow = useRef<THREE.Group>(null);
  const myocardium = useRef<THREE.Mesh>(null);
  const phase = useRef(0);

  useFrame((_, dt) => {
    phase.current = (phase.current + (dt * bpm) / 60) % 1;
    const t = phase.current;
    const pulse = (c: number, w: number) => (bpm > 0 ? Math.exp(-(((t - c) / w) ** 2)) : 0);
    // Atrial kick first, then ventricular systole (which also distends the great arteries).
    const a = pulse(0.02, 0.05);
    const v = pulse(0.16, 0.07) + 0.25 * pulse(0.34, 0.05);
    ventricles.current?.scale.set(1 - 0.035 * v, 1 - 0.018 * v, 1 - 0.035 * v);
    atria.current?.scale.setScalar(1 - 0.04 * a + 0.015 * v);
    outflow.current?.scale.setScalar(1 + 0.008 * v);
  });

  return (
    <group rotation={HEART_ROTATION} position={[0.1, -0.2, 0]}>
      {pins && <AnatomyPins occluders={[myocardium, outflow, atria]} />}
      {/* Opaque anatomy swallows pointer events so arteries behind it can't be picked through it. */}
      <group ref={outflow} onPointerMove={stop} onClick={stop}>
        <GreatVessels />
      </group>
      <group ref={atria} position={[0, 0.7, 0]} onPointerMove={stop} onClick={stop}>
        <group position={[0, -0.7, 0]}>
          <Atria />
        </group>
      </group>
      <group ref={ventricles}>
        <Myocardium meshRef={myocardium} probs={probs} selected={selected} mode={mode} onSelect={onSelect} />
        <Veins />
        {ARTERIES.map((seg) => (
          <Artery
            key={seg.name}
            seg={seg}
            probs={probs}
            selected={selected}
            hovered={hovered}
            occluder={myocardium}
            onSelect={onSelect}
            onHover={onHover}
          />
        ))}
      </group>
    </group>
  );
}

const stop = (e: ThreeEvent<PointerEvent | MouseEvent>) => e.stopPropagation();

interface MyocardiumProps {
  meshRef: RefObject<THREE.Mesh>;
  probs: ArteryProbs | null;
  selected: Vessel | null;
  mode: ViewMode;
  onSelect: (v: Vessel | null) => void;
}

function Myocardium({ meshRef, probs, selected, mode, onSelect }: MyocardiumProps) {
  const geometry = useMemo(() => buildMyocardium(), []);
  const { muscle } = getTextures();

  useEffect(() => {
    const param = geometry.getAttribute("param");
    const fat = geometry.getAttribute("fat");
    const shade = geometry.getAttribute("shade");
    const color = geometry.getAttribute("color") as THREE.BufferAttribute;
    const c = new THREE.Color();
    const fatC = new THREE.Color();
    const tint = new THREE.Color();
    const tmp = new THREE.Color();
    for (let i = 0; i < param.count; i++) {
      const s = shade.getX(i);
      const f = fat.getX(i);
      c.copy(MUSCLE_DARK).lerp(MUSCLE_LIGHT, s);
      fatC.copy(FAT_DARK).lerp(FAT_LIGHT, s);
      c.lerp(fatC, f);
      const w = territoryWeights(param.getX(i), param.getY(i));
      if (probs && mode === "territories") {
        tint.setRGB(0, 0, 0);
        for (const v of ["lad", "lcx", "rca"] as const) {
          tint.add(riskColor(probs[v], tmp).multiplyScalar(w[v]));
        }
        c.lerp(tint, 0.62 - 0.25 * f);
      } else if (selected) {
        // Highlight the myocardium supplied by the selected artery.
        tint.copy(probs ? riskColor(probs[selected], tmp) : NEUTRAL_ARTERY);
        c.lerp(tint, 0.55 * Math.max(0, (w[selected] - 0.35) / 0.65));
      }
      color.setXYZ(i, c.r, c.g, c.b);
    }
    color.needsUpdate = true;
  }, [geometry, probs, selected, mode]);

  // Clicking the heart wall selects the artery that perfuses that region.
  const selectTerritory = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (!e.face) return;
    const param = geometry.getAttribute("param");
    const w = territoryWeights(param.getX(e.face.a), param.getY(e.face.a));
    const vessel = (Object.keys(w) as Vessel[]).reduce((a, b) => (w[a] >= w[b] ? a : b));
    onSelect(vessel === selected ? null : vessel);
  };

  return (
    <mesh ref={meshRef} geometry={geometry} onClick={selectTerritory} onPointerMove={stop}>
      <meshPhysicalMaterial
        vertexColors
        roughness={0.42}
        clearcoat={1}
        clearcoatRoughness={0.2}
        sheen={1}
        sheenRoughness={0.45}
        sheenColor="#ff5a48"
        normalMap={muscle}
        normalScale={new THREE.Vector2(0.22, 0.22)}
        envMapIntensity={0.9}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function Atria() {
  const { tissue } = getTextures();
  return (
    <>
      {ATRIA.map((a, i) => (
        <Chamber key={a.name} chamber={a} seed={i * 17.3} normalMap={tissue} />
      ))}
    </>
  );
}

function Chamber({ chamber, seed, normalMap }: { chamber: Chamber; seed: number; normalMap: THREE.Texture }) {
  const geometry = useMemo(() => buildChamber(chamber.crinkle, seed), [chamber.crinkle, seed]);
  return (
    <mesh
      geometry={geometry}
      position={chamber.center as unknown as THREE.Vector3Tuple}
      scale={chamber.radii as unknown as THREE.Vector3Tuple}
      rotation={(chamber.rotation ?? [0, 0, 0]) as unknown as THREE.Vector3Tuple}
    >
      <meshPhysicalMaterial
        color="#7a2f2b"
        roughness={0.45}
        clearcoat={0.9}
        clearcoatRoughness={0.25}
        sheen={0.8}
        sheenRoughness={0.5}
        sheenColor="#ff6650"
        normalMap={normalMap}
        normalScale={new THREE.Vector2(0.5, 0.5)}
      />
    </mesh>
  );
}

function GreatVessels() {
  const { tissue } = getTextures();
  const tubes = useMemo(
    () =>
      Object.entries(GREAT_VESSELS).map(([name, def]) => {
        const curve = new THREE.CatmullRomCurve3(def.points.map((p) => new THREE.Vector3(...p)));
        const profile = def.profile ?? (() => 1);
        return {
          name,
          // Open-ended: the cut ends show the vessel lumen, as on a dissected specimen.
          geometry: taperedTube(curve, (t) => def.radius * profile(t), 96, 32),
          // Natural tissue tones (as on a specimen), not textbook red/blue: arterial walls
          // are pinkish-red, the thin-walled veins a darker maroon.
          color: name === "pulmonary" || name.endsWith("PA")
            ? "#c86a5e"
            : def.deoxygenated || name.endsWith("pv") ? "#7e3438" : "#cf7465",
        };
      }),
    [],
  );
  return (
    <>
      {tubes.map((t) => (
        <mesh key={t.name} geometry={t.geometry}>
          <meshPhysicalMaterial
            color={t.color}
            roughness={0.4}
            clearcoat={0.8}
            clearcoatRoughness={0.25}
            sheen={0.6}
            sheenColor="#ffb0a0"
            normalMap={tissue}
            normalScale={new THREE.Vector2(0.3, 0.3)}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </>
  );
}

/** Cardiac veins: anatomical context only, so they ignore pointer events. */
function Veins() {
  const geometries = useMemo(
    () => VEINS.map((seg) => taperedTube(vesselCurve(seg), vesselRadius(seg), 140, 10)),
    [],
  );
  return (
    <>
      {geometries.map((g, i) => (
        <mesh key={VEINS[i].name} geometry={g} raycast={() => null}>
          <meshPhysicalMaterial color={VEIN_COLOR} roughness={0.32} clearcoat={0.9} clearcoatRoughness={0.2} />
        </mesh>
      ))}
    </>
  );
}

interface ArteryProps {
  seg: VesselPath;
  probs: ArteryProbs | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  occluder: RefObject<THREE.Mesh>;
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
}

/** A schematic-heart artery: projects the (u, v) path onto the analytic surface. */
function Artery({ seg, occluder, ...rest }: ArteryProps) {
  const { curve, radiusAt, labelPos } = useMemo(() => {
    const curve = vesselCurve(seg);
    // Lift the label off the vessel so it isn't occluded by its own tube.
    const p = curve.getPoint(0.3);
    const labelPos = p.clone().add(new THREE.Vector3(p.x, 0, p.z).normalize().multiplyScalar(0.18));
    return { curve, radiusAt: vesselRadius(seg), labelPos };
  }, [seg]);
  return (
    <ArteryTube
      vessel={seg.vessel === "lm" || seg.vessel === "vein" ? null : seg.vessel}
      name={seg.name}
      curve={curve}
      radiusAt={radiusAt}
      labelPos={seg.label ? labelPos : null}
      occluders={[occluder]}
      {...rest}
    />
  );
}
