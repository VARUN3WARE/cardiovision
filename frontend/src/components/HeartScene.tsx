import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Bvh, ContactShadows, Environment, Lightformer, OrbitControls, PerformanceMonitor } from "@react-three/drei";
import { Bloom, EffectComposer, N8AO, ToneMapping, Vignette } from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";
import { Suspense, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import * as THREE from "three";
import AnatomyPins from "./AnatomyPins";
import ArteryTube, { type ArteryProbs } from "./ArteryTube";
import AtlasHeart, { type AtlasPick, type ChestLayers } from "./AtlasHeart";
import ErrorBoundary from "./ErrorBoundary";
import VesselInspector from "./VesselInspector";
import { VESSEL_IDS } from "../anatomy/registry";
import type { ExplainResponse, MetricsResponse } from "../api/client";
import SpecimenHeart from "./SpecimenHeart";
import { BAND_COLOR, formatPct, getVesselCuts, vesselBand, type Band } from "../risk";
import { ATLAS_CREDIT, atlasAvailable } from "./atlasAnatomy";
import { SPECIMEN_CREDIT, specimenAvailable } from "./specimenAnatomy";
import {
  ARTERIES, ATRIA, FAT_DARK, FAT_LIGHT, GREAT_VESSELS, MUSCLE_DARK, MUSCLE_LIGHT, NEUTRAL_ARTERY,
  VEINS, VEIN_COLOR, buildChamber, buildMyocardium, makeMuscleNormalMap, makeTissueNormalMap,
  riskColor, surfacePoint, taperedTube, territoryWeights, vesselCurve, vesselRadius,
  type Chamber, type Vessel, type VesselPath,
} from "./heartAnatomy";

type HeartModel = "specimen" | "atlas" | "schematic";

// Anatomical orientation of the heart in the chest: apex points left, anterior and inferior.
const HEART_ROTATION = new THREE.Euler(-0.25, -0.35, 0.62);
const DEFAULT_VIEW = new THREE.Vector3(0, 0.015, 1);
const HEART_DISTANCE = 6.8;
const CHEST_DISTANCE = 21;

/** Standard anatomical views. World frame: +X patient's left, +Y superior, +Z anterior. */
const ANATOMICAL_VIEWS: { key: string; title: string; dir: [number, number, number] }[] = [
  { key: "A", title: "Anterior (front)", dir: [0, 0.015, 1] },
  { key: "P", title: "Posterior (back)", dir: [0, 0.015, -1] },
  { key: "L", title: "Patient's left", dir: [1, 0.015, 0] },
  { key: "R", title: "Patient's right", dir: [-1, 0.015, 0] },
  { key: "S", title: "Superior (from above)", dir: [0, 1, 0.02] },
  { key: "I", title: "Inferior (from below)", dir: [0, -1, 0.02] },
];

const MODEL_INFO: Record<HeartModel, { label: string; title: string }> = {
  specimen: { label: "Realistic", title: "Sculpted, textured heart model" },
  atlas: {
    label: "Atlas",
    title: "BodyParts3D anatomical atlas in the chest: every coronary branch is a named FMA structure",
  },
  schematic: { label: "Schematic", title: "Schematic heart with epicardial fat and cardiac veins" },
};

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

type ViewMode = "arteries" | "territories";

interface HeartSceneProps {
  probs: ArteryProbs | null;
  /** Inputs changed since these estimates were computed. */
  pending?: boolean;
  /** Full results and validation metrics, for the vessel inspector. */
  result?: ExplainResponse | null;
  metrics?: MetricsResponse | null;
  /** Artery highlighted from outside the scene (e.g. hovering the bullseye). */
  highlight?: Vessel | null;
  selected: Vessel | null;
  onSelect: (vessel: Vessel | null) => void;
  bpm: number;
}

export default function HeartScene({
  probs, pending = false, result = null, metrics = null, highlight = null, selected, onSelect, bpm,
}: HeartSceneProps) {
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
  const [atlasOk, setAtlasOk] = useState<boolean | null>(null);
  const [atlasFailed, setAtlasFailed] = useState(false);
  const [atlasViews, setAtlasViews] = useState<Record<Vessel, THREE.Vector3> | null>(null);
  const [chest, setChest] = useState(true);
  const [layers, setLayers] = useState<ChestLayers>({ skin: true, skeleton: true, organs: true });
  const [layerMenu, setLayerMenu] = useState(false);
  const [xray, setXray] = useState(false);
  const [pick, setPick] = useState<AtlasPick | null>(null);
  const [view, setView] = useState<{ dir: THREE.Vector3; n: number } | null>(null);
  const [fly, setFly] = useState<{ from: number; to: number; n: number } | null>(null);

  useEffect(() => {
    specimenAvailable().then(setAvailable);
    atlasAvailable().then(setAtlasOk);
  }, []);
  const useSpecimen = available === true && !specimenFailed && model === "specimen";
  const useAtlas = atlasOk === true && !atlasFailed && model === "atlas";
  const showChest = useAtlas && chest;
  const ready =
    framesReady && available !== null && atlasOk !== null &&
    (!useSpecimen || specimenViews !== null) && (!useAtlas || atlasViews !== null);
  const models = (["specimen", "atlas", "schematic"] as const).filter(
    (m) => m === "schematic" || (m === "specimen" ? available && !specimenFailed : atlasOk && !atlasFailed),
  );

  const chooseModel = (m: HeartModel) => {
    setModel(m);
    setPick(null);
    // Entering the atlas with the chest shown: start outside the chest and fly in to the heart.
    if (m === "atlas" && chest && !reducedMotion()) setFly((f) => ({ from: CHEST_DISTANCE, to: HEART_DISTANCE, n: (f?.n ?? 0) + 1 }));
  };
  const showPick = pick && (pick.vessel === "lm" || pick.vessel === selected) ? pick : null;
  const rwma = useMemo(() => {
    const exp = result?.explanations.cad;
    const i = exp ? exp.feature_names.indexOf("Region RWMA") : -1;
    return exp && i >= 0 ? exp.feature_values[i] : null;
  }, [result]);

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
        dpr={[1, 1.5]}
        gl={{ antialias: !fx }}
        onPointerMissed={() => {
          onSelect(null);
          setPick(null);
        }}
      >
        <color attach="background" args={["#141a26"]} />
        <Lighting />

        {/* BVH keeps pointer and label-occlusion raycasts cheap on the dense surface mesh. */}
        <Bvh firstHitOnly>
          {available === null || atlasOk === null ? null : useAtlas ? (
            <ErrorBoundary fallback={null} onError={() => setAtlasFailed(true)}>
              <Suspense fallback={null}>
                <AtlasHeart
                  probs={probs}
                  selected={selected}
                  hovered={hovered ?? highlight}
                  mode={mode}
                  xray={xray}
                  chest={chest ? layers : null}
                  bpm={beating ? bpm : 0}
                  onSelect={onSelect}
                  onHover={setHovered}
                  onPick={setPick}
                  onReady={setAtlasViews}
                />
              </Suspense>
            </ErrorBoundary>
          ) : useSpecimen ? (
            <ErrorBoundary fallback={null} onError={() => setSpecimenFailed(true)}>
              <Suspense fallback={null}>
                <SpecimenHeart
                  probs={probs}
                  selected={selected}
                  hovered={hovered ?? highlight}
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
              hovered={hovered ?? highlight}
              mode={mode}
              pins={pins}
              bpm={beating ? bpm : 0}
              onSelect={onSelect}
              onHover={setHovered}
            />
          )}
        </Bvh>

        {!showChest && (
          // Rendered once per model (the heartbeat barely moves the shadow), not every frame.
          <ContactShadows key={`${model}-${ready}`} frames={1} position={[0, -1.85, 0]} opacity={0.6} scale={7} blur={2.6} far={3.5} resolution={512} />
        )}

        {/* Turn HD off by itself if frames stay below ~24 fps (ignored while the page is hidden). */}
        <PerformanceMonitor
          bounds={() => [24, 1000]}
          flipflops={2}
          onDecline={() => {
            if (document.visibilityState === "visible") setFx(false);
          }}
        />

        {fx && (
          <EffectComposer multisampling={4}>
            <N8AO aoRadius={0.3} distanceFalloff={0.8} intensity={2.2} quality="medium" halfRes />
            <Bloom mipmapBlur luminanceThreshold={0.9} luminanceSmoothing={0.2} intensity={0.8} radius={0.6} />
            <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
            <Vignette offset={0.3} darkness={0.7} />
          </EffectComposer>
        )}

        <FirstFrame onReady={() => setFramesReady(true)} />
        {import.meta.env.DEV && <DevHook />}
        <FitCamera />
        <CameraRig
          focus={selected}
          resetKey={resetKey}
          view={view}
          fly={fly}
          views={useAtlas && atlasViews ? atlasViews : useSpecimen && specimenViews ? specimenViews : VESSEL_VIEWS}
        />
        <OrbitControls enablePan={false} minDistance={3} maxDistance={showChest ? 26 : 12} makeDefault />
      </Canvas>

      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#141a26] text-xs text-gray-400">
          <span className="animate-pulse">Building 3D heart model…</span>
        </div>
      )}

      {/* View controls */}
      <div className="absolute left-3 top-3 z-20 flex flex-wrap gap-1 text-xs">
        {models.length > 1 && (
          <div className="mr-1 flex overflow-hidden rounded" role="radiogroup" aria-label="Heart model">
            {models.map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={model === m}
                onClick={() => chooseModel(m)}
                title={MODEL_INFO[m].title}
                className={`px-2 py-1 ${model === m ? "bg-slate-200 text-slate-900" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
              >
                {MODEL_INFO[m].label}
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
        {useAtlas && (
          <>
            <button
              onClick={() => {
                if (!chest && !reducedMotion()) setFly((f) => ({ from: CHEST_DISTANCE, to: HEART_DISTANCE, n: (f?.n ?? 0) + 1 }));
                setChest((c) => !c);
              }}
              aria-pressed={chest}
              className={`rounded px-2 py-1 ${chest ? "bg-sky-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
              title="Show the heart in the chest (BodyParts3D skin, skeleton, diaphragm, airway) with a cutaway window"
            >
              Chest
            </button>
            <div className="relative -ml-1">
              <button
                onClick={() => setLayerMenu((m) => !m)}
                aria-expanded={layerMenu}
                aria-label="Chest layers"
                title="Choose which chest layers to show"
                className="rounded bg-slate-900/80 px-1.5 py-1 text-gray-300 hover:bg-slate-700"
              >
                ▾
              </button>
              {layerMenu && (
                <div
                  className="absolute left-0 top-7 z-20 w-40 rounded bg-slate-900/95 p-2 shadow-lg"
                  onMouseLeave={() => setLayerMenu(false)}
                >
                  {([["skin", "Skin"], ["skeleton", "Skeleton"], ["organs", "Diaphragm & airway"]] as const).map(([k, label]) => (
                    <label key={k} className="flex cursor-pointer items-center gap-2 py-0.5 text-gray-200">
                      <input
                        type="checkbox"
                        checked={layers[k]}
                        onChange={() => {
                          setLayers((l) => ({ ...l, [k]: !l[k] }));
                          setChest(true);
                        }}
                      />
                      {label}
                    </label>
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={() => setXray((x) => !x)}
              aria-pressed={xray}
              className={`rounded px-2 py-1 ${xray ? "bg-sky-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
              title="See-through heart wall: shows the vessels on the back of the heart from the front"
            >
              X-ray
            </button>
          </>
        )}
        {!useAtlas && (
        <button
          onClick={() => setPins((p) => !p)}
          className={`rounded px-2 py-1 ${pins ? "bg-sky-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
          title="Show numbered anatomy annotations"
        >
          Anatomy
        </button>
        )}
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

      {pending && probs && (
        <div role="status" className="absolute left-1/2 top-12 -translate-x-1/2 rounded bg-amber-950/90 px-2 py-1 text-[10px] text-amber-200">
          Inputs changed: colours show the previous estimate
        </div>
      )}

      {/* Anatomical view buttons */}
      <div className="absolute bottom-12 right-3 grid grid-cols-3 gap-0.5 text-[10px]" role="group" aria-label="Standard views">
        {ANATOMICAL_VIEWS.map((v) => (
          <button
            key={v.key}
            title={v.title}
            onClick={() => setView((cur) => ({ dir: new THREE.Vector3(...v.dir).normalize(), n: (cur?.n ?? 0) + 1 }))}
            className="h-6 w-6 rounded bg-slate-900/80 font-semibold text-gray-300 hover:bg-slate-700"
          >
            {v.key}
          </button>
        ))}
      </div>

      {/* Selected vessel: inspector card; a clicked left-main branch gets a short note instead. */}
      {selected ? (
        <VesselInspector
          vessel={selected}
          result={result}
          metrics={metrics}
          pick={showPick}
          onClose={() => {
            onSelect(null);
            setPick(null);
          }}
        />
      ) : showPick?.vessel === "lm" && (
        <div className="absolute right-3 top-12 max-w-[260px] rounded bg-slate-900/90 p-3 text-xs text-gray-300">
          <div className="text-white first-letter:uppercase">{showPick.branch.name}</div>
          <div className="text-gray-500">{showPick.branch.fma} · BodyParts3D {showPick.branch.fj}</div>
          <div className="mt-1 text-gray-400">Left main: not labelled in this dataset, so no model output; shown neutral.</div>
        </div>
      )}

      {useAtlas && (
        <a
          href={ATLAS_CREDIT.url}
          target="_blank"
          rel="noreferrer"
          className="absolute bottom-3 right-3 max-w-[240px] text-right text-[9px] leading-tight text-gray-500 hover:text-gray-300"
        >
          Reference anatomy, not this patient&apos;s imaging. Right dominance assumed. {ATLAS_CREDIT.text}
        </a>
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
      <div className="absolute bottom-3 left-3 z-20 w-60 text-[10px] text-gray-400">
        <MapSemantics rwma={rwma} />
        {mode === "territories" && (
          <div className="mb-2 rounded bg-slate-900/80 p-1.5 text-gray-300">
            Muscle shaded by the artery that usually supplies it (standard anatomy, right dominance assumed).
            Illustrative: not a perfusion or wall-motion map.
          </div>
        )}
        <div className="mb-2 text-gray-500">Drag to rotate · scroll to zoom · click an artery or heart wall</div>
        <div className="mb-1 text-gray-300">
          {probs ? "Chance of ≥50% narrowing in each artery (not % blocked)" : "Run a prediction to colour the arteries"}
        </div>
        <BandLegend />
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

/** Development only: exposes the three.js state as window.__cv for scripted checks. */
function DevHook() {
  const state = useThree();
  useEffect(() => {
    (window as unknown as { __cv?: unknown }).__cv = state;
  }, [state]);
  return null;
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

/**
 * What the map does and does not show (the brief: clinical and ECG findings are not anatomical
 * coordinates, and wall-motion data is not a lesion map). Always visible, with the rules on demand.
 */
function MapSemantics({ rwma }: { rwma: number | null }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-2 space-y-1">
      <div className="flex items-start gap-1 rounded bg-slate-900/80 p-1.5 text-gray-300">
        <span>Vessel-level estimates: one colour per artery. Position along the artery is not predicted.</span>
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-label="What this map shows"
          className="shrink-0 rounded-full border border-gray-500 px-1 leading-tight text-gray-300 hover:bg-slate-700"
        >
          i
        </button>
      </div>
      {open && (
        <ol className="list-decimal space-y-0.5 rounded bg-slate-900/95 p-2 pl-5 text-gray-300">
          <li>Each label is one model&apos;s chance of ≥50% narrowing somewhere in that artery.</li>
          <li>Each artery tree has a single colour; nothing varies along it.</li>
          <li>Branches take their parent artery&apos;s colour (an assumption: they have no labels of their own).</li>
          <li>Territories assume standard anatomy with right dominance.</li>
          <li>Findings such as ECG changes or wall motion are model inputs, not locations on the heart.</li>
          <li>The anatomy is a reference model, not this patient&apos;s imaging.</li>
        </ol>
      )}
      {rwma !== null && rwma > 0 && (
        <div className="rounded bg-slate-900/80 p-1.5 text-gray-300">
          Wall-motion abnormality score {rwma}: location not recorded in the data, so not drawn on the heart.
        </div>
      )}
    </div>
  );
}

/** Three band swatches with their ranges; the middle (undecided) band is striped, as in 3D. */
function BandLegend() {
  const [lo, hi] = getVesselCuts();
  const bands: { band: Band; label: string; range: string }[] = [
    { band: "Lower", label: "lower", range: `<${Math.round(lo * 100)}%` },
    { band: "Intermediate", label: "undecided", range: `${Math.round(lo * 100)}–${Math.round(hi * 100)}%` },
    { band: "Higher", label: "higher", range: `≥${Math.round(hi * 100)}%` },
  ];
  return (
    <div className="flex gap-1">
      {bands.map(({ band, label, range }) => (
        <div key={band} className="flex-1">
          <div
            className="h-2 rounded"
            style={{
              background: band === "Intermediate"
                ? `repeating-linear-gradient(135deg, ${BAND_COLOR[band]} 0 4px, #7a4f0a 4px 7px)`
                : BAND_COLOR[band],
            }}
          />
          <div className="mt-0.5 leading-tight">
            <span className="text-gray-300">{label}</span> {range}
          </div>
        </div>
      ))}
    </div>
  );
}

function reducedMotion() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Swing the camera round to face the selected vessel's territory (or a standard view), and run the
 * chest-to-heart fly-in, then hand control back to the orbit controls.
 */
function CameraRig({ focus, resetKey, views, view, fly }: {
  focus: Vessel | null;
  resetKey: number;
  views: Record<Vessel, THREE.Vector3>;
  view: { dir: THREE.Vector3; n: number } | null;
  fly: { from: number; to: number; n: number } | null;
}) {
  const { camera, controls } = useThree();
  const target = useRef<THREE.Vector3 | null>(null);
  const flight = useRef<{ from: number; to: number; t: number } | null>(null);

  useEffect(() => {
    if (focus) target.current = views[focus].clone();
  }, [focus, views]);
  useEffect(() => {
    if (resetKey) target.current = DEFAULT_VIEW.clone().normalize();
  }, [resetKey]);
  useEffect(() => {
    if (view) target.current = view.dir.clone();
  }, [view]);
  useEffect(() => {
    if (!fly) return;
    flight.current = { from: fly.from, to: fly.to, t: 0 };
    camera.position.setLength(fly.from);
    // Arrive from a three-quarter view, slightly above, so the chest reads as 3D.
    target.current = DEFAULT_VIEW.clone().normalize();
    camera.position.copy(new THREE.Vector3(-0.55, 0.3, 1).normalize().multiplyScalar(fly.from));
  }, [fly, camera]);

  useFrame((_, dt) => {
    const f = flight.current;
    if (f) {
      f.t = Math.min(1, f.t + dt / 2.2);
      const e = f.t < 0.25 ? 0 : (f.t - 0.25) / 0.75; // pause on the chest, then ease in
      const s = e * e * (3 - 2 * e);
      camera.position.setLength(f.from + (f.to - f.from) * s);
      if (f.t >= 1) flight.current = null;
    }
    if (target.current) {
      const dist = camera.position.length();
      const dir = camera.position.clone().normalize();
      dir.lerp(target.current, Math.min(1, dt * (f ? 1.4 : 4))).normalize();
      camera.position.copy(dir.multiplyScalar(dist));
      if (!f && dir.angleTo(target.current) < 0.01) target.current = null;
    }
    if (f || target.current) (controls as unknown as { update?: () => void } | null)?.update?.();
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
        <Myocardium meshRef={myocardium} probs={probs} selected={selected} focus={selected ?? hovered} mode={mode} onSelect={onSelect} />
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
  /** Artery whose territory is highlighted: the selected one, else the hovered one. */
  focus: Vessel | null;
  mode: ViewMode;
  onSelect: (v: Vessel | null) => void;
}

function Myocardium({ meshRef, probs, selected, focus, mode, onSelect }: MyocardiumProps) {
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
        for (const v of VESSEL_IDS) {
          tint.add(riskColor(probs[v], tmp).multiplyScalar(w[v]));
        }
        c.lerp(tint, 0.62 - 0.25 * f);
      } else if (focus) {
        // Highlight the myocardium supplied by the selected (or hovered) artery.
        tint.copy(probs ? riskColor(probs[focus], tmp) : NEUTRAL_ARTERY);
        c.lerp(tint, 0.55 * Math.max(0, (w[focus] - 0.35) / 0.65));
      }
      color.setXYZ(i, c.r, c.g, c.b);
    }
    color.needsUpdate = true;
  }, [geometry, probs, focus, mode]);

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
