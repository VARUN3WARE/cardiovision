import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Bvh, ContactShadows, Environment, Lightformer, OrbitControls, PerformanceMonitor } from "@react-three/drei";
import { Bloom, EffectComposer, N8AO, ToneMapping, Vignette } from "@react-three/postprocessing";
import { ToneMappingMode } from "postprocessing";
import { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from "react";
import * as THREE from "three";
import AnatomyPins from "./AnatomyPins";
import ArteryTube, { type ArteryProbs } from "./ArteryTube";
import AtlasHeart, { type AtlasLab, type AtlasPick, type ChestLayers } from "./AtlasHeart";
import ErrorBoundary from "./ErrorBoundary";
import VesselInspector, { VesselChip } from "./VesselInspector";
import EcgMonitor from "../twin/EcgMonitor";
import { advanceClock, beatTiming, twinFromResult, type BeatTiming, type TwinParams } from "../twin/twin";
import { useHeartMotion } from "../twin/useHeartMotion";
import { VESSEL, VESSEL_IDS } from "../anatomy/registry";
import ReasoningFlow, { type OverlayRect, type ReasoningFrame } from "../reasoning/ReasoningFlow";
import { groupAttributions, sigmoid, typicalColumns } from "../reasoning/dataGroups";
import { useFeatureInfo } from "../reasoning/featureInfo";
import { reasoningHover } from "../reasoning/hoverStore";
import { FlowAnchor, supportHull } from "../reasoning/anchors";
import LabPanel, { LabBanner, LabLegend } from "../lab/LabPanel";
import { LAB_DEFAULT, baselineText, labFromQuery, labPatient, labTwin, type LabSettings } from "../lab/labState";
import { labState } from "../twin/coronaryPhysiology";
import type { ExplainResponse, FeaturesResponse, MetricsResponse } from "../api/client";
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
// Ischaemia lab panel: docked at the right when the 3D view is wide enough, else a sheet at the bottom.
const LAB_PANEL_W = 262;
const LAB_SIDE_MIN_W = 600;

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
  /** Full results shown by the colours, inspector and reasoning streams (a what-if scenario's while one is on). */
  result?: ExplainResponse | null;
  /**
   * The patient's own recorded result. The function twin (heartbeat, ECG), the lab's starting values and the
   * wall-motion note always use it; when it differs from `result`, a what-if scenario is on.
   */
  recorded?: ExplainResponse | null;
  metrics?: MetricsResponse | null;
  /** /features, to label typical (not entered) values in the reasoning view (fetched when not given). */
  info?: FeaturesResponse | null;
  /** Artery highlighted from outside the scene (e.g. hovering the bullseye). */
  highlight?: Vessel | null;
  selected: Vessel | null;
  onSelect: (vessel: Vessel | null) => void;
  bpm: number;
}

export default function HeartScene({
  probs, pending = false, result = null, recorded, metrics = null, info, highlight = null, selected, onSelect, bpm,
}: HeartSceneProps) {
  // What the patient's own heart does: the recorded result, never a scenario's hypothetical inputs.
  const patient = recorded ?? result;
  const whatIf = recorded != null && result != null && recorded !== result;
  const [mode, setMode] = useState<ViewMode>("arteries");
  const [beating, setBeating] = useState(true);
  const [showEcg, setShowEcg] = useState(true);
  // Ischaemia lab, a teaching sandbox on the Atlas heart (deep link: ?lab=lad|lcx|rca&narrowing=80&exercise=1).
  const [labOn, setLabOn] = useState(() => labFromQuery(window.location.search) !== null);
  const [labSettings, setLabSettings] = useState<LabSettings>(() => labFromQuery(window.location.search) ?? LAB_DEFAULT);
  // The chest view to restore when the lab closes (shown by default, so also after a lab deep link).
  const chestBeforeLab = useRef<boolean | null>(labFromQuery(window.location.search) ? true : null);
  // Reasoning flow (deep link: ?reasoning=1, optionally &model=lad|lcx|rca). The lab takes precedence. It
  // hides the chest (the streams need a clear background); the chest comes back when it closes.
  const [reasoning, setReasoning] = useState(reasoningFromQuery);
  const chestBeforeReasoning = useRef<boolean | null>(reasoningFromQuery() ? true : null);
  const [fx, setFx] = useState(true);
  const [pins, setPins] = useState(false);
  const [hovered, setHovered] = useState<Vessel | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [framesReady, setFramesReady] = useState(false);
  // Realistic model when its file is present and loads; the schematic heart otherwise.
  // Deep link: ?heart=atlas | realistic | schematic
  const [model, setModel] = useState<HeartModel>(initialModel);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [specimenFailed, setSpecimenFailed] = useState(false);
  const [specimenViews, setSpecimenViews] = useState<Record<Vessel, THREE.Vector3> | null>(null);
  const [atlasOk, setAtlasOk] = useState<boolean | null>(null);
  const [atlasFailed, setAtlasFailed] = useState(false);
  const [atlasViews, setAtlasViews] = useState<Record<Vessel, THREE.Vector3> | null>(null);
  const [chest, setChest] = useState(() => labFromQuery(window.location.search) === null && !reasoningFromQuery());
  const [layers, setLayers] = useState<ChestLayers>({ skin: true, skeleton: true, organs: true });
  const [layerMenu, setLayerMenu] = useState(false);
  const [moreMenu, setMoreMenu] = useState(false);
  const [xray, setXray] = useState(false);
  const [pick, setPick] = useState<AtlasPick | null>(null);
  const [view, setView] = useState<{ dir: THREE.Vector3; n: number } | null>(null);
  // Opening straight into the atlas (deep link) starts with the chest fly-in too (not into the lab or the
  // reasoning view: no chest there).
  const [fly, setFly] = useState<{ from: number; to: number; n: number } | null>(() =>
    initialModel() === "atlas" && !reducedMotion() && labFromQuery(window.location.search) === null && !reasoningFromQuery()
      ? { from: CHEST_DISTANCE, to: HEART_DISTANCE, n: 1 } : null);
  // Narrow views: the selected artery shows as a one-line chip until "details" opens the full card.
  const [inspectorOpen, setInspectorOpen] = useState(false);
  // Zoomed in so far that some reasoning streams would end off screen (they are hidden).
  const [streamsOffscreen, setStreamsOffscreen] = useState(false);
  useEffect(() => setInspectorOpen(false), [selected]);
  const viewRef = useRef<HTMLDivElement>(null);
  const { width: viewWidth, height: viewHeight } = useElementSize(viewRef);
  // DOM overlays the 3D labels and cards must avoid (toolbar, notices column, legend, view buttons).
  const toolbarRef = useRef<HTMLDivElement>(null);
  const noticesRef = useRef<HTMLDivElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);
  const viewsRef = useRef<HTMLDivElement>(null);
  const creditRef = useRef<HTMLDivElement>(null);
  const overlay = useOverlayLayout(viewRef, toolbarRef, noticesRef, [legendRef, viewsRef, creditRef]);

  useEffect(() => {
    specimenAvailable().then(setAvailable);
    atlasAvailable().then(setAtlasOk);
    const m = new URLSearchParams(window.location.search).get("model");
    if (m && (VESSEL_IDS as readonly string[]).includes(m)) onSelect(m as Vessel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    if (m !== "atlas" && labOn) closeLab(); // the lab runs on the Atlas heart only
    setModel(m);
    setPick(null);
    // Entering the atlas with the chest shown: start outside the chest and fly in to the heart.
    if (m === "atlas" && chest && !reducedMotion()) setFly((f) => ({ from: CHEST_DISTANCE, to: HEART_DISTANCE, n: (f?.n ?? 0) + 1 }));
  };
  const showPick = pick && (pick.vessel === "lm" || pick.vessel === selected) ? pick : null;
  // Function twin: the patient's recorded pulse, EF, blood pressure and ECG findings drive the heart (also while
  // a what-if scenario colours the arteries: the heartbeat and ECG never show hypothetical inputs).
  const twin = useMemo(() => twinFromResult(patient, bpm), [patient, bpm]);
  // Ischaemia lab: the patient's resting pulse, systolic BP and age plus a generic sandbox narrowing. While it
  // runs, the twin beats at the lab's exercise rate and the monitor shows the lab's ST depression (marked "lab").
  const labActive = labOn && useAtlas;
  const labIn = useMemo(() => labPatient(patient, twin), [patient, twin]);
  const lab = useMemo(
    () => (labActive ? labState(labSettings.vessel, labSettings.narrowing / 100, labSettings.exercise, labIn) : null),
    [labActive, labSettings, labIn],
  );
  const atlasLab: AtlasLab | null = useMemo(() => (lab ? { vessel: lab.vessel, state: lab } : null), [lab]);
  const shownTwin = useMemo(() => (lab ? labTwin(twin, lab) : twin), [lab, twin]);
  const timing = useMemo(() => beatTiming(shownTwin), [shownTwin]);
  const labSide = viewWidth >= LAB_SIDE_MIN_W;
  // Narrow 3D view (e.g. 1024 px windows): tighter toolbar, with X-ray, HD and Reset in a "⋯" menu, and the
  // reasoning view's data types as chips above and below the heart.
  const compact = viewWidth > 0 && viewWidth < 520;
  // Too narrow for the full vessel card beside the heart: a one-line chip until "details" is clicked.
  const narrowCard = viewWidth > 0 && viewWidth < 420;
  const pad = compact ? "px-1.5" : "px-2";

  const openLab = () => {
    if (reasoning) {
      setReasoning(false);
      if (chestBeforeReasoning.current !== null) setChest(chestBeforeReasoning.current);
    }
    setPick(null);
    setShowEcg(true);
    if (model !== "atlas") setModel("atlas");
    // The lab needs a clear view of the heart: the chest comes back when the lab closes.
    chestBeforeLab.current = reasoning ? chestBeforeReasoning.current ?? chest : chest;
    chestBeforeReasoning.current = null;
    setChest(false);
    setFly(null);
    setLabOn(true);
  };
  function closeLab() {
    setLabOn(false);
    if (chestBeforeLab.current !== null) setChest(chestBeforeLab.current);
    chestBeforeLab.current = null;
  }
  const toggleReasoning = () => {
    if (reasoning) {
      setReasoning(false);
      if (chestBeforeReasoning.current !== null) setChest(chestBeforeReasoning.current);
      chestBeforeReasoning.current = null;
      return;
    }
    chestBeforeReasoning.current = labOn ? chestBeforeLab.current ?? chest : chest;
    if (labOn) closeLab();
    setChest(false);
    setFly(null);
    setReasoning(true);
  };
  // A click on empty space (or on a reasoning stream with nothing behind it) clears the selection.
  const background = () => {
    select(null);
    setPick(null);
    reasoningHover.set(null);
  };
  // In the lab, clicking an artery or a wall moves the sandbox narrowing there (the patient's selection is untouched).
  const select = (v: Vessel | null) => {
    if (labActive) {
      if (v) setLabSettings((st) => ({ ...st, vessel: v }));
      return;
    }
    onSelect(v);
  };
  // Face the sandbox narrowing (and its artery's territory) when the lab opens or the narrowing moves.
  const [labViews, setLabViews] = useState<Record<Vessel, THREE.Vector3> | null>(null);
  useEffect(() => {
    if (!labActive || !labViews) return;
    setView((cur) => ({ dir: labViews[labSettings.vessel].clone(), n: (cur?.n ?? 0) + 1 }));
  }, [labActive, labViews, labSettings.vessel]);
  // The patient's recorded wall-motion grade (a what-if contrast changes the model's input, not the record).
  const rwma = useMemo(() => {
    const exp = patient?.explanations.cad;
    const i = exp ? exp.feature_names.indexOf("Region RWMA") : -1;
    return exp && i >= 0 ? exp.feature_values[i] : null;
  }, [patient]);

  // Reasoning view: the shown result's attributions for the selected model, with typical fill-in values marked.
  const target = selected ?? "cad";
  const features = useFeatureInfo(info, reasoning && result !== null, result);
  const reasoningExp = result?.explanations[target] ?? null;
  const typical = useMemo(
    () => (result && reasoningExp ? typicalColumns(reasoningExp, result.imputed, features) : null),
    [result, reasoningExp, features],
  );
  // During a what-if drag, fresh probabilities arrive before their explanation: the streams then trail them.
  const explained = reasoningExp ? sigmoid(groupAttributions(reasoningExp).logOdds) : null;
  const lagging = !!result && explained !== null && Math.abs(explained - result.results[target].probability) > 1e-4;
  const reasoningFrame = useMemo<ReasoningFrame>(
    () => ({ top: overlay.toolbar, notices: overlay.notices, bottoms: overlay.bottoms }),
    [overlay],
  );
  const belowNotices = (overlay.notices?.bottom ?? overlay.toolbar) + 6;
  // The legend shrinks to the band scale while the reasoning view (or a narrow view's vessel card) needs the room.
  const legendCollapsed = reasoning || (narrowCard && !!selected && inspectorOpen);
  const stackViews = viewWidth > 0 && viewWidth < 12 + (legendCollapsed ? 208 : 240) + 8 + 78 + 12;
  // Model credit (for the atlas also the reference-anatomy caveat): bottom right, or under the legend when narrow.
  const creditInLegend = compact && !labActive;
  const credit = useAtlas ? (
    <a href={ATLAS_CREDIT.url} target="_blank" rel="noreferrer" className="block text-[9px] leading-tight text-gray-500 hover:text-gray-300">
      Reference anatomy, not this patient&apos;s imaging. Right dominance assumed. {ATLAS_CREDIT.text}
    </a>
  ) : useSpecimen ? (
    <a href={SPECIMEN_CREDIT.url} target="_blank" rel="noreferrer" className="block text-[9px] leading-tight text-gray-500 hover:text-gray-300">
      Heart model: “{SPECIMEN_CREDIT.title}” by {SPECIMEN_CREDIT.author}, {SPECIMEN_CREDIT.license}
    </a>
  ) : null;

  const xrayButton = useAtlas && (
    <button
      onClick={() => setXray((x) => !x)}
      aria-pressed={xray}
      className={`rounded ${pad} py-1 ${xray ? "bg-sky-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
      title="See-through heart wall: shows the vessels on the back of the heart from the front"
    >
      X-ray
    </button>
  );
  const hdButton = (
    <button
      onClick={() => setFx((f) => !f)}
      className={`rounded bg-slate-900/80 ${pad} py-1 text-gray-300 hover:bg-slate-700`}
      title="Ambient occlusion, bloom and vignette. Turn off on slow GPUs."
    >
      {fx ? "HD on" : "HD off"}
    </button>
  );
  const viewButtons = ANATOMICAL_VIEWS.map((v) => (
    <button
      key={v.key}
      title={v.title}
      onClick={() => setView((cur) => ({ dir: new THREE.Vector3(...v.dir).normalize(), n: (cur?.n ?? 0) + 1 }))}
      className="h-6 w-6 rounded bg-slate-900/80 text-[10px] font-semibold text-gray-300 hover:bg-slate-700"
    >
      {v.key}
    </button>
  ));
  const resetButton = (
    <button
      onClick={() => {
        onSelect(null);
        setResetKey((k) => k + 1);
      }}
      className={`rounded bg-slate-900/80 ${pad} py-1 text-gray-300 hover:bg-slate-700`}
    >
      Reset
    </button>
  );

  useEffect(() => {
    document.body.style.cursor = hovered ? "pointer" : "auto";
    return () => {
      document.body.style.cursor = "auto";
    };
  }, [hovered]);

  return (
    <div className="flex h-full w-full flex-col overflow-hidden rounded">
    <div ref={viewRef} className="relative min-h-0 flex-1">
      <Canvas
        camera={{ position: [0, 0.1, 6.8], fov: 40 }}
        dpr={[1, 1.5]}
        gl={{ antialias: !fx }}
        onPointerMissed={background}
      >
        <color attach="background" args={["#141a26"]} />
        <TwinClock timing={timing} running={beating} />
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
                  twin={shownTwin}
                  timing={timing}
                  lab={atlasLab}
                  onLabViews={setLabViews}
                  onSelect={select}
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
                  twin={shownTwin}
                  timing={timing}
                  onSelect={select}
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
              twin={shownTwin}
              timing={timing}
              onSelect={select}
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

        {reasoning && reasoningExp && (
          <ReasoningFlow
            exp={reasoningExp}
            target={target}
            typical={typical}
            stale={pending || lagging}
            whatIf={whatIf}
            frame={reasoningFrame}
            narrow={compact}
            onBackgroundClick={background}
            onOffscreen={setStreamsOffscreen}
          />
        )}

        <FirstFrame onReady={() => setFramesReady(true)} />
        {import.meta.env.DEV && <DevHook />}
        <FitCamera />
        {/* Lab: keep the heart centred in the part of the view the panel leaves free (left of the side panel, or
            between the toolbar, ~3 rows, and the bottom sheet, 52% of the height). */}
        <ViewShift
          x={labActive && labSide ? (LAB_PANEL_W + 12) / 2 : 0}
          y={labActive && !labSide && viewHeight ? Math.max(0, 0.26 - 48 / viewHeight) : 0}
        />
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
      <div ref={toolbarRef} className="absolute left-3 top-3 z-20 flex flex-wrap gap-1 text-xs">
        {models.length > 1 && (
          <div className="mr-1 flex overflow-hidden rounded" role="radiogroup" aria-label="Heart model">
            {models.map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={model === m}
                onClick={() => chooseModel(m)}
                title={MODEL_INFO[m].title}
                className={`${pad} py-1 ${model === m ? "bg-slate-200 text-slate-900" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
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
            className={`rounded ${pad} py-1 ${mode === m ? "bg-red-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
          >
            {m === "arteries" ? "Arteries" : "Territories"}
          </button>
        ))}
        {useAtlas && (
          <>
            <button
              onClick={() => {
                if (!chest && !reducedMotion()) setFly((f) => ({ from: CHEST_DISTANCE, to: HEART_DISTANCE, n: (f?.n ?? 0) + 1 }));
                chestBeforeReasoning.current = null; // the user's own choice stays when the reasoning view closes
                setChest((c) => !c);
              }}
              aria-pressed={chest}
              className={`rounded ${pad} py-1 ${chest ? "bg-sky-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
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
                          chestBeforeReasoning.current = null;
                          setChest(true);
                        }}
                      />
                      {label}
                    </label>
                  ))}
                </div>
              )}
            </div>
            {!compact && xrayButton}
          </>
        )}
        {!useAtlas && (
        <button
          onClick={() => setPins((p) => !p)}
          className={`rounded ${pad} py-1 ${pins ? "bg-sky-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
          title="Show numbered anatomy annotations"
        >
          Anatomy
        </button>
        )}
        <button
          onClick={() => setBeating((b) => !b)}
          className={`rounded bg-slate-900/80 ${pad} py-1 text-gray-300 hover:bg-slate-700`}
          title={lab ? "Toggle heartbeat animation (rate set by the lab's exercise slider)" : "Toggle heartbeat animation (rate = patient pulse)"}
        >
          {beating ? `♥ ${lab ? Math.round(lab.hr) : bpm}` : "♡ paused"}
        </button>
        <button
          onClick={() => setShowEcg((e) => !e)}
          aria-pressed={showEcg}
          className={`rounded ${pad} py-1 ${showEcg ? "bg-emerald-800 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
          title="Illustrative ECG monitor built from the patient's recorded findings, in step with the heartbeat"
        >
          ECG
        </button>
        <button
          onClick={toggleReasoning}
          aria-pressed={reasoning}
          className={`rounded ${pad} py-1 ${reasoning ? "bg-violet-700 text-white" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
          title="Show how much each data type (history, symptoms, ECG, echo, labs, pre-test score) moved the selected model's estimate"
        >
          Reasoning
        </button>
        {atlasOk && !atlasFailed && (
          <button
            onClick={() => (labOn ? closeLab() : openLab())}
            aria-pressed={labOn}
            className={`rounded ${pad} py-1 ${labOn ? "bg-amber-500 font-semibold text-slate-950" : "bg-slate-900/80 text-gray-300 hover:bg-slate-700"}`}
            title="Ischaemia lab: what a narrowed artery does to blood flow, with textbook physiology (a teaching sandbox on the Atlas heart, not this patient's arteries)"
          >
            Lab
          </button>
        )}
        {compact ? (
          <div className="relative">
            <button
              onClick={() => setMoreMenu((m) => !m)}
              aria-expanded={moreMenu}
              aria-label="More view controls"
              title={useAtlas ? "Standard views, X-ray, HD and Reset" : "Standard views, HD and Reset"}
              className="rounded bg-slate-900/80 px-1.5 py-1 text-gray-300 hover:bg-slate-700"
            >
              ⋯
            </button>
            {moreMenu && (
              <div className="absolute right-0 top-7 z-20 flex flex-col gap-1 rounded bg-slate-900/95 p-1.5 shadow-lg" onMouseLeave={() => setMoreMenu(false)}>
                <div role="group" aria-label="Standard views" className="grid grid-cols-3 gap-0.5">{viewButtons}</div>
                {xrayButton}
                {hdButton}
                {resetButton}
              </div>
            )}
          </div>
        ) : (
          <>
            {hdButton}
            {resetButton}
          </>
        )}
      </div>

      {/* Notices under the toolbar, stacked in one centred column so they never cover it or each other. */}
      <div
        ref={noticesRef}
        className="pointer-events-none absolute inset-x-3 z-10 flex flex-col items-center gap-1"
        style={{ top: overlay.toolbar + 6 }}
      >
        {pending && probs && (
          <div role="status" className="rounded bg-amber-950/90 px-2 py-1 text-[10px] text-amber-200">
            Inputs changed: colours show the previous estimate
          </div>
        )}
        {whatIf && !labActive && (
          <div role="note" className="max-w-full rounded border border-sky-700/70 bg-sky-950/90 px-2 py-1 text-center text-[10px] leading-snug text-sky-100">
            What-if: colours show the scenario; heartbeat and ECG show recorded values
          </div>
        )}
        {reasoning && (
          <ReasoningCaption
            result={result}
            selected={selected}
            explained={explained}
            whatIf={whatIf}
            stale={pending ? "previous estimate" : lagging ? "updating" : streamsOffscreen ? "zoom out to see every stream" : null}
            narrow={compact}
          />
        )}
        {selected && result && !labActive && (reasoning || (narrowCard && !inspectorOpen)) && (
          <VesselChip
            vessel={selected}
            result={result}
            metrics={metrics}
            onClose={() => {
              onSelect(null);
              setPick(null);
            }}
            onDetails={() => {
              if (reasoning) toggleReasoning();
              setInspectorOpen(true);
            }}
            detailsTitle={reasoning ? "Close Reasoning and show this artery's details" : "Show this artery's details"}
          />
        )}
      </div>

      {labActive && (
        <>
          {labSide && (
            <LabBanner
              className="absolute z-20 max-w-[min(92%,440px)] -translate-x-1/2"
              style={{ left: `calc((100% - ${LAB_PANEL_W + 12}px) / 2)`, top: belowNotices }}
            />
          )}
          {lab && (
            <LabPanel
              settings={labSettings}
              onChange={setLabSettings}
              state={lab}
              patient={labIn}
              baseline={baselineText(patient, twin, labIn)}
              stReadable={twin.bbb !== "LBBB"}
              onClose={closeLab}
              className={labSide ? "absolute bottom-3 right-3 z-20" : "absolute inset-x-2 bottom-2 z-20 max-h-[52%]"}
              style={labSide ? { width: LAB_PANEL_W, top: overlay.toolbar + 6 } : undefined}
              banner={!labSide}
            />
          )}
        </>
      )}

      {/* Anatomical view buttons (in the "⋯" menu in narrow views) */}
      {!compact && (
        <div
          ref={viewsRef}
          className="absolute bottom-12 right-3 z-20 grid grid-cols-3 gap-0.5"
          style={labActive ? (labSide ? { right: LAB_PANEL_W + 24 } : { bottom: "calc(52% + 1.75rem)" }) : stackViews ? { bottom: overlay.legend + 6 } : undefined}
          role="group"
          aria-label="Standard views"
        >
          {viewButtons}
        </div>
      )}

      {/* Selected vessel: inspector card below the notices, above the bottom controls (in narrow views a chip in
          the notices column until "details"; hidden while the reasoning view or the lab use that space); a clicked
          left-main branch gets a short note instead. */}
      {reasoning || labActive ? null : selected ? (
        narrowCard && !inspectorOpen ? null : (
          <VesselInspector
            vessel={selected}
            result={result}
            metrics={metrics}
            pick={showPick}
            onClose={() => {
              onSelect(null);
              setPick(null);
            }}
            style={{
              top: belowNotices,
              width: Math.min(270, Math.max(180, viewWidth - 24)),
              maxHeight: Math.max(120, viewHeight - belowNotices - (narrowCard ? Math.max(overlay.legend, overlay.views) : overlay.views) - 6),
            }}
          />
        )
      ) : showPick?.vessel === "lm" && (
        <div className="absolute right-3 z-20 max-w-[260px] rounded bg-slate-900/90 p-3 text-xs text-gray-300" style={{ top: belowNotices }}>
          <div className="text-white first-letter:uppercase">{showPick.branch.name}</div>
          <div className="text-gray-500">{showPick.branch.fma} · BodyParts3D {showPick.branch.fj}</div>
          <div className="mt-1 text-gray-400">Left main: not labelled in this dataset, so no model output; shown neutral.</div>
        </div>
      )}

      {credit && !creditInLegend && (
        <div
          ref={creditRef}
          className={`absolute bottom-3 right-3 ${useAtlas ? "max-w-[240px]" : "max-w-[220px]"} text-right`}
          style={useAtlas && labActive ? (labSide ? { right: LAB_PANEL_W + 24 } : { bottom: "calc(52% + 0.5rem)" }) : undefined}
        >
          {credit}
        </div>
      )}

      {/* Legend (the lab has its own: its arteries show the sandbox, not the estimates). While the reasoning view
          (or a narrow view's vessel card) needs the room, only the band scale. */}
      {labActive ? (
        labSide && (
          <div ref={legendRef} className="absolute bottom-3 left-3 z-20 w-56">
            <LabLegend />
          </div>
        )
      ) : (
      <div
        ref={legendRef}
        className={`absolute bottom-3 left-3 z-20 ${legendCollapsed ? "w-52" : "w-60"} max-w-[calc(100%-1.5rem)] text-[10px] text-gray-400`}
      >
        {legendCollapsed ? (
          <div className="mb-1 text-gray-300">
            {probs ? (compact ? "Chance of ≥50% narrowing per artery" : "Chance of ≥50% narrowing · one colour per artery") : "Run a prediction to colour the arteries"}
          </div>
        ) : (
          <>
            <MapSemantics rwma={rwma} short={compact} />
            {mode === "territories" && (
              <div className="mb-2 rounded bg-slate-900/80 p-1.5 text-gray-300">
                Muscle shaded by the artery that usually supplies it (standard anatomy, right dominance assumed).
                Illustrative: not a perfusion or wall-motion map.
              </div>
            )}
            {!compact && <div className="mb-2 text-gray-500">Drag to rotate · scroll to zoom · click an artery or heart wall</div>}
            <div className="mb-1 text-gray-300">
              {probs ? "Chance of ≥50% narrowing in each artery (not % blocked)" : "Run a prediction to colour the arteries"}
            </div>
          </>
        )}
        <BandLegend />
        {credit && creditInLegend && <div className="mt-1.5">{credit}</div>}
      </div>
      )}
    </div>
    {showEcg && (
      <div className="h-[92px] shrink-0">
        <EcgMonitor params={shownTwin} timing={timing} />
      </div>
    )}
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

/** Advances the shared heart clock once per frame (before any heart reads it). */
function TwinClock({ timing, running }: { timing: BeatTiming; running: boolean }) {
  useFrame((_, dt) => advanceClock(dt, timing, running));
  return null;
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

/**
 * Shift the rendered view (not the camera) so the heart sits in the part of the canvas an overlay leaves
 * free: x pixels to the left, y × the canvas height upwards. Labels and picking follow the same projection.
 */
function ViewShift({ x, y }: { x: number; y: number }) {
  const { camera, size } = useThree();
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    if ((!x && !y) || !size.width || !size.height) cam.clearViewOffset();
    else cam.setViewOffset(size.width, size.height, x, y * size.height, size.width, size.height);
  }, [camera, size.width, size.height, x, y]);
  useEffect(() => () => (camera as THREE.PerspectiveCamera).clearViewOffset(), [camera]);
  return null;
}

/** Size of an element, kept up to date. */
function useElementSize(ref: RefObject<HTMLElement>) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => setSize((s) => (s.width === el.clientWidth && s.height === el.clientHeight ? s : { width: el.clientWidth, height: el.clientHeight }));
    const ro = new ResizeObserver(read);
    ro.observe(el);
    read();
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

interface OverlayLayout {
  /** Bottom of the toolbar (px from the top of the 3D view). */
  toolbar: number;
  /** The centred notices column, when it shows anything. */
  notices: OverlayRect | null;
  /** Overlays along the bottom edge: legend, view buttons, credit line. */
  bottoms: OverlayRect[];
  /** Tops of the legend and of the view buttons, as px from the bottom of the 3D view. */
  legend: number;
  views: number;
}

const rectOf = (el: HTMLElement | null): OverlayRect | null =>
  el && el.offsetWidth > 0
    ? { left: el.offsetLeft, right: el.offsetLeft + el.offsetWidth, top: el.offsetTop, bottom: el.offsetTop + el.offsetHeight }
    : null;

/**
 * Where the 3D view's DOM overlays are, kept up to date (after every render, as positions follow state, and on
 * resizes), so the reasoning towers, captions and cards can stay clear of them.
 */
function useOverlayLayout(
  view: RefObject<HTMLElement>,
  toolbar: RefObject<HTMLElement>,
  notices: RefObject<HTMLElement>,
  bottomRefs: RefObject<HTMLElement>[],
): OverlayLayout {
  const [layout, setLayout] = useState<OverlayLayout>({ toolbar: 64, notices: null, bottoms: [], legend: 0, views: 0 });
  const read = useCallback(() => {
    const v = view.current;
    if (!v) return;
    const H = v.clientHeight;
    const t = toolbar.current, n = notices.current;
    let box: OverlayRect | null = null;
    if (n && n.childElementCount > 0) {
      let left = Infinity, right = -Infinity;
      for (const c of Array.from(n.children) as HTMLElement[]) {
        left = Math.min(left, n.offsetLeft + c.offsetLeft);
        right = Math.max(right, n.offsetLeft + c.offsetLeft + c.offsetWidth);
      }
      box = { left, right, top: n.offsetTop, bottom: n.offsetTop + n.offsetHeight };
    }
    const rects = bottomRefs.map((r) => rectOf(r.current));
    const [legendRect, viewsRect] = rects;
    const next: OverlayLayout = {
      toolbar: t ? t.offsetTop + t.offsetHeight : 0,
      notices: box,
      bottoms: rects.filter((r): r is OverlayRect => r !== null),
      legend: legendRect ? H - legendRect.top : 0,
      views: viewsRect ? H - viewsRect.top : 0,
    };
    setLayout((prev) => (JSON.stringify(prev) === JSON.stringify(next) ? prev : next));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, toolbar, notices, ...bottomRefs]);
  useLayoutEffect(() => read());
  useEffect(() => {
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => read());
    for (const r of [view, toolbar, notices, ...bottomRefs]) if (r.current) ro.observe(r.current);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [read, view, toolbar, notices, ...bottomRefs]);
  return layout;
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
function MapSemantics({ rwma, short = false }: { rwma: number | null; short?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-2 space-y-1">
      <div className="flex items-start gap-1 rounded bg-slate-900/80 p-1.5 text-gray-300">
        <span>
          {short
            ? "One colour per artery: position along it is not predicted."
            : "Vessel-level estimates: one colour per artery. Position along the artery is not predicted."}
        </span>
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
          {short ? (
            <>Wall-motion abnormality grade {rwma} of 4: where is not recorded, so not drawn.</>
          ) : (
            <>
              Regional wall-motion abnormality grade {rwma} of 4: where it is was not recorded in the data, so it is
              not drawn on the heart.
            </>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Reasoning view caption: which model's evidence is flowing where, and what the streams are (and are not).
 * Narrow views show one line, with the rest behind (i).
 */
function ReasoningCaption({ result, selected, explained, whatIf, stale, narrow }: {
  result: ExplainResponse | null;
  selected: Vessel | null;
  /** The probability the shown attributions add up to. */
  explained: number | null;
  whatIf: boolean;
  /** Why the streams are dimmed, if they are. */
  stale: string | null;
  narrow: boolean;
}) {
  const [open, setOpen] = useState(false);
  const pct = explained !== null ? ` (${formatPct(explained)})` : "";
  const target = selected
    ? `${VESSEL[selected].label} model → ${VESSEL[selected].label} label`
    : `${narrow ? "CAD" : "Overall CAD"} model${pct} → whole heart`;
  return (
    <div
      role="note"
      className="w-max max-w-full rounded bg-slate-900/85 px-2.5 py-1 text-center text-[10px] leading-snug text-gray-300 shadow sm:max-w-[min(100%,440px)]"
    >
      <div className="font-semibold text-white">
        Reasoning: {result ? target : "run an estimate to see the evidence"}
        {result && whatIf && <span className="font-normal text-sky-300"> (what-if inputs)</span>}
        {result && stale && <span className="font-normal text-amber-300"> · {stale}</span>}
        {narrow && <span className="font-normal text-gray-300"> · data, not anatomy</span>}
        {narrow && (
          <button
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            aria-label="What the streams show"
            className="pointer-events-auto ml-1 rounded-full border border-gray-500 px-1 font-normal leading-tight text-gray-300 hover:bg-slate-700"
          >
            i
          </button>
        )}
      </div>
      {(!narrow || open) && (
        <>
          <div>Data streams, not anatomy: width = how much each data type moved this estimate</div>
          <div className="text-gray-400">
            <span className="text-red-400">red pushes up</span> · <span className="text-sky-400">blue pulls down</span> ·
            grey ≈ little effect · associations in this referral cohort, not causes · hover or tap a data type for its
            findings
          </div>
        </>
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

function initialModel(): HeartModel {
  if (labFromQuery(window.location.search)) return "atlas"; // the lab runs on the Atlas heart
  const h = new URLSearchParams(window.location.search).get("heart");
  return h === "atlas" ? "atlas" : h === "schematic" ? "schematic" : "specimen";
}

function reasoningFromQuery() {
  return new URLSearchParams(window.location.search).get("reasoning") === "1" && labFromQuery(window.location.search) === null;
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
    // After the orbit controls (-1), before the reasoning overlay (-0.5) reads the camera.
  }, -0.75);
  return null;
}

interface HeartProps {
  probs: ArteryProbs | null;
  selected: Vessel | null;
  hovered: Vessel | null;
  mode: ViewMode;
  pins: boolean;
  twin: TwinParams;
  timing: BeatTiming;
  onSelect: (v: Vessel | null) => void;
  onHover: (v: Vessel | null) => void;
}

function Heart({ probs, selected, hovered, mode, pins, twin, timing, onSelect, onHover }: HeartProps) {
  const ventricles = useRef<THREE.Group>(null);
  const atria = useRef<THREE.Group>(null);
  const outflow = useRef<THREE.Group>(null);
  const myocardium = useRef<THREE.Mesh>(null);
  const myoGeometry = useMemo(() => buildMyocardium(), []);
  // Cardiac mass (ventricles and atria, not the great vessels): the reasoning view's halo and tower clearance.
  const cardiac = useMemo(() => {
    const ball = new THREE.SphereGeometry(1.1, 12, 8).getAttribute("position");
    return supportHull([
      { positions: myoGeometry.getAttribute("position") },
      ...ATRIA.map((a) => ({
        positions: ball,
        matrix: new THREE.Matrix4().compose(
          new THREE.Vector3(...a.center),
          new THREE.Quaternion().setFromEuler(new THREE.Euler(...(a.rotation ?? [0, 0, 0]))),
          new THREE.Vector3(...a.radii),
        ),
      })),
    ]);
  }, [myoGeometry]);

  // Function twin: atrial kick after the P wave, then ventricular ejection scaled by the patient's EF
  // (which also distends the great arteries, more with higher blood pressure); LBBB adds apical rocking.
  useHeartMotion(twin, timing, (m) => {
    const v = m.ventricular;
    ventricles.current?.scale.set(1 - m.squeeze * v, 1 - 0.5 * m.squeeze * v, 1 - m.squeeze * v);
    if (ventricles.current) ventricles.current.rotation.z = 0.05 * m.rock;
    atria.current?.scale.setScalar(1 - 0.045 * m.atrial + 0.015 * v);
    outflow.current?.scale.setScalar(1 + m.aortic * v);
  });

  return (
    <group rotation={HEART_ROTATION} position={[0.1, -0.2, 0]}>
      <FlowAnchor id="heart" position={cardiac.centre} hull={cardiac.hull} />
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
        <Myocardium meshRef={myocardium} geometry={myoGeometry} probs={probs} selected={selected} focus={selected ?? hovered} mode={mode} onSelect={onSelect} />
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
  geometry: THREE.BufferGeometry;
  probs: ArteryProbs | null;
  selected: Vessel | null;
  /** Artery whose territory is highlighted: the selected one, else the hovered one. */
  focus: Vessel | null;
  mode: ViewMode;
  onSelect: (v: Vessel | null) => void;
}

function Myocardium({ meshRef, geometry, probs, selected, focus, mode, onSelect }: MyocardiumProps) {
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
    <>
      <ArteryTube
        vessel={seg.vessel === "lm" || seg.vessel === "vein" ? null : seg.vessel}
        name={seg.name}
        curve={curve}
        radiusAt={radiusAt}
        labelPos={seg.label ? labelPos : null}
        occluders={[occluder]}
        {...rest}
      />
      {seg.label && seg.vessel !== "lm" && seg.vessel !== "vein" && <FlowAnchor id={seg.vessel} position={labelPos} />}
    </>
  );
}
