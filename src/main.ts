import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils, type VRM } from "@pixiv/three-vrm";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { WsClient, type ConnectionState, type SpeakMessage, type TranscriptMessage, type LogMessage, type SettingsMessage } from "./ws-client";import { speakWithLipsync, getMouthOpenValue, getSpeechProgress } from "./lipsync";
import { logVrmReport } from "./vrm-report";
import { MicInput, blobToBase64 } from "./mic";
import { CameraInput } from "./camera";
import { classifyFile, resizeImageToBase64Jpeg, readTextFile } from "./file-upload";
import { IDLE_BONE_NAMES, applyIdleOffsets, idleOffsets } from "./idle-motion";
import { BlinkController, GazeController, addHeadFollow, mulberry32, type GazeFrame } from "./eyes";
import { ExpressionLife, blendTarget } from "./expression-life";

// Phase 7: VRM avatar migration -- see docs/DECISIONS.md for the full
// reasoning. Replaces pixi.js + pixi-live2d5 (a 2D Cubism rig) with
// three.js + @pixiv/three-vrm (a WebGL 3D VRM renderer), so a VRoid
// Studio export can be used instead of a Live2D model. Drop your
// exported .vrm file at MODEL_PATH below -- see README.md for the full
// walkthrough.
const MODEL_PATH =
  (window as unknown as { __YURI_MODEL_PATH__?: string }).__YURI_MODEL_PATH__ ?? "/vrm/yuri.vrm";

// Hand-tuned starting point for a bust-up framing at roughly a real VRM
// humanoid's actual scale (VRM models are authored in real-world meters,
// unlike Live2D's arbitrary internal units). Distance/height are offsets
// from the model's own head bone position (computed after load, see
// boot() below) rather than fixed world coordinates -- this way framing
// adapts to whatever height/proportions the actual loaded model turns
// out to have, instead of a blind guess at absolute numbers that only
// happens to work for one specific model.
const CAMERA_FOV_DEGREES = 32;
const CAMERA_DISTANCE_FROM_HEAD = 1.2;
// Camera slightly above the head, not dead-center, so the model's own eyes are looking slightly down at the camera rather than straight at it -- this reads as a more natural, relaxed pose than a dead-on stare.
const CAMERA_HEIGHT_OFFSET_FROM_HEAD = 0.05;

// VRM's bind/rest pose is a T-pose by default (arms straight out to the
// sides) -- that's normal for a skeleton's rest pose, not something wrong
// with an export, but nothing poses it into anything more natural unless
// code explicitly does so; there's no idle animation clip involved here.
// Rotation values below were derived empirically, not guessed -- see
// docs/DECISIONS.md: loaded a real sample VRM in a Node script and
// computed actual hand-bone world positions via forward kinematics for
// several candidate rotations, picking the one that visibly brought the
// hand down to a natural at-the-side height. VRM's "normalized" humanoid
// bone space is specifically designed to use a consistent convention
// across every VRM model regardless of the source rig, so these values
// should transfer correctly to any other model, not just the one they
// were derived against -- unlike the camera framing above, which
// necessarily depends on each model's own actual proportions.
function applyIdlePose(vrm: VRM): void {
  const humanoid = vrm.humanoid;
  if (!humanoid) return;

  const leftUpperArm = humanoid.getNormalizedBoneNode("leftUpperArm");
  const rightUpperArm = humanoid.getNormalizedBoneNode("rightUpperArm");
  const leftLowerArm = humanoid.getNormalizedBoneNode("leftLowerArm");
  const rightLowerArm = humanoid.getNormalizedBoneNode("rightLowerArm");

  // Mirror-symmetric by construction (confirmed, not assumed) -- left and
  // right take opposite-sign rotations around the same axis.
  leftUpperArm?.rotation.set(0, 0, -1.35);
  rightUpperArm?.rotation.set(0, 0, 1.35);
  // A small bend at the elbow so the arms don't look ramrod-straight
  // glued to her sides -- purely aesthetic, safe to omit if a given
  // model doesn't have separate lower-arm bones for some reason (the
  // optional chaining above already no-ops in that case).
  leftLowerArm?.rotation.set(0, -0.15, 0);
  rightLowerArm?.rotation.set(0, 0.15, 0);
}

// Matches orchestrator/persona.py's VALID_EMOTIONS exactly -- the app-
// facing emotion tags the LLM chooses from and setTargetEmotion() below
// accepts. Module-level (not just local to boot()'s emotion-blend
// system) so setupHud's manual expression-test button below can cycle
// the same list without a second hardcoded copy going stale relative to
// it. NOT the same list as the VRM's own preset names -- see
// VRM_PRESET_NAMES/EMOTION_BLENDS just below for why those two had to
// split apart.
const EMOTION_NAMES = ["happy", "angry", "sad", "teasing", "surprised", "neutral"] as const;
type EmotionName = (typeof EMOTION_NAMES)[number];

// The VRM's own standard expression preset names -- what actually gets
// passed to vrm.expressionManager.setValue(). Four of the six
// EMOTION_NAMES above map straight onto one of these; "teasing" doesn't,
// since there's no standard VRM preset by that name (VRoid Studio never
// exports one unless someone hand-authors it). Kept as its own list
// (rather than reusing EMOTION_NAMES for both jobs, like before this
// split) because updateEmotion() needs to know every underlying preset
// any blend below might touch, independent of which app-facing emotion
// is currently active.
const VRM_PRESET_NAMES = ["happy", "angry", "sad", "relaxed", "surprised", "neutral"] as const;
type VrmPresetName = (typeof VRM_PRESET_NAMES)[number];

// How each app-facing emotion actually gets rendered on the model.
// Four of these are a plain 1:1 handoff to the identically-named VRM
// preset. "teasing" is a genuine composite: the model's own "relaxed"
// preset already supplies a mouth curling into a smirk, blended with a
// slice of "angry" for a narrower, more knowing eye instead of relaxed's
// plain half-lidded look -- the combination is meant to read as a
// sultry, mischievous smirk rather than either preset alone. "happy" is
// capped below full intensity (0.8, not 1) as a mitigation for VRoid
// Studio's default "Joy" preset commonly over-puckering the mouth and
// squeezing the eyes fully shut at 100% weight -- a known, common VRoid
// quirk, not something specific to this model. That mitigation is a
// blunt instrument, though: it softens the effect everywhere but can't
// reshape what the preset actually contains. The real, precise fix for
// how "happy"/"joy" actually looks lives in VRoid Studio's own
// expression editor on the source model (adjust or swap the eye/mouth
// shapes that preset pulls from, then re-export the .vrm) -- that's
// authoring-time mesh sculpting, not something a weight number in this
// file can substitute for. These are first-guess starting weights either
// way, not a confirmed final look -- there's no renderer in this sandbox
// to actually see the result (same caveat as the lighting setup above)
// -- tune both the 0.8 and the 0.3 up or down against the real model.
const EMOTION_BLENDS: Record<EmotionName, Partial<Record<VrmPresetName, number>>> = {
  happy: { happy: 0.8 },
  angry: { angry: 0.8 },
  sad: { sad: 1 },
  teasing: { relaxed: 1, sad: 0.05, angry: 0.05},
  surprised: { surprised: 1 },
  neutral: { neutral: 1 },
};

async function boot(): Promise<void> {
  const canvas = document.getElementById("avatar-canvas") as HTMLCanvasElement;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(window.innerWidth, window.innerHeight);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(CAMERA_FOV_DEGREES, window.innerWidth / window.innerHeight, 0.1, 20);

  // VRoid's toon (MToon) materials still need at least one real light in
  // the scene to shade correctly, unlike an unlit 2D sprite -- but a
  // single strong, off-axis directional light (the original (1,1,1)
  // setup) is a likely cause of the pale/bright patch the user reported
  // on the jacket (see docs/DECISIONS.md): MToon's toon shading uses a
  // fairly hard transition between lit and shadowed bands, and a strong
  // light coming from a steep side angle can turn that transition into
  // an unnaturally sharp, oddly-placed bright patch rather than a smooth
  // gradient. This is a genuine guess, not a confirmed fix -- there's no
  // renderer in this sandbox to actually see the result -- but flatter,
  // more front-on, lower-intensity lighting is the standard fix for
  // exactly this kind of toon-shading artifact, and is also just a
  // better match for the flat, even VTuber look the user wants generally.
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
  // Positioned roughly where the camera itself sits (see CAMERA_* below)
  // rather than off to one side -- light coming from close to the
  // viewer's own direction minimizes the self-shadowing/harsh side
  // lighting a toon material is most likely to render badly.
  keyLight.position.set(0, 1.6, 2.2);
  scene.add(keyLight);
  // A hemisphere light (soft light from "above"/"below" blended by each
  // surface's own normal direction) fills in shadows more evenly than a
  // flat ambient light would, which matters more for toon materials --
  // an evenly-lit flat ambient can still leave the *key* light's harsh
  // transition band visible, where a hemisphere light softens it.
  scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d8e0, 1.1));

  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));

  const gltf = await loader.loadAsync(MODEL_PATH).catch((err) => {
    // The avatar is user-supplied and not in the repo. Fail with a name the
    // catch at the bottom of this file can recognise and turn into an
    // instruction on screen, instead of the blank window a bare console
    // error leaves behind.
    const missing = new Error(
      `avatar model not found at ${MODEL_PATH} -- put a .vrm there (see public/vrm/README.txt)`,
    );
    missing.name = "MissingAvatarError";
    console.error("[yuri]", missing.message, err);
    throw missing;
  });
  const vrm: VRM = gltf.userData.vrm;

  // rotateVRM0 is a no-op for VRM1 exports -- only legacy VRM0.x models
  // (older VRoid Studio versions default to this) face the wrong way for
  // three.js's -Z-forward convention and need the 180-degree turn.
  VRMUtils.rotateVRM0(vrm);
  // Standard best-practice cleanup from every official three-vrm example:
  // merges skinned meshes/morph targets where safe to cut down draw calls
  // and keep facial-expression blending consistent across submeshes --
  // both are no-ops if there's nothing to combine, safe to always call.
  VRMUtils.combineSkeletons(vrm.scene);
  VRMUtils.combineMorphs(vrm);

  applyIdlePose(vrm);
  logVrmReport(vrm);

  scene.add(vrm.scene);

  // Frame the camera relative to the model's own actual head position,
  // computed only now (after adding to the scene and posing it) so world
  // matrices are up to date -- see this constant's own comment above on
  // why this isn't a fixed world-space coordinate.
  const headNode = vrm.humanoid?.getNormalizedBoneNode("head");
  const headWorldPosition = new THREE.Vector3();
  if (headNode) {
    vrm.scene.updateMatrixWorld(true);
    headNode.getWorldPosition(headWorldPosition);
  } else {
    // Extremely unlikely for any real VRM export (a head bone is
    // required by the spec), but fall back to a reasonable guess rather
    // than crash boot() over a malformed file.
    headWorldPosition.set(0, 1.4, 0);
    console.warn("[yuri] VRM has no head bone in its humanoid map -- using a fallback camera position");
  }
  camera.position.set(
    headWorldPosition.x,
    headWorldPosition.y + CAMERA_HEIGHT_OFFSET_FROM_HEAD,
    headWorldPosition.z + CAMERA_DISTANCE_FROM_HEAD,
  );
  camera.lookAt(headWorldPosition);

  function layout(): void {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  }
  layout();
  window.addEventListener("resize", layout);

  // Procedural idle body motion (src/idle-motion.ts, Phase 13 step 1):
  // breathing, weight shift, head drift. Bones are looked up once; the
  // offsets are written every frame in animate() just before vrm.update().
  // IDLE_MOTION_INTENSITY is 0..1 (0 = perfectly still) -- the one knob
  // until the Phase 13 debug menu exists.
  const IDLE_MOTION_INTENSITY = 1;
  const idleBones = Object.fromEntries(
    IDLE_BONE_NAMES.map((name) => [name, vrm.humanoid?.getNormalizedBoneNode(name) ?? null]),
  );
  let idleTime = 0;

  // Eyes (src/eyes.ts, Phase 13 step 2): a gaze with micro-saccades and
  // glances written to vrm.lookAt, and a blink with variety (single, double,
  // slow when relaxed, fewer while she speaks, one with a glance). Replaces
  // the old fixed-length blink loop. The head's share of a glance is added
  // in animate(), right after the idle layer, via addHeadFollow().
  const eyeRng = mulberry32((Math.random() * 0xffffffff) >>> 0);
  const gaze = new GazeController(eyeRng);
  const blinker = new BlinkController(eyeRng);
  let gazeFrame: GazeFrame = { yaw: 0, pitch: 0, headYaw: 0, headPitch: 0 };

  function updateEyes(delta: number, speaking: boolean): void {
    gazeFrame = gaze.update(delta, speaking);
    if (gaze.consumeBlinkRequest()) blinker.requestBlink();
    const closure = blinker.update(delta, speaking);
    vrm.expressionManager?.setValue("blink", closure);
    if (vrm.lookAt) {
      vrm.lookAt.yaw = gazeFrame.yaw;
      vrm.lookAt.pitch = gazeFrame.pitch;
    }
  }

  // Phase 8: gradual expression blending, not a per-line snap (matching
  // docs/ROADMAP.md's own original scoping note for this phase). One
  // named emotion is "current" at a time; every frame, each underlying
  // VRM preset weight eases toward its target in the blendTarget() weights for the current mood
  // (or 0, if that preset isn't part of the current blend) rather than
  // jumping there instantly -- reads as a gradual mood shift over a
  // fraction of a second instead of a jarring instant switch. Tracked
  // per VRM preset now (not per app-facing emotion) since "teasing"
  // needs two presets (relaxed + angry) eased independently at once.
  // Phase 13 step 3: the mood is no longer a bare target that snaps to
  // neutral the moment her audio goes idle. ExpressionLife (src/expression-
  // life.ts) holds a tagged mood while she is busy and for a beat after, then
  // fades it to neutral; blendTarget() turns its frame into the preset
  // weights the easing below chases.
  const life = new ExpressionLife(mulberry32((Math.random() * 0xffffffff) >>> 0));
  const currentPresetWeights: Record<VrmPresetName, number> = {
    happy: 0,
    angry: 0,
    sad: 0,
    relaxed: 0,
    surprised: 0,
    neutral: 1,
  };
  // Per-second blend rate -- at this rate a full swing between two
  // expressions takes well under a second, reading as a smooth but
  // prompt mood shift rather than a slow fade or an instant snap.
  const EMOTION_BLEND_SPEED = 4;

  function setTargetEmotion(emotion?: string): void {
    // Absent or unrecognized -- see ws-client.ts's protocol comment on
    // why this is common and expected, not an error: leave the current
    // expression alone rather than guessing or snapping to "neutral"
    // just because this particular turn didn't produce a usable tag.
    if (!emotion) return;
    if (!(EMOTION_NAMES as readonly string[]).includes(emotion)) return;
    life.set(emotion);
  }

  function updateEmotion(delta: number, busy: boolean): void {
    if (!vrm.expressionManager) return;
    const step = Math.min(1, EMOTION_BLEND_SPEED * delta);
    const targets = blendTarget(EMOTION_BLENDS, life.update(delta, busy), VRM_PRESET_NAMES);
    for (const preset of VRM_PRESET_NAMES) {
      const target = targets[preset];
      const next = currentPresetWeights[preset] + (target - currentPresetWeights[preset]) * step;
      currentPresetWeights[preset] = next;
      vrm.expressionManager.setValue(preset, next);
    }
  }

  const hud = setupHud(setTargetEmotion);

  const clock = new THREE.Clock();
  function animate(): void {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    updateEyes(delta, getSpeechProgress() !== null);
    updateEmotion(delta, hud.isBusy());
    // Advances the caption's per-word reveal against the currently
    // playing clip's real elapsed time -- piggybacking on this loop
    // rather than a second rAF/interval, same reasoning as
    // getMouthOpenValue() below (one shared loop, everything reads off
    // it once a frame).
    hud.tick();
    if (vrm.expressionManager) {
      vrm.expressionManager.setValue("aa", getMouthOpenValue());
    }
    idleTime += delta;
    applyIdleOffsets(idleBones, idleOffsets(idleTime, IDLE_MOTION_INTENSITY));
    addHeadFollow(idleBones, gazeFrame);
    vrm.update(delta);
    renderer.render(scene, camera);
  }
  animate();
}

interface Hud {
  /** Called once a frame from boot()'s shared render loop -- advances
   * the caption's per-word reveal against the currently playing clip. */
  tick(): void;
  /** True while a turn is generating or her audio is playing -- the signal
   * the expression layer holds a mood on (it fades once she is quiet). */
  isBusy(): boolean;
}

function setupHud(setTargetEmotion: (emotion?: string) => void): Hud {
  const input = document.getElementById("input-box") as HTMLInputElement;
  const statusOrb = document.getElementById("status-orb") as HTMLDivElement;
  const micButton = document.getElementById("mic-button") as HTMLButtonElement;
  const actionButton = document.getElementById("action-button") as HTMLButtonElement;
  const caption = document.getElementById("caption") as HTMLDivElement;
  const hud = document.getElementById("hud") as HTMLDivElement;
  const quickActionButton = document.getElementById("quick-action-button") as HTMLButtonElement;
  const quickActionMenu = document.getElementById("quick-action-menu") as HTMLDivElement;
  const qaTempModeButton = document.getElementById("qa-temp-mode") as HTMLButtonElement;
  const qaTempModeState = document.getElementById("qa-temp-mode-state") as HTMLSpanElement;
  const qaOcrWatchButton = document.getElementById("qa-ocr-watch") as HTMLButtonElement;
  const qaOcrWatchState = document.getElementById("qa-ocr-watch-state") as HTMLSpanElement;
  const qaApiModelButton = document.getElementById("qa-api-model") as HTMLButtonElement;
  const qaApiModelState = document.getElementById("qa-api-model-state") as HTMLSpanElement;
  const qaCameraButton = document.getElementById("qa-camera") as HTMLButtonElement;
  const qaCameraState = document.getElementById("qa-camera-state") as HTMLSpanElement;
  const qaUploadButton = document.getElementById("qa-upload") as HTMLButtonElement;
  const fileUploadInput = document.getElementById("file-upload-input") as HTMLInputElement;
  const pendingAttachmentEl = document.getElementById("pending-attachment") as HTMLDivElement;
  const pendingAttachmentNameEl = document.getElementById("pending-attachment-name") as HTMLSpanElement;
  const pendingAttachmentRemoveButton = document.getElementById("pending-attachment-remove") as HTMLButtonElement;
  const qaLogButton = document.getElementById("qa-log") as HTMLButtonElement;
  const qaEmotionTestButton = document.getElementById("qa-emotion-test") as HTMLButtonElement;
  const logPanel = document.getElementById("log-panel") as HTMLDivElement;
  const logPanelBody = document.getElementById("log-panel-body") as HTMLDivElement;
  const logEmptyMessage = document.getElementById("log-empty-message") as HTMLParagraphElement;
  const logClearButton = document.getElementById("log-clear-button") as HTMLButtonElement;
  const logCloseButton = document.getElementById("log-close-button") as HTMLButtonElement;

  const STOP_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>';
  const SEND_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.4 20.6 21 12 3.4 3.4 3 10l12 2-12 2z" /></svg>';

  const setState = (state: ConnectionState) => {
    statusOrb.classList.remove("connected", "listening");
    statusOrb.title = state;
    if (state !== "offline") statusOrb.classList.add(state);
  };

  // Toggled true right when a turn starts (sendUserText/sendUserAudio),
  // false once BOTH of two independent things have happened: the
  // orchestrator has said turn_end, AND SpeakQueue has finished playing
  // every chunk it was given. Those two used to be conflated (turnActive
  // flipped false on turn_end alone), but turn_end only means "done
  // generating" -- audio for earlier sentences can still be queued and
  // playing for several more seconds after that arrives (see
  // ws-client.ts's own protocol comment). Collapsing them made the
  // button flip back to mic/send while she was still audibly mid-reply,
  // so pressing "stop" was no longer possible for the tail end of a
  // turn. orchestratorDone/audioIdle below are the two raw signals;
  // turnActive is always their combination, recomputed through
  // recomputeTurnActive() rather than set directly from either one.
  let orchestratorDone = true;
  let audioIdle = true;
  let turnActive = false;
  // Set true the instant the user clicks "stop," cleared the instant a
  // new turn actually starts (submitTurn/onClip below). Exists because
  // GPT-SoVITS synthesis can legitimately take tens of seconds per
  // chunk (confirmed on the user's own machine) -- if a chunk was
  // already mid-synthesis server-side when stop was clicked, there's a
  // real window where it can still land as a `speak` message after the
  // click handler already reset turnActive to false. Without this
  // guard, that straggler's queue.push() would fire onActive() and flip
  // the stop button back on for a reply the user already dismissed, for
  // no reason they could see -- exactly the "stop button won't go away"
  // symptom reported during real debugging. onTurnEnd is deliberately
  // NOT gated on this -- it only sets orchestratorDone (already true
  // from the manual override) and refreshes the log panel, both
  // harmless no-ops if it arrives late.
  let turnStopped = false;

  // Quick-action menu's Upload item, attach-then-send flow: set the
  // instant a file is picked and successfully read (attachFile below),
  // cleared on send (submitTurn) or explicit removal
  // (pendingAttachmentRemoveButton). Holding the already-read/resized
  // content here (not just the File object) means submitTurn never has
  // to redo any file I/O -- attachFile does that work exactly once,
  // when the file is picked, not again at send time.
  type PendingAttachment = { filename: string; kind: "image" | "text"; content: string };
  let pendingAttachment: PendingAttachment | null = null;

  // Full-body sandbox round: true whenever this window is the "observer"
  // (the sandbox got there first) -- see ws-client.ts's top-of-file
  // comment. Distinct from turnActive: this is about *which window* may
  // talk to her at all, not whether a reply is currently in flight.
  let observerLocked = false;

  // Phase 9's conversation-log panel. `logUserName` defaults to "You"
  // until the first real `log` message arrives with config.yaml's
  // session.user_name -- covers the brief window before the orchestrator
  // has replied at all, and the (should-never-happen) case where that
  // field is somehow missing.
  let logPanelOpen = false;
  let logUserName = "You";

  // Suggested by the user: show a timestamp per logged turn. `turn.ts` is
  // SQLite's `datetime('now')` -- UTC, space-separated, no timezone
  // marker (see memory/db.py's schema) -- so it needs an explicit "this
  // is UTC" hint (the 'Z' below) before handing it to Date, or the
  // browser would otherwise interpret it as already-local time and shift
  // it wrong. Falls back to the raw string if it somehow doesn't parse
  // (a malformed row shouldn't hide the rest of that turn), and is only
  // computed once here rather than re-parsed by anything downstream.
  function formatLogTimestamp(ts: string): string {
    const parsed = new Date(`${ts.replace(" ", "T")}Z`);
    if (Number.isNaN(parsed.getTime())) return ts;
    return parsed.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  }

  function renderLog(turns: LogMessage["turns"]): void {
    logPanelBody.querySelectorAll(".log-turn").forEach((el) => el.remove());
    logEmptyMessage.hidden = turns.length > 0;
    for (const turn of turns) {
      const row = document.createElement("div");
      row.className = "log-turn";

      const timeLine = document.createElement("div");
      timeLine.className = "log-turn-time";
      timeLine.textContent = formatLogTimestamp(turn.ts);
      row.appendChild(timeLine);

      // Unprompted comments (Task Guide chides, Continuous OCR remarks) are
      // logged with an empty user string -- nobody said anything. Skip the
      // user line for those instead of rendering a dangling "Name:" with
      // nothing after it.
      if (turn.user.trim()) {
        const userLine = document.createElement("div");
        userLine.className = "log-turn-user";
        const userName = document.createElement("span");
        userName.className = "log-turn-name";
        userName.textContent = `${logUserName}:`;
        userLine.append(userName, document.createTextNode(turn.user));
        row.appendChild(userLine);
      }

      // A turn stopped before she said anything (see app.py's _run_turn)
      // still gets a row, just with nothing after her name -- an honest
      // record of what actually happened, not something to hide.
      const assistantLine = document.createElement("div");
      assistantLine.className = "log-turn-assistant";
      const assistantName = document.createElement("span");
      assistantName.className = "log-turn-name";
      assistantName.textContent = "Luna:";
      assistantLine.append(assistantName, document.createTextNode(turn.assistant));
      row.appendChild(assistantLine);

      logPanelBody.appendChild(row);
    }
    // New turns land at the bottom (oldest-first, see store.py) -- keep
    // the scroll pinned there so the panel reads like a live chat log
    // rather than leaving the user scrolled up on old messages every
    // time it refreshes.
    logPanelBody.scrollTop = logPanelBody.scrollHeight;
  }

  function setLogPanelOpen(open: boolean): void {
    logPanelOpen = open;
    logPanel.hidden = !open;
    qaLogButton.classList.toggle("active", open);
    if (open) client.sendGetLog();
  }

  function recomputeTurnActive(): void {
    setTurnActive(!orchestratorDone || !audioIdle);
  }

  // One button, not two sitting side by side -- it morphs between stop
  // and send (same element, same click handler, icon/label/behavior
  // swapped based on state) rather than showing/hiding two separate
  // buttons that happen to alternate. Hidden entirely with nothing typed
  // and no reply in flight, since there's nothing for it to do in that
  // state -- the mic button covers that case instead. Called on every
  // turnActive change and every keystroke in the input box.
  function updateInputButtons(): void {
    // Observer-locked: neither button has anything to do (there's nowhere
    // for a click to go -- the orchestrator will just decline it), so hide
    // both rather than leaving one active-looking but non-functional.
    if (observerLocked) {
      actionButton.hidden = true;
      micButton.hidden = true;
      return;
    }
    const hasText = input.value.trim().length > 0 || pendingAttachment !== null;
    if (turnActive) {
      actionButton.hidden = false;
      actionButton.classList.add("stop-mode");
      actionButton.innerHTML = STOP_ICON;
      actionButton.title = "Stop response";
      actionButton.setAttribute("aria-label", "Stop response");
      micButton.hidden = true;
    } else if (hasText) {
      actionButton.hidden = false;
      actionButton.classList.remove("stop-mode");
      actionButton.innerHTML = SEND_ICON;
      actionButton.title = "Send message";
      actionButton.setAttribute("aria-label", "Send message");
      micButton.hidden = true;
    } else {
      actionButton.hidden = true;
      micButton.hidden = false;
    }
  }

  function setTurnActive(active: boolean): void {
    turnActive = active;
    input.disabled = active || observerLocked;
    updateInputButtons();
  }

  function clearPendingAttachment(): void {
    pendingAttachment = null;
    pendingAttachmentEl.hidden = true;
    updateInputButtons();
  }

  // Replaces the old submitText -- now handles both a plain typed
  // message and a pending attachment (with or without a typed caption
  // alongside it), since after the attach-then-send redesign there are
  // three ways this can fire: text alone, attachment alone, or both
  // together. Text-alone keeps the exact same sendUserText path it
  // always had; either case involving an attachment goes through
  // sendUserFile with whatever's in the input box (possibly empty) as
  // the caption.
  function submitTurn(): void {
    const text = input.value.trim();
    if (pendingAttachment) {
      client.sendUserFile(pendingAttachment.filename, pendingAttachment.kind, pendingAttachment.content, text);
      clearPendingAttachment();
    } else {
      if (!text) return;
      client.sendUserText(text);
    }
    input.value = "";
    orchestratorDone = false;
    turnStopped = false;
    recomputeTurnActive();
  }

  // Matches the HTML's own default `hidden` attribute on action-button
  // (mic visible at rest) but called explicitly rather than relying on
  // that alone -- keeps this the single source of truth for the initial
  // state instead of two places that have to agree by accident.
  updateInputButtons();

  // Drives the #caption element (see style.css): a soft, glowing overlay
  // with no background box, whose words pop in one at a time roughly in
  // time with the currently-playing clip -- not the whole sentence
  // slammed onto screen at once. GPT-SoVITS's API returns finished WAV
  // bytes with no per-word timestamps (confirmed against
  // orchestrator/tts.py), so there's no ground truth to sync against
  // frame-perfectly; instead each word is allotted a share of the clip's
  // real duration proportional to its own character length (a longer
  // word gets more of the clip's time than "a" does) -- a common
  // approximation for exactly this situation, close enough to read as
  // "in sync" without needing real forced alignment.
  let captionWordEls: HTMLSpanElement[] = [];
  // Fraction (0..1) of the clip's duration at which each word should
  // become the active one -- captionWordStarts[i] is word i's own start
  // point, computed once per line in startCaptionLine, not every tick.
  let captionWordStarts: number[] = [];
  let captionActiveIndex = -1;
  // How many already-spoken words stay fully visible right behind the
  // highlight before they start individually collapsing/fading out --
  // this (not the end-of-clip fade) is what keeps a long sentence from
  // ever growing into one big block on screen.
  const CAPTION_TRAIL_WORDS = 4;

  function startCaptionLine(text: string): void {
    const words = text.trim().split(/\s+/).filter(Boolean);
    caption.innerHTML = "";
    captionWordEls = words.map((word) => {
      const span = document.createElement("span");
      span.className = "caption-word";
      span.textContent = word;
      caption.appendChild(span);
      return span;
    });
    const totalChars = words.reduce((sum, w) => sum + w.length, 0) || 1;
    let seenChars = 0;
    captionWordStarts = words.map((w) => {
      const start = seenChars / totalChars;
      seenChars += w.length;
      return start;
    });
    captionActiveIndex = -1;
    caption.classList.add("visible");
    // Reveal the first word immediately rather than waiting for the
    // first real tick() -- getSpeechProgress() reliably returns null for
    // a frame or two after play() while the browser's still loading
    // audio metadata, and an empty caption during that gap reads as
    // broken rather than just "about to start."
    applyCaptionProgress(0);
  }

  function tickCaptionProgress(): void {
    if (captionWordEls.length === 0) return;
    const progress = getSpeechProgress();
    if (!progress) return;
    const frac = Math.min(1, progress.elapsed / progress.duration);
    let idx = 0;
    for (let i = 0; i < captionWordStarts.length; i++) {
      if (captionWordStarts[i]! <= frac) idx = i;
      else break;
    }
    applyCaptionProgress(idx);
  }

  // Actually mutates the DOM for a given active word index -- split out
  // from tickCaptionProgress so startCaptionLine can also call it
  // directly for word 0 without duplicating this logic. Cheap to call
  // repeatedly (early-returns if the index hasn't actually changed since
  // last time) since tickCaptionProgress calls it every frame. Every
  // word's class is recomputed fresh from its distance to activeIndex
  // each time this runs, rather than tracked as separate one-shot
  // transitions -- so a word sliding from "spoken" into "fading" (or
  // back, if activeIndex ever needed to move backward) falls out
  // naturally with no extra state to keep in sync.
  function applyCaptionProgress(activeIndex: number): void {
    if (activeIndex === captionActiveIndex) return;
    captionActiveIndex = activeIndex;
    captionWordEls.forEach((el, i) => {
      if (i === activeIndex) {
        const alreadyShown = el.classList.contains("shown");
        el.classList.add("shown", "active");
        el.classList.remove("spoken", "fading");
        if (!alreadyShown) {
          // Word is going from display:none to visible for the first
          // time -- force a reflow before adding .pop so the browser
          // treats the keyframe animation as a fresh start rather than
          // a no-op on a class that (from its perspective) never left.
          el.classList.remove("pop");
          void el.offsetWidth;
          el.classList.add("pop");
        }
      } else if (i < activeIndex) {
        el.classList.add("shown");
        el.classList.remove("active", "pop");
        if (activeIndex - i > CAPTION_TRAIL_WORDS) {
          // Fallen out of the trailing window -- collapse and fade this
          // one word out individually instead of leaving it dimmed
          // forever until the whole line disappears at once.
          el.classList.add("fading");
          el.classList.remove("spoken");
        } else {
          el.classList.add("spoken");
          el.classList.remove("fading");
        }
      } else {
        el.classList.remove("shown", "active", "spoken", "fading", "pop");
      }
    });
  }

  function endCaptionLine(): void {
    caption.classList.remove("visible");
  }

  const defaultPlaceholder = input.placeholder;
  const TEMP_MODE_PLACEHOLDER = "Temp chat — nothing said here is kept…";
  let placeholderTimer: number | undefined;
  // Generic "flash a message in the input's placeholder for a few
  // seconds" helper -- originally just for STT transcripts, now also
  // used by the expression-test button below so both share one revert
  // timer instead of two independently racing each other. Reverts to
  // TEMP_MODE_PLACEHOLDER instead of defaultPlaceholder while Temp Chat
  // is on (tempModeEnabled, declared further below but already set by
  // the time this timeout can actually fire -- see its own comment) --
  // otherwise a transcript flash or an expression-test click would
  // silently revert the input to the normal placeholder mid Temp Chat,
  // contradicting the "must have a highlighter" requirement this exists
  // to satisfy.
  function flashPlaceholder(text: string): void {
    window.clearTimeout(placeholderTimer);
    input.placeholder = text;
    placeholderTimer = window.setTimeout(() => {
      input.placeholder = tempModeEnabled ? TEMP_MODE_PLACEHOLDER : defaultPlaceholder;
    }, 4000);
  }

  function showTranscript(text: string): void {
    flashPlaceholder(text ? `Heard: "${text}"` : "Didn't catch that -- try again?");
  }

  const queue = new SpeakQueue({
    onCaptionStart: (text) => startCaptionLine(text),
    onCaptionEnd: () => endCaptionLine(),
    onActive: () => {
      audioIdle = false;
      recomputeTurnActive();
    },
    onIdle: () => {
      audioIdle = true;
      recomputeTurnActive();
      // She's actually done talking (not just done generating). This used
      // to snap the face to neutral right here; the expression layer now
      // holds the mood for a beat and fades it (Phase 13 step 3), so it
      // does not freeze on her face and does not vanish the instant the
      // audio ends either.
    },
  });
  const client = new WsClient({
    surface: "shell",
    onStateChange: setState,
    // Server-owned switch; see renderApiModel. Fires on connect and after every
    // change, so the tray and the menu badge always agree.
    onSettings: (msg) => renderApiModel(msg),
    // Resend UI-held toggle state whenever the socket (re)opens: the send
    // helpers drop messages while the socket isn't OPEN, so a toggle clicked
    // during startup or a reconnect never reached the orchestrator. Temp
    // Chat is per-connection server-side (defaults off, so only "on" needs
    // sending); Continuous OCR is process-wide, so its current value is sent
    // either way and the menu badge and the backend always agree. Both
    // variables are declared further down -- fine, this only runs after the
    // socket opens, long after setup has finished.
    onOpen: () => {
      if (tempModeEnabled) client.sendSetTempMode(true);
      client.sendSetOcrWatch(ocrWatchEnabled);
    },
    onSpeak: (msg: SpeakMessage) => {      // A chunk from the turn we already stopped, arriving late (see
      // turnStopped's own comment above) -- drop it silently rather
      // than let it revive the stop button for a reply that's already
      // dismissed. Cleared the moment a real new turn starts.
      if (turnStopped) return;
      queue.push(msg);
    },
    onTranscript: (msg: TranscriptMessage) => showTranscript(msg.text),
    onTurnEnd: (emotion) => {
      orchestratorDone = true;
      recomputeTurnActive();
      setTargetEmotion(emotion);
      // Keeps an already-open panel live during a conversation, without
      // needing a manual reopen -- cheap (one small WS round trip) since
      // this only fires when the panel is actually visible.
      if (logPanelOpen) client.sendGetLog();
    },
    onLog: (msg: LogMessage) => {
      logUserName = msg.user_name || "You";
      renderLog(msg.turns);
    },
    // Sandbox connected first -- she's already talking there. Lock input
    // here rather than let both windows race to start a turn (which would
    // mean two TTS clips potentially playing over each other -- see
    // ws-client.ts's top-of-file comment). Flips back the moment the
    // sandbox disconnects and the orchestrator promotes this connection.
    onSurfaceStatus: (role) => {
      observerLocked = role === "observer";
      input.disabled = observerLocked || turnActive;
      updateInputButtons();
      if (observerLocked) flashPlaceholder("Luna's active in the sandbox right now…");
    },
    // Phase 5 -- camera.py (orchestrator) asking this connection for a
    // frame. Referenced here before its own `const camera` declaration
    // further below in this same setup function -- safe because this is
    // a callback, only actually invoked later once a real
    // request_camera_frame message arrives, by which point camera has
    // long since been constructed (same closure/TDZ reasoning
    // tempModeEnabled's own forward-reference in flashPlaceholder
    // already relies on above). captureFrame() returning null (not
    // armed, or the video isn't ready yet) is sent through exactly as
    // null -- camera.py's request_frame turns that into the same
    // "camera isn't armed" message the model sees either way, so there
    // is nothing this handler needs to distinguish or react to.
    onRequestCameraFrame: () => {
      client.sendCameraFrame(camera.captureFrame());
    },
  });
  client.connect();

  // Quick-action menu: opened/closed by quickActionButton, closed again
  // by picking any item, clicking outside it, or Escape -- standard
  // dropdown behavior. aria-expanded kept in sync for screen readers;
  // the rotated-+ visual (see .active in style.css) is the sighted
  // equivalent.
  let quickActionMenuOpen = false;
  function setQuickActionMenuOpen(open: boolean): void {
    quickActionMenuOpen = open;
    quickActionMenu.hidden = !open;
    quickActionButton.classList.toggle("active", open);
    quickActionButton.setAttribute("aria-expanded", String(open));
  }
  quickActionButton.addEventListener("click", () => {
    setQuickActionMenuOpen(!quickActionMenuOpen);
  });
  document.addEventListener("click", (event) => {
    if (!quickActionMenuOpen) return;
    const target = event.target as Node;
    if (quickActionMenu.contains(target) || quickActionButton.contains(target)) return;
    setQuickActionMenuOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && quickActionMenuOpen) setQuickActionMenuOpen(false);
  });

  // Temp Chat -- quick-action menu's own toggle (see app.py's
  // temp_mode/temp_turn_flags for what this actually does server-side:
  // no recall, no forget, no task tracking, and excluded from
  // end-of-session consolidation). Purely a local boolean plus one
  // fire-and-forget message -- there's nothing to await or roll back,
  // see ws-client.ts's sendSetTempMode for why.
  let tempModeEnabled = false;
  function setTempModeEnabled(enabled: boolean): void {
    tempModeEnabled = enabled;
    qaTempModeState.textContent = enabled ? "On" : "Off";
    qaTempModeState.classList.toggle("on", enabled);
    hud.classList.toggle("temp-mode-active", enabled);
    input.placeholder = enabled ? TEMP_MODE_PLACEHOLDER : defaultPlaceholder;
    client.sendSetTempMode(enabled);
  }
  qaTempModeButton.addEventListener("click", () => {
    setTempModeEnabled(!tempModeEnabled);
    // Deliberately does NOT close the menu -- a toggle the user might
    // want to flip back within the next few seconds (e.g. "actually
    // no, turn it back off") shouldn't require reopening the menu to
    // undo. The other items below do close it, since they're one-shot
    // actions rather than a state you'd immediately reconsider.
  });

  // Continuous OCR -- quick-action menu's own toggle (see
  // orchestrator/ocr_watch.py: ambient screen-watching, independent of
  // Task Guide Mode). Same shape as Temp Chat's own toggle right above --
  // a local boolean, a badge update, one fire-and-forget message. No
  // HUD-level border/glow the way Temp Chat gets -- this doesn't change
  // how a turn behaves the way Temp Chat does, it just occasionally
  // speaks unprompted, so the menu's own on/off badge is indicator
  // enough.
  let ocrWatchEnabled = false;
  function setOcrWatchEnabled(enabled: boolean): void {
    ocrWatchEnabled = enabled;
    qaOcrWatchState.textContent = enabled ? "On" : "Off";
    qaOcrWatchState.classList.toggle("on", enabled);
    client.sendSetOcrWatch(enabled);
  }
  qaOcrWatchButton.addEventListener("click", () => {
    setOcrWatchEnabled(!ocrWatchEnabled);
    // Same reasoning as Temp Chat's button above -- stays open for quick
    // reconsideration.
  });

  // "Use API model" -- the only toggle whose truth lives on the server (and in
  // data/settings.json), because the router is what obeys it and a second
  // window must agree. The click only *requests* a change; the badge is drawn
  // from the server's reply, so it can never claim "On" while the backend has
  // no key (it shows "No key" instead).
  let apiModelEnabled = false;
  function renderApiModel(settings: SettingsMessage): void {
    apiModelEnabled = settings.use_api_model;
    qaApiModelState.textContent = settings.use_api_model
      ? (settings.cloud_available ? "On" : "No key")
      : "Off";
    qaApiModelState.classList.toggle("on", settings.use_api_model && settings.cloud_available);
    qaApiModelState.classList.toggle("warn", settings.use_api_model && !settings.cloud_available);
    // Keep the tray check item in step (no-op outside Tauri).
    invoke("sync_cloud_toggle", { enabled: settings.use_api_model }).catch(() => {});
  }
  qaApiModelButton.addEventListener("click", () => {
    client.sendSetApiModel(!apiModelEnabled);
    // Stays open, same as the other stateful toggles.
  });

  qaLogButton.addEventListener("click", () => {
    setLogPanelOpen(!logPanelOpen);
    setQuickActionMenuOpen(false);
  });
  logCloseButton.addEventListener("click", () => {
    setLogPanelOpen(false);
  });
  logClearButton.addEventListener("click", () => {
    // No confirmation dialog -- this mirrors "forget that" (Phase 3)
    // rather than a destructive-file-delete pattern: low stakes (it's a
    // convenience view of things already said out loud, not the only
    // copy of anything), and a confirm() popup would be an odd, very
    // un-Luna interruption for a one-button UI this small.
    client.sendClearLog();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    submitTurn();
  });
  input.addEventListener("input", updateInputButtons);

  // "/" focuses the chatbox from anywhere in the window, same idea as
  // Discord/Slack's own "/" shortcut, so typing to her doesn't need a
  // click first. Listens on the whole document, not just window, since
  // with nothing else focusable in this HUD, document.activeElement is
  // normally <body> and keydown bubbles up to document either way.
  // Ignored (falls through to being typed normally) whenever the input
  // is already focused, so "/" still works as a literal character inside
  // a message; also ignored with any modifier held, so this doesn't
  // hijack an OS/browser-level Ctrl+/ or similar. A disabled input
  // (mid-turn) simply can't receive focus, so this is a harmless no-op
  // in that state rather than needing its own guard.
  document.addEventListener("keydown", (event) => {
    if (event.key !== "/") return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (document.activeElement === input) return;
    event.preventDefault();
    input.focus();
  });

  actionButton.addEventListener("click", () => {
    if (turnActive) {
      // Both fire immediately, independently -- queue.stopAll() kills
      // client-side audio/lipsync right now without waiting on a round
      // trip; sendStop() separately tells the orchestrator to cancel
      // generation server-side (saves compute, and records a truthful
      // partial reply in history instead of a full one nobody heard the
      // end of). orchestratorDone/audioIdle are set directly here rather
      // than left for turn_end/onIdle to report back -- this is a manual
      // override of both signals at once, not something that should wait
      // on either arriving on its own; the button disappearing should
      // feel instant, same as the audio actually stopping. turnStopped
      // additionally suppresses any straggler chunk this same turn sends
      // afterward (see its own comment above).
      turnStopped = true;
      queue.stopAll();
      client.sendStop();
      orchestratorDone = true;
      audioIdle = true;
      recomputeTurnActive();
      setTargetEmotion("neutral");
    } else {
      submitTurn();
    }
  });

  // Dev/QA toggle: click through every VRM expression Luna actually
  // supports, one per click, entirely independent of anything the
  // orchestrator says -- lets you confirm the model's expression
  // blendshapes are wired up correctly without needing to provoke each
  // mood out of the LLM by conversation. Starts at index -1 so the very
  // first click lands on EMOTION_NAMES[0] rather than skipping it.
  let emotionTestIndex = -1;
  qaEmotionTestButton.addEventListener("click", () => {
    emotionTestIndex = (emotionTestIndex + 1) % EMOTION_NAMES.length;
    const emotion = EMOTION_NAMES[emotionTestIndex]!;
    setTargetEmotion(emotion);
    flashPlaceholder(`Testing expression: ${emotion}`);
    setQuickActionMenuOpen(false);
  });

  const mic = new MicInput({
    onRecordingChange: (recording) => {
      micButton.classList.toggle("recording", recording);
      micButton.setAttribute("aria-pressed", String(recording));
    },
    onClip: async (blob) => {
      const audioB64 = await blobToBase64(blob);
      client.sendUserAudio(audioB64);
      orchestratorDone = false;
      turnStopped = false;
      recomputeTurnActive();
    },
    onError: (err) => {
      // Most likely mic permission denied, or no input device -- nothing
      // client-side to recover from beyond logging; the user can grant
      // permission (or plug in a mic) and click again.
      console.error("[yuri] mic capture failed -- check mic permission", err);
    },
  });
  micButton.addEventListener("click", () => {
    mic.toggle();
  });

  // Quick-action menu's Camera toggle (Phase 5) -- see camera.ts's own
  // top comment for the full "armed but no preview, tray icon is the
  // only indicator" design. onArmedChange/onError both just update the
  // menu's own badge; the tray icon update happens inside camera.ts
  // itself (via the Rust set_camera_indicator command), not here.
  const camera = new CameraInput({
    onArmedChange: (armed) => {
      qaCameraState.textContent = armed ? "On" : "Off";
      qaCameraState.classList.toggle("on", armed);
    },
    onError: (err) => {
      // Most likely getUserMedia permission denied, or no camera present
      // -- nothing client-side to recover from beyond logging; same
      // shape as the mic's own onError above. Badge stays "Off" since
      // arm() never actually completed.
      console.error("[yuri] camera access failed -- check camera permission", err);
    },
  });
  qaCameraButton.addEventListener("click", () => {
    if (camera.isArmed) {
      void camera.disarm();
    } else {
      void camera.arm();
    }
    // Deliberately does NOT close the menu -- same reasoning as Temp
    // Chat's own button above: a toggle worth immediately reconsidering
    // shouldn't require reopening the menu to undo.
  });

  // Quick-action menu's Upload Image/File item. Attach-then-send: picking
  // a file no longer starts a turn immediately -- it reads/resizes the
  // file once (attachFile below) and stages it as pendingAttachment, so
  // the user can add a typed caption in the normal input box before
  // actually sending (submitTurn handles both cases). Still gated the
  // same way the text input already is (disabled while turnActive/
  // observerLocked) -- attaching mid-turn has no obvious use and would
  // just complicate when the eventual send is allowed to fire.
  qaUploadButton.addEventListener("click", () => {
    if (turnActive || observerLocked) return;
    setQuickActionMenuOpen(false);
    fileUploadInput.click();
  });
  fileUploadInput.addEventListener("change", () => {
    const file = fileUploadInput.files?.[0];
    // Reset immediately so picking the exact same file again still
    // fires a change event next time -- browsers don't fire `change`
    // for selecting an already-selected file otherwise.
    fileUploadInput.value = "";
    if (file) void attachFile(file);
  });
  pendingAttachmentRemoveButton.addEventListener("click", () => {
    clearPendingAttachment();
  });

  async function attachFile(file: File): Promise<void> {
    const classified = classifyFile(file);
    if (classified.kind === "unsupported") {
      flashPlaceholder(classified.reason);
      return;
    }
    let content: string;
    try {
      content = classified.kind === "image" ? await resizeImageToBase64Jpeg(file) : await readTextFile(file);
    } catch (err) {
      console.error("[yuri] failed to read uploaded file", err);
      flashPlaceholder("Couldn't read that file -- try a different one?");
      return;
    }
    pendingAttachment = { filename: file.name, kind: classified.kind, content };
    pendingAttachmentNameEl.textContent = `File uploaded: ${file.name}`;
    pendingAttachmentEl.hidden = false;
    updateInputButtons();
    input.focus();
  }

  // F9 push-to-talk, works even when Luna's window isn't focused -- the
  // actual global shortcut registration lives in src-tauri/src/lib.rs
  // (Rust), which just emits this event; all the recording logic stays
  // here in one place, same start()/stop() the mic button itself uses.
  // Guarded against turnActive here specifically because it's a global
  // hotkey, not a click on the (now-hidden) mic button -- hiding the
  // button doesn't stop F9 from still firing while a reply's in flight.
  listen<string>("hotkey-talk", (event) => {
    if (turnActive || observerLocked) return;
    if (event.payload === "pressed") {
      mic.start();
    } else if (event.payload === "released") {
      mic.stop();
    }
  }).catch((err) => {
    console.error("[yuri] couldn't attach F9 push-to-talk listener", err);
  });

  // Tray menu's "Use API model" check item. Rust just reports the new check
  // state; the request goes over the same WS as the menu toggle, and the badge
  // is updated from the server's reply like everything else here.
  listen<boolean>("yuri://use-api-model", (event) => {
    client.sendSetApiModel(Boolean(event.payload));
  }).catch((err) => {
    console.error("[yuri] couldn't attach tray API-model listener", err);
  });

  return {
    tick: () => tickCaptionProgress(),
    isBusy: () => !orchestratorDone || !audioIdle,
  };
}

// Phase 2's orchestrator sends one reply as several `speak` messages --
// one per sentence, streamed as the LLM produces them (see
// orchestrator/app.py / chunking.py) -- instead of Phase 1's one message
// per reply. Without a queue, a second chunk arriving while the first is
// still playing would call speakWithLipsync() again, starting a second
// Audio element/AnalyserNode racing the first one. This plays each queued
// chunk to completion before starting the next. No `model`/`vrm`
// reference needed here as of Phase 7 -- lipsync.ts's mouth-openness
// value is read by main.ts's own single shared render loop, not pushed
// into the model directly from here the way the old Live2D version did.
interface SpeakQueueOptions {
  /** Called with the full sentence text right as its clip starts
   * playing -- setupHud's startCaptionLine() splits it into words and
   * reveals them progressively from there, timed against the same
   * clip's real elapsed/duration (see lipsync.ts's getSpeechProgress).
   * Fired from playNext(), not push(), so it stays correctly timed even
   * when a chunk sits queued behind an earlier one. */
  onCaptionStart: (text: string) => void;
  /** Called whenever a clip finishes playing -- whether or not another
   * is already queued behind it -- or when stopAll() cuts playback
   * short. Firing this between every chunk (not only when the whole
   * queue empties) is what gives consecutive sentences a clean
   * fade-out/fade-in gap instead of one line's words abruptly being
   * replaced by the next line's mid-transition. */
  onCaptionEnd: () => void;
  /** Fires once, on the 0-pending/not-playing -> playing transition
   * (i.e. when this queue has something to say for the first time since
   * it was last empty) -- not on every chunk, since chunks 2..n start
   * from playNext()'s own "ended" callback, not from push(). */
  onActive: () => void;
  /** Fires once the queue has nothing left pending and nothing currently
   * playing -- either the last chunk finished naturally, or stopAll()
   * was called. This is the "audio side" half of main.ts's combined
   * turnActive signal (see recomputeTurnActive). */
  onIdle: () => void;
}

class SpeakQueue {
  private pending: SpeakMessage[] = [];
  private playing = false;
  private currentHandle: ReturnType<typeof speakWithLipsync> | null = null;

  // A short pause between chunks, not zero. Two back-to-back HTMLAudioElements
  // chained on "ended" can have a few ms of overlap at the boundary (the next
  // element's play() has its own startup latency, and audio already queued in
  // the previous element's Web Audio routing can trail slightly past "ended")
  // -- this masks that, and as a side effect makes the sentence-by-sentence
  // delivery sound like natural pauses between sentences rather than abrupt
  // bursts.
  private static readonly GAP_MS = 150;

  constructor(private opts: SpeakQueueOptions) {}

  push(msg: SpeakMessage): void {
    const wasIdle = !this.playing && this.pending.length === 0;
    this.pending.push(msg);
    if (wasIdle) this.opts.onActive();
    if (!this.playing) this.playNext();
  }

  /** The "stop response" button: drops everything still queued and stops
   * whatever's currently playing immediately -- not a graceful fade,
   * since the point is for it to feel instant. Doesn't wait on the
   * orchestrator at all; ws-client.ts's sendStop() is fired alongside
   * this, separately, for the backend-side cancellation. */
  stopAll(): void {
    this.pending = [];
    this.currentHandle?.stop();
    this.currentHandle = null;
    this.playing = false;
    this.opts.onCaptionEnd();
    // Not routed through onIdle -- main.ts's stop handler sets its own
    // idle/done flags directly and synchronously alongside this call, so
    // firing onIdle here too would just be a redundant second
    // recomputeTurnActive() call for the same outcome.
  }

  private playNext(): void {
    const msg = this.pending.shift();
    if (!msg) {
      this.playing = false;
      this.currentHandle = null;
      this.opts.onIdle();
      return;
    }
    this.playing = true;
    this.opts.onCaptionStart(msg.text);

    const blob = base64ToBlob(msg.audio_b64, msg.mime);
    const url = URL.createObjectURL(blob);

    const handle = speakWithLipsync(url);
    this.currentHandle = handle;
    handle.onFinish(() => {
      URL.revokeObjectURL(url);
      this.currentHandle = null;
      this.opts.onCaptionEnd();
      window.setTimeout(() => this.playNext(), SpeakQueue.GAP_MS);
    });
  }
}

function base64ToBlob(base64: string, mime: string): Blob {
  const bytes = atob(base64);
  const buffer = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buffer[i] = bytes.charCodeAt(i);
  return new Blob([buffer], { type: mime });
}

boot().catch((err) => {
  console.error("[yuri] failed to boot", err);
  if ((err as Error)?.name === "MissingAvatarError") showAvatarHint(String((err as Error).message));
});

/**
 * Visible, actionable failure for the one missing asset that stops boot.
 * A blank transparent window with an error in DevTools is indistinguishable
 * from "the app is broken"; this says exactly which file to add and where.
 */
function showAvatarHint(detail: string): void {
  if (document.getElementById("asset-hint")) return;
  const hint = document.createElement("div");
  hint.id = "asset-hint";
  hint.innerHTML = `
    <h1>Yuri needs an avatar</h1>
    <p>${detail}</p>
    <p>Drop your <code>.vrm</code> in <code>public/vrm/yuri.vrm</code>, and your
    animations in <code>public/vrm-animations/</code>, then refresh.</p>
    <p class="muted">Chat and voice run without an avatar once Phase 2 lands; for now
    the model is required to start.</p>`;
  document.body.appendChild(hint);
}
