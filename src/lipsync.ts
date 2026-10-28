// VRM lipsync -- architecturally different from the Live2D version this
// replaces, and deliberately so, not just a mechanical port. The Live2D
// version had to hook into Cubism5InternalModel's "beforeModelUpdate"
// event because Cubism restores parameters from a snapshot at the start
// of every frame, wiping anything set from an independent rAF loop before
// it could ever visibly land (see docs/DECISIONS.md's STT/lipsync entries
// for that whole saga). VRM's VRMExpressionManager has no such
// snapshot/restore cycle -- confirmed by actually reading
// @pixiv/three-vrm-core's bundled source (not assumed just because it
// seemed architecturally different): `setValue()` just sets
// `expression.weight` directly, and `update()` reads whatever that
// current weight is and applies it -- so a plain independent
// requestAnimationFrame loop calling setValue() each frame, right before
// vrm.update(delta), works correctly here. That loop lives in main.ts
// (there's one shared scene-wide rAF loop, not one per audio clip), which
// is why this file doesn't drive anything itself -- it only tracks
// *which* mouth-openness value that shared loop should read this frame.

// ---- Mouth-openness math (pure, so it can be checked without a renderer) --
//
// A real session showed her mouth snapped wide open -- fangs and tongue
// showing -- on single frames. The model's teeth are fine: that is simply
// what VRoid's "aa" shape looks like at full weight, and the old driver
// pushed it there constantly, for three reasons:
//   1. It read only 128 samples (an AnalyserNode's frequencyBinCount, half
//      of fftSize 256) = a few milliseconds, shorter than one voiced pitch
//      cycle's worth of structure, so each frame saw whatever part of the
//      waveform it happened to land on and the value flapped 0..1 per frame.
//   2. rms * 4 clipped at 1.0 for ordinary speech (the TTS output is loud).
//   3. Nothing smoothed or capped it, and no noise gate kept room tone from
//      nudging the mouth.
// Fix: read a ~20-40 ms window, gate the floor, map through a soft curve,
// cap the opening at MOUTH_MAX_OPEN (full "aa" is never reached), and ease
// toward the target with a fast attack and a slower release.
//
// These numbers were tuned by eye on the real model only for MOUTH_MAX_OPEN
// (0.6 read slightly small, so 0.8); the rest are untouched first guesses.

/** Highest "aa" weight ever applied. 1.0 shows teeth and tongue on this model; 0.6 was
 * a touch too small in use, so 0.8 (the user's call). */
export const MOUTH_MAX_OPEN = 0.8;
/** Below this RMS (room tone, codec noise, the tail of a word) the mouth stays shut. */
export const MOUTH_NOISE_GATE = 0.02;
/** RMS that maps to a fully-open (= MOUTH_MAX_OPEN) mouth. Typical loud speech sits near 0.2. */
export const MOUTH_RMS_FOR_MAX = 0.28;
/** Per-second ease rates: open quickly so syllables land, close a bit slower so it doesn't chatter. */
export const MOUTH_ATTACK_PER_S = 28;
export const MOUTH_RELEASE_PER_S = 14;

/** RMS of a Web Audio byte time-domain buffer (128 = silence), 0..1. */
export function rmsOfTimeDomain(samples: Uint8Array): number {
  if (samples.length === 0) return 0;
  let sumSquares = 0;
  for (let i = 0; i < samples.length; i++) {
    const centered = (samples[i]! - 128) / 128;
    sumSquares += centered * centered;
  }
  return Math.sqrt(sumSquares / samples.length);
}

/** Where the mouth wants to be for a given RMS: 0 below the gate, then a
 * gentle curve up to MOUTH_MAX_OPEN. Never exceeds MOUTH_MAX_OPEN. */
export function mouthTargetFromRms(rms: number): number {
  if (rms <= MOUTH_NOISE_GATE) return 0;
  const level = Math.min(1, (rms - MOUTH_NOISE_GATE) / (MOUTH_RMS_FOR_MAX - MOUTH_NOISE_GATE));
  // sqrt-ish curve: quiet syllables still visibly move the mouth, loud ones
  // don't slam it open.
  return Math.pow(level, 0.7) * MOUTH_MAX_OPEN;
}

/** One smoothing step toward `target` over `dtSeconds`. Opens with the
 * attack rate, closes with the release rate; frame-rate independent. */
export function easeMouth(current: number, target: number, dtSeconds: number): number {
  const rate = target > current ? MOUTH_ATTACK_PER_S : MOUTH_RELEASE_PER_S;
  const step = 1 - Math.exp(-rate * Math.max(0, dtSeconds));
  return current + (target - current) * step;
}

let currentMouthDriver: (() => number) | null = null;

// The value actually shown, eased toward the playing clip's target. Lives at
// module level (not inside a clip's closure) so the mouth keeps easing shut
// across the gap between two sentence clips and when a clip is stopped,
// instead of snapping to 0 the instant its driver is cleared.
let shownMouth = 0;
let lastMouthReadAt = performance.now();

/** Called from main.ts's single shared animation loop, once per frame,
 * before vrm.update(delta). Returns 0 (mouth closed) whenever nothing is
 * currently speaking -- not an error case, just the resting state. Call it
 * exactly once per frame: it advances the smoothing by the time since the
 * previous call. */
export function getMouthOpenValue(): number {
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastMouthReadAt) / 1000);
  lastMouthReadAt = now;
  const target = currentMouthDriver ? currentMouthDriver() : 0;
  shownMouth = easeMouth(shownMouth, target, dt);
  // Snap the last sliver shut so it doesn't hover just open.
  return shownMouth < 0.01 ? 0 : shownMouth;
}

// Same one-clip-at-a-time invariant as currentMouthDriver above (there's
// only ever one Audio element actually playing), tracked separately
// rather than reusing that driver's closure because captions need the
// raw elapsed/duration numbers, not a derived mouth-openness value.
// This is what lets main.ts approximate per-word caption timing against
// GPT-SoVITS's *actual* clip length instead of guessing a fixed
// words-per-second rate -- GPT-SoVITS's API returns finished WAV bytes
// with no per-word timestamps (confirmed against orchestrator/tts.py),
// so real phoneme-level sync isn't available; this is the best signal
// there is to sync against.
let currentSpeechEl: HTMLAudioElement | null = null;

/** Returns the elapsed/total seconds of whatever clip is currently
 * playing, or null if nothing is playing or its duration isn't known yet
 * (metadata hasn't loaded -- normal for the first frame or two after
 * play() is called). Callers should treat null as "don't move the
 * caption forward yet," not as an error. */
export function getSpeechProgress(): { elapsed: number; duration: number } | null {
  if (!currentSpeechEl || currentSpeechEl.paused || currentSpeechEl.ended) return null;
  const duration = currentSpeechEl.duration;
  if (!isFinite(duration) || duration <= 0) return null;
  return { elapsed: currentSpeechEl.currentTime, duration };
}

export interface SpeakHandle {
  onFinish(cb: () => void): void;
  /**
   * Stops playback immediately (the "stop response" button) and releases
   * the AnalyserNode/MediaElementSource -- but deliberately does NOT fire
   * the onFinish callback the way the natural "ended" event does.
   * SpeakQueue's stopAll() clears its whole pending queue itself and
   * resets its own state directly; if stop() also fired onFinish, that
   * callback's own playNext() call would fire a redundant, confusing
   * extra step on an already-cleared queue. Two separate paths (natural
   * end vs. manual stop) are simpler to reason about than one shared path
   * with a "but not this time" flag threaded through it.
   */
  stop(): void;
}

let sharedAudioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext {
  if (!sharedAudioCtx) {
    sharedAudioCtx = new AudioContext();
  }
  // A context created before the page has had any user interaction starts
  // "suspended", and audio routed through a suspended context is silent --
  // which is exactly what happened to an unprompted (Continuous OCR / Task
  // Guide) remark that arrived while the user was in a game and hadn't
  // touched this window since launch: SoVITS rendered it, the frontend got
  // it, nothing came out. resume() is a no-op on a running context, so it is
  // called every time; the warning makes the next occurrence self-diagnosing.
  if (sharedAudioCtx.state === "suspended") {
    console.warn("[luna] AudioContext was suspended; resuming before playback");
    sharedAudioCtx.resume().catch((err) => console.error("[luna] AudioContext resume failed", err));
  }
  return sharedAudioCtx;
}

/**
 * Plays `audioUrl` through the browser's audio output and, for as long as
 * it's playing, makes getMouthOpenValue() return a live mouth-openness
 * reading (0..MOUTH_MAX_OPEN: gated, curved and eased RMS of the waveform --
 * see the math block at the top of this file) instead of the resting 0. main.ts's shared render loop is what
 * actually applies that value to the VRM's "aa" expression each frame --
 * this function only owns the audio element and the analyser reading it,
 * not anything about the 3D scene.
 */
export function speakWithLipsync(audioUrl: string): SpeakHandle {
  const audioEl = new Audio(audioUrl);
  audioEl.crossOrigin = "anonymous";

  const ctx = getAudioContext();
  const source = ctx.createMediaElementSource(audioEl);
  const analyser = ctx.createAnalyser();
  // 1024 samples = ~21-23 ms at a 44.1/48 kHz context, enough to cover a few
  // pitch cycles. Read the FULL buffer (fftSize samples, not
  // frequencyBinCount, which is only half of it).
  analyser.fftSize = 1024;
  source.connect(analyser);
  analyser.connect(ctx.destination);

  const timeDomain = new Uint8Array(analyser.fftSize);
  let finishCallback: (() => void) | undefined;
  let cleaned = false;

  /** This clip's raw target (0..MOUTH_MAX_OPEN); the smoothing happens in
   * getMouthOpenValue() above. */
  function getMouthOpen(): number {
    if (audioEl.paused || audioEl.ended) return 0;
    analyser.getByteTimeDomainData(timeDomain);
    return mouthTargetFromRms(rmsOfTimeDomain(timeDomain));
  }

  function cleanup(): void {
    if (cleaned) return;
    cleaned = true;
    // Only clear the shared driver if it's still ours -- if something
    // else already started speaking (and became the current driver)
    // between this clip finishing and this cleanup running, clearing it
    // unconditionally would wipe the wrong one out.
    if (currentMouthDriver === getMouthOpen) currentMouthDriver = null;
    if (currentSpeechEl === audioEl) currentSpeechEl = null;
    source.disconnect();
    analyser.disconnect();
  }

  function onEnded(): void {
    cleanup();
    finishCallback?.();
  }

  currentMouthDriver = getMouthOpen;
  currentSpeechEl = audioEl;
  audioEl.addEventListener("ended", onEnded);
  audioEl.play().catch((err) => console.error("[luna] audio playback failed", err));

  return {
    onFinish(cb) {
      finishCallback = cb;
    },
    stop() {
      // No-op if it already finished naturally (onEnded already ran
      // cleanup) or was already stopped -- avoids pausing/cleaning up
      // twice for no reason.
      if (audioEl.paused || audioEl.ended) return;
      audioEl.pause();
      cleanup();
    },
  };
}
