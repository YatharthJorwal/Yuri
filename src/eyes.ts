// Eyes for the shell body (Phase 13, step 2): a gaze with micro-saccades and
// occasional glances, and a blink with variety. Pure logic -- no three.js, no
// DOM, randomness injected -- so timing and bounds are tested under
// `npm run test:fe`. main.ts reads one frame from each controller per
// animate() tick and writes it to vrm.lookAt (yaw/pitch) and the "blink"
// expression; the head follows glances (not saccades) through addHeadFollow().
//
// Units: gaze angles are the INPUT to three-vrm's lookAt, in degrees. The
// rig then range-maps them to eye-bone rotation, and for the VRM default
// (10 degrees of eye at 90 degrees of input) the eyes only move about a
// ninth of what is asked. So the numbers below look large on purpose.
// Unverified against this model's own range maps -- tune by eye, they are
// plain exported constants.
//
// Pitch sign: three-vrm drives the eye bone with +X for positive pitch and
// the head bone takes +X as "tip forward", so eyes and head agree whatever
// the preset names say. Which of those looks up is what PITCH_UP encodes
// (reading the source it is -1: negative pitch looks up). If glances look
// down where she should look up, flip PITCH_UP.

export type Rng = () => number;

/** Small deterministic PRNG so tests (and any replay) are reproducible. */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function between(rng: Rng, lo: number, hi: number): number {
  return lo + (hi - lo) * rng();
}

function smoothstep01(x: number): number {
  const c = Math.min(1, Math.max(0, x));
  return c * c * (3 - 2 * c);
}

// ---------------------------------------------------------------- blinking

export interface BlinkShape {
  close: number; // seconds closing
  hold: number; // seconds fully shut
  open: number; // seconds opening (slower than closing, like a real lid)
}

export const BLINK_SINGLE: BlinkShape = { close: 0.07, hold: 0, open: 0.11 };
export const BLINK_QUICK: BlinkShape = { close: 0.055, hold: 0, open: 0.09 };
export const BLINK_SLOW: BlinkShape = { close: 0.16, hold: 0.12, open: 0.26 };

export const BLINK_INTERVAL_MIN_S = 2.2;
export const BLINK_INTERVAL_MAX_S = 5.5;
export const BLINK_SPEAKING_INTERVAL_SCALE = 1.6; // fewer blinks while talking
export const BLINK_RELAXED_INTERVAL_SCALE = 1.15;
export const RELAXED_AFTER_QUIET_S = 15; // quiet this long = relaxed
export const BLINK_SLOW_CHANCE = 0.12; // relaxed only
export const BLINK_DOUBLE_CHANCE = 0.14;
export const BLINK_DOUBLE_CHANCE_SPEAKING = 0.05;
export const BLINK_DOUBLE_GAP_S = 0.09;

function shapeDuration(s: BlinkShape): number {
  return s.close + s.hold + s.open;
}

function shapeWeight(s: BlinkShape, u: number): number {
  if (u < 0) return 0;
  if (u < s.close) return smoothstep01(u / s.close);
  u -= s.close;
  if (u < s.hold) return 1;
  u -= s.hold;
  if (u < s.open) return smoothstep01(1 - u / s.open);
  return 0;
}

interface BlinkEvent {
  shape: BlinkShape;
  at: number; // seconds after the sequence started
}

export class BlinkController {
  private nextIn: number;
  private active = false;
  private clock = 0;
  private endsAt = 0;
  private events: BlinkEvent[] = [];
  private quiet = 0;
  private sinceEnd = 10;

  constructor(private readonly rng: Rng) {
    this.nextIn = between(rng, 1.5, 3.5);
  }

  /** A blink on demand (the gaze asks for one when she glances away). Dropped
   * if one is already under way or one just finished. */
  requestBlink(): void {
    if (this.active || this.sinceEnd < 0.25) return;
    this.start([{ shape: BLINK_QUICK, at: 0 }]);
  }

  /** Eyelid closure 0..1 for this frame. */
  update(dt: number, speaking: boolean): number {
    this.quiet = speaking ? 0 : this.quiet + dt;
    const relaxed = !speaking && this.quiet > RELAXED_AFTER_QUIET_S;

    if (this.active) {
      this.clock += dt;
      let w = 0;
      for (const e of this.events) w = Math.max(w, shapeWeight(e.shape, this.clock - e.at));
      if (this.clock >= this.endsAt) {
        this.active = false;
        this.sinceEnd = 0;
        this.nextIn = this.interval(speaking, relaxed);
      }
      return Math.min(1, Math.max(0, w));
    }

    this.sinceEnd += dt;
    this.nextIn -= dt;
    if (this.nextIn <= 0) this.start(this.pick(speaking, relaxed));
    return 0;
  }

  private interval(speaking: boolean, relaxed: boolean): number {
    const scale = speaking ? BLINK_SPEAKING_INTERVAL_SCALE : relaxed ? BLINK_RELAXED_INTERVAL_SCALE : 1;
    return between(this.rng, BLINK_INTERVAL_MIN_S, BLINK_INTERVAL_MAX_S) * scale;
  }

  private pick(speaking: boolean, relaxed: boolean): BlinkEvent[] {
    const r = this.rng();
    if (relaxed && r < BLINK_SLOW_CHANCE) return [{ shape: BLINK_SLOW, at: 0 }];
    const doubleChance = speaking ? BLINK_DOUBLE_CHANCE_SPEAKING : BLINK_DOUBLE_CHANCE;
    if (r < (relaxed ? BLINK_SLOW_CHANCE : 0) + doubleChance) {
      const d = shapeDuration(BLINK_QUICK);
      return [
        { shape: BLINK_QUICK, at: 0 },
        { shape: BLINK_QUICK, at: d + BLINK_DOUBLE_GAP_S },
      ];
    }
    return [{ shape: BLINK_SINGLE, at: 0 }];
  }

  private start(events: BlinkEvent[]): void {
    this.events = events;
    this.clock = 0;
    this.endsAt = Math.max(...events.map((e) => e.at + shapeDuration(e.shape)));
    this.active = true;
  }
}

// ------------------------------------------------------------------- gaze

export const PITCH_UP = -1; // see the header: sign that looks up. UNVERIFIED.
export const GAZE_MAX_YAW_DEG = 80;
export const GAZE_MAX_PITCH_DEG = 30;

export const SACCADE_YAW_DEG = 12;
export const SACCADE_PITCH_DEG = 7;
export const SACCADE_INTERVAL_MIN_S = 0.5;
export const SACCADE_INTERVAL_MAX_S = 2.2;
export const SACCADE_SPEAKING_SCALE = 0.6; // steadier while she addresses you

export const GLANCE_YAW_MIN_DEG = 30;
export const GLANCE_YAW_MAX_DEG = 70;
export const GLANCE_PITCH_MAX_DEG = 14;
export const GLANCE_UP_CHANCE = 0.6; // thinking looks go up more than down
export const GLANCE_SPEAKING_SCALE = 0.6; // smaller glances while speaking
export const GLANCE_INTERVAL_S: [number, number] = [7, 16];
export const GLANCE_SPEAKING_INTERVAL_S: [number, number] = [14, 28];
export const GLANCE_HOLD_S: [number, number] = [0.8, 2.0];
export const GLANCE_SPEAKING_HOLD_S: [number, number] = [0.5, 1.2];
export const GLANCE_BLINK_CHANCE = 0.6; // a glance often comes with a blink

// Exponential-smoothing rates (per second): eyes snap, the head lags them.
export const SACCADE_RATE = 30;
export const EYE_GLANCE_RATE = 14;
export const EYE_RETURN_RATE = 9;
export const HEAD_RATE = 5;

// How much of a glance the head takes up (radians of head per radian of
// input). Saccades never reach the head.
export const HEAD_FOLLOW_YAW = 0.15;
export const HEAD_FOLLOW_PITCH = 0.2;

export interface GazeFrame {
  /** lookAt input, degrees: glance + micro-saccade, eyes only. */
  yaw: number;
  pitch: number;
  /** The glance on its own, eased slowly, in the same units: drives the head. */
  headYaw: number;
  headPitch: number;
}

interface Angles {
  yaw: number;
  pitch: number;
}

function ease(cur: number, target: number, rate: number, dt: number): number {
  return cur + (target - cur) * (1 - Math.exp(-rate * dt));
}

function clamp(v: number, lim: number): number {
  return Math.max(-lim, Math.min(lim, v));
}

export class GazeController {
  private glance: Angles = { yaw: 0, pitch: 0 }; // where she means to look
  private offset: Angles = { yaw: 0, pitch: 0 }; // micro-saccade on top
  private eyeBase: Angles = { yaw: 0, pitch: 0 };
  private eyeOffset: Angles = { yaw: 0, pitch: 0 };
  private head: Angles = { yaw: 0, pitch: 0 };
  private eyeBaseRate = EYE_GLANCE_RATE;
  private saccadeIn: number;
  private glanceIn: number;
  private holdLeft = 0;
  private blinkWanted = false;

  constructor(private readonly rng: Rng) {
    this.saccadeIn = between(rng, 0.4, 1.4);
    this.glanceIn = between(rng, 4, 9);
  }

  /** True once per glance that wants a blink with it; reading clears it. */
  consumeBlinkRequest(): boolean {
    const v = this.blinkWanted;
    this.blinkWanted = false;
    return v;
  }

  update(dt: number, speaking: boolean): GazeFrame {
    if (this.holdLeft > 0) {
      this.holdLeft -= dt;
      if (this.holdLeft <= 0) this.endGlance(speaking);
    } else {
      this.glanceIn -= dt;
      if (this.glanceIn <= 0) this.startGlance(speaking);
    }

    this.saccadeIn -= dt;
    if (this.saccadeIn <= 0) {
      const steady = speaking ? SACCADE_SPEAKING_SCALE : 1;
      const pull = this.rng() < 0.65 ? 0.5 : 1; // most saccades are the small ones
      this.offset = {
        yaw: between(this.rng, -SACCADE_YAW_DEG, SACCADE_YAW_DEG) * steady * pull,
        pitch: between(this.rng, -SACCADE_PITCH_DEG, SACCADE_PITCH_DEG) * steady * pull,
      };
      this.saccadeIn = between(this.rng, SACCADE_INTERVAL_MIN_S, SACCADE_INTERVAL_MAX_S);
    }

    this.eyeBase.yaw = ease(this.eyeBase.yaw, this.glance.yaw, this.eyeBaseRate, dt);
    this.eyeBase.pitch = ease(this.eyeBase.pitch, this.glance.pitch, this.eyeBaseRate, dt);
    this.eyeOffset.yaw = ease(this.eyeOffset.yaw, this.offset.yaw, SACCADE_RATE, dt);
    this.eyeOffset.pitch = ease(this.eyeOffset.pitch, this.offset.pitch, SACCADE_RATE, dt);
    this.head.yaw = ease(this.head.yaw, this.glance.yaw, HEAD_RATE, dt);
    this.head.pitch = ease(this.head.pitch, this.glance.pitch, HEAD_RATE, dt);

    return {
      yaw: clamp(this.eyeBase.yaw + this.eyeOffset.yaw, GAZE_MAX_YAW_DEG),
      pitch: clamp(this.eyeBase.pitch + this.eyeOffset.pitch, GAZE_MAX_PITCH_DEG),
      headYaw: this.head.yaw,
      headPitch: this.head.pitch,
    };
  }

  private startGlance(speaking: boolean): void {
    const scale = speaking ? GLANCE_SPEAKING_SCALE : 1;
    const dir = this.rng() < 0.5 ? -1 : 1;
    const up = this.rng() < GLANCE_UP_CHANCE ? PITCH_UP : -PITCH_UP;
    this.glance = {
      yaw: dir * between(this.rng, GLANCE_YAW_MIN_DEG, GLANCE_YAW_MAX_DEG) * scale,
      pitch: up * between(this.rng, 0, GLANCE_PITCH_MAX_DEG) * scale,
    };
    const [lo, hi] = speaking ? GLANCE_SPEAKING_HOLD_S : GLANCE_HOLD_S;
    this.holdLeft = between(this.rng, lo, hi);
    this.eyeBaseRate = EYE_GLANCE_RATE;
    if (this.rng() < GLANCE_BLINK_CHANCE) this.blinkWanted = true;
  }

  private endGlance(speaking: boolean): void {
    this.glance = { yaw: 0, pitch: 0 };
    this.eyeBaseRate = EYE_RETURN_RATE;
    const [lo, hi] = speaking ? GLANCE_SPEAKING_INTERVAL_S : GLANCE_INTERVAL_S;
    this.glanceIn = between(this.rng, lo, hi);
  }
}

// ------------------------------------------------------------- head follow

const DEG2RAD = Math.PI / 180;

/** Anything with mutable rotation.x / rotation.y: a normalized VRM bone, or a test double. */
export type FollowBone = { rotation: { x: number; y: number } } | null | undefined;

/**
 * Add the head's share of the current glance on top of whatever the idle
 * layer wrote this frame (call after applyIdleOffsets, which resets those
 * bones each frame). The neck takes 40% and the head 60%, as in idle-motion.
 */
export function addHeadFollow(bones: { neck?: FollowBone; head?: FollowBone }, g: GazeFrame): void {
  const yaw = g.headYaw * DEG2RAD * HEAD_FOLLOW_YAW;
  const pitch = g.headPitch * DEG2RAD * HEAD_FOLLOW_PITCH;
  if (bones.neck) {
    bones.neck.rotation.y += 0.4 * yaw;
    bones.neck.rotation.x += 0.4 * pitch;
  }
  if (bones.head) {
    bones.head.rotation.y += 0.6 * yaw;
    bones.head.rotation.x += 0.6 * pitch;
  }
}
