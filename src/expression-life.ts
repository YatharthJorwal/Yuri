// Expression life for the shell face (Phase 13, step 3). Before this, the
// emotion tag set a target, the face eased to it, and the moment her audio
// went idle main.ts snapped the target back to "neutral" -- so a mood
// vanished the instant she stopped talking. A person's face keeps the mood
// for a beat after the line and then relaxes. Pure logic (no three.js, no
// DOM, randomness injected) so the timing is tested under `npm run test:fe`.
//
// How it works: set(emotion) starts a mood at a random intensity (so the
// tenth "happy" is not identical to the first). While she is busy (a turn is
// generating or her audio is playing -- main.ts passes that in) the mood
// holds at full strength. Once she is quiet it holds for the emotion's own
// time, then fades to the neutral baseline over its own decay time. While it
// holds, it breathes by a few percent so the face is never perfectly static.
//
// Output is an {emotion, strength} frame; blendTarget() turns that into the
// preset weights main.ts's existing per-frame easing chases: strength of the
// emotion's blend, plus the rest from the neutral blend, so a fade is a smooth
// crossfade and the hand-off to plain neutral is continuous.
//
// Not here: extra visemes from audio band energy (the roadmap lists it as
// optional) -- that needs the audio analyser in lipsync.ts and cannot be
// judged without a renderer.

import type { Rng } from "./eyes";

export interface EmotionProfile {
  hold: number; // seconds at full strength after she goes quiet
  decay: number; // seconds to fade to neutral after the hold
}

export const BASELINE = "neutral";

// First-guess timings, tune by eye: a flash of surprise passes quickly, a sad
// or angry look lingers and relaxes slowly.
export const EMOTION_PROFILES: Record<string, EmotionProfile> = {
  happy: { hold: 6, decay: 5 },
  angry: { hold: 5, decay: 7 },
  sad: { hold: 8, decay: 8 },
  teasing: { hold: 7, decay: 5 },
  surprised: { hold: 1.5, decay: 2.5 },
  neutral: { hold: 0, decay: 0 },
};
export const FALLBACK_PROFILE: EmotionProfile = { hold: 6, decay: 5 };

export const INTENSITY_MIN = 0.8; // each mood starts at 0.8..1.0 of its blend
export const DRIFT_DEPTH = 0.1; // the held mood breathes between 90% and 100%

export interface ExpressionFrame {
  emotion: string;
  /** 0..1 weight of `emotion`'s blend; the remainder goes to the baseline. */
  strength: number;
}

function smoothstep01(x: number): number {
  const c = Math.min(1, Math.max(0, x));
  return c * c * (3 - 2 * c);
}

// Slow wobble in [-1, 1]: three unrelated sines (periods ~ 19 s, 11 s, 7 s).
function wobble(t: number): number {
  const TAU = Math.PI * 2;
  return (
    0.5 * Math.sin(TAU * 0.053 * t + 0.9) +
    0.3 * Math.sin(TAU * 0.091 * t + 2.4) +
    0.2 * Math.sin(TAU * 0.143 * t + 4.1)
  );
}

export class ExpressionLife {
  private emotion = BASELINE;
  private intensity = 1;
  private age = 0; // seconds she has been quiet since the mood was set/last spoken
  private clock = 0;

  constructor(
    private readonly rng: Rng,
    private readonly profiles: Record<string, EmotionProfile> = EMOTION_PROFILES,
  ) {}

  /** Start (or refresh) a mood. "neutral" is the baseline and snaps back at once. */
  set(emotion: string): void {
    this.emotion = emotion;
    this.age = 0;
    this.intensity = emotion === BASELINE ? 1 : INTENSITY_MIN + (1 - INTENSITY_MIN) * this.rng();
  }

  /** `busy`: a turn is generating or her audio is playing -- the mood holds. */
  update(dt: number, busy: boolean): ExpressionFrame {
    this.clock += dt;
    if (this.emotion === BASELINE) return { emotion: BASELINE, strength: 1 };

    if (busy) this.age = 0;
    else this.age += dt;

    const p = this.profiles[this.emotion] ?? FALLBACK_PROFILE;
    let fade = 1;
    if (this.age > p.hold) {
      if (p.decay <= 0 || this.age >= p.hold + p.decay) {
        this.emotion = BASELINE;
        return { emotion: BASELINE, strength: 1 };
      }
      fade = 1 - smoothstep01((this.age - p.hold) / p.decay);
    }
    const drift = 1 - DRIFT_DEPTH * (0.5 + 0.5 * wobble(this.clock));
    return { emotion: this.emotion, strength: fade * this.intensity * drift };
  }
}

/**
 * Preset weights for this frame: the emotion's blend scaled by its strength,
 * plus the baseline's blend for the rest. Every preset in `presets` gets a
 * value, so main.ts can ease all of them.
 */
export function blendTarget<P extends string>(
  blends: Record<string, Partial<Record<P, number>>>,
  frame: ExpressionFrame,
  presets: readonly P[],
  baseline: string = BASELINE,
): Record<P, number> {
  const mood = blends[frame.emotion] ?? {};
  const rest = blends[baseline] ?? {};
  const out = {} as Record<P, number>;
  for (const preset of presets) {
    out[preset] = (mood[preset] ?? 0) * frame.strength + (rest[preset] ?? 0) * (1 - frame.strength);
  }
  return out;
}
