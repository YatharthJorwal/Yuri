import { test } from "node:test";
import assert from "node:assert/strict";
import { mulberry32 } from "../src/eyes.ts";
import {
  DRIFT_DEPTH,
  EMOTION_PROFILES,
  FALLBACK_PROFILE,
  INTENSITY_MIN,
  ExpressionLife,
  blendTarget,
} from "../src/expression-life.ts";

const DT = 1 / 60;

function life(seed = 1) {
  return new ExpressionLife(mulberry32(seed));
}

// Runs the controller for `seconds` with a fixed busy flag; returns frames.
function run(l, seconds, busy) {
  const out = [];
  for (let i = 0; i < seconds / DT; i++) out.push(l.update(DT, busy));
  return out;
}

// Seconds of quiet until the frame flips to the neutral baseline.
function secondsToNeutral(emotion, seed = 1) {
  const l = life(seed);
  l.set(emotion);
  for (let i = 0; i < 120 / DT; i++) {
    if (l.update(DT, false).emotion === "neutral") return i * DT;
  }
  return Infinity;
}

test("a fresh controller is plain neutral", () => {
  const f = life().update(DT, false);
  assert.deepEqual(f, { emotion: "neutral", strength: 1 });
});

test("a mood holds at full strength while she is busy, however long", () => {
  const l = life();
  l.set("sad");
  const frames = run(l, 120, true);
  for (const f of frames) {
    assert.equal(f.emotion, "sad");
    assert.ok(f.strength >= INTENSITY_MIN * (1 - DRIFT_DEPTH) - 1e-9, `strength ${f.strength}`);
    assert.ok(f.strength <= 1);
  }
});

test("once quiet it holds for the emotion's own time, then fades, then is neutral", () => {
  const { hold, decay } = EMOTION_PROFILES.happy;
  const l = life();
  l.set("happy");
  const inHold = run(l, hold - 0.5, false);
  assert.ok(inHold.every((f) => f.emotion === "happy" && f.strength > 0.7), "still held");
  const fading = run(l, decay / 2 + 0.5, false);
  assert.ok(fading.at(-1).strength < 0.6 && fading.at(-1).emotion === "happy", "part-way faded");
  run(l, decay, false);
  assert.deepEqual(l.update(DT, false), { emotion: "neutral", strength: 1 });
});

test("the time to neutral matches hold + decay for each emotion", () => {
  for (const name of ["happy", "angry", "sad", "teasing", "surprised"]) {
    const { hold, decay } = EMOTION_PROFILES[name];
    const t = secondsToNeutral(name);
    assert.ok(Math.abs(t - (hold + decay)) < 0.1, `${name}: ${t}s vs ${hold + decay}s`);
  }
});

test("a flash of surprise passes faster than sadness", () => {
  assert.ok(secondsToNeutral("surprised") < secondsToNeutral("sad") / 2);
});

test("any pause in busyness restarts the hold, so a mood survives chunk gaps", () => {
  const l = life();
  l.set("surprised"); // hold 1.5 s
  for (let round = 0; round < 10; round++) {
    run(l, 1, false); // quiet gap shorter than the hold
    run(l, 1, true); // next chunk starts
  }
  const f = l.update(DT, true);
  assert.equal(f.emotion, "surprised");
});

test("setting the same emotion again refreshes it; a new one replaces it", () => {
  const l = life();
  l.set("happy");
  run(l, 9, false); // deep into the fade
  assert.ok(l.update(DT, false).strength < 0.7);
  l.set("happy");
  assert.ok(l.update(DT, false).strength > 0.7, "refreshed to full");
  l.set("angry");
  assert.equal(l.update(DT, false).emotion, "angry");
});

test("setting neutral snaps straight back", () => {
  const l = life();
  l.set("happy");
  run(l, 1, true);
  l.set("neutral");
  assert.deepEqual(l.update(DT, true), { emotion: "neutral", strength: 1 });
});

test("each mood starts at a slightly different intensity, within bounds", () => {
  const seen = new Set();
  for (let seed = 1; seed <= 30; seed++) {
    const l = life(seed);
    l.set("happy");
    const s = l.update(0, true).strength; // dt 0: clock stays 0, drift fixed
    seen.add(s.toFixed(4));
    assert.ok(s >= INTENSITY_MIN * (1 - DRIFT_DEPTH) - 1e-9 && s <= 1, `s ${s}`);
  }
  assert.ok(seen.size > 10, "intensity should vary between moods");
});

test("a held mood breathes but never leaves its band", () => {
  const l = life(3);
  l.set("teasing");
  const s = run(l, 120, true).map((f) => f.strength);
  assert.ok(Math.max(...s) - Math.min(...s) > 0.02, "should drift a little");
  assert.ok(Math.min(...s) >= INTENSITY_MIN * (1 - DRIFT_DEPTH) - 1e-9);
});

test("the fade is smooth: no frame-to-frame jump, including the hand-off to neutral", () => {
  const l = life(4);
  l.set("sad");
  let prevMix = 1; // weight on the mood, treating neutral as strength 0 of it
  let worst = 0;
  let first = true;
  for (let i = 0; i < 40 / DT; i++) {
    const f = l.update(DT, false);
    const mix = f.emotion === "neutral" ? 0 : f.strength;
    if (!first) worst = Math.max(worst, Math.abs(mix - prevMix));
    prevMix = mix;
    first = false;
  }
  assert.ok(worst < 0.02, `largest per-frame step ${worst}`);
});

test("an emotion with no profile falls back instead of crashing", () => {
  const l = life();
  l.set("bewildered");
  run(l, FALLBACK_PROFILE.hold - 1, false);
  assert.equal(l.update(DT, false).emotion, "bewildered");
  run(l, FALLBACK_PROFILE.decay + 2, false);
  assert.equal(l.update(DT, false).emotion, "neutral");
});

test("same seed, same face", () => {
  const trace = (seed) => {
    const l = life(seed);
    l.set("happy");
    return run(l, 30, false);
  };
  assert.deepEqual(trace(5), trace(5));
  assert.notDeepEqual(trace(5), trace(6));
});

const BLENDS = {
  happy: { happy: 0.8 },
  teasing: { relaxed: 1, sad: 0.05, angry: 0.05 },
  neutral: { neutral: 1 },
};
const PRESETS = ["happy", "angry", "sad", "relaxed", "surprised", "neutral"];

test("blendTarget: full strength is the blend, zero is neutral, half is a crossfade", () => {
  const full = blendTarget(BLENDS, { emotion: "teasing", strength: 1 }, PRESETS);
  assert.deepEqual(full, { happy: 0, angry: 0.05, sad: 0.05, relaxed: 1, surprised: 0, neutral: 0 });
  const none = blendTarget(BLENDS, { emotion: "teasing", strength: 0 }, PRESETS);
  assert.deepEqual(none, { happy: 0, angry: 0, sad: 0, relaxed: 0, surprised: 0, neutral: 1 });
  const half = blendTarget(BLENDS, { emotion: "happy", strength: 0.5 }, PRESETS);
  assert.ok(Math.abs(half.happy - 0.4) < 1e-12 && Math.abs(half.neutral - 0.5) < 1e-12);
});

test("blendTarget: every preset gets a value in 0..1, and an unknown emotion degrades to neutral", () => {
  for (const strength of [0, 0.25, 0.8, 1]) {
    const t = blendTarget(BLENDS, { emotion: "teasing", strength }, PRESETS);
    assert.deepEqual(Object.keys(t).sort(), [...PRESETS].sort());
    for (const v of Object.values(t)) assert.ok(v >= 0 && v <= 1);
  }
  const odd = blendTarget(BLENDS, { emotion: "nonsense", strength: 0.9 }, PRESETS);
  assert.ok(Math.abs(odd.neutral - 0.1) < 1e-12);
});
