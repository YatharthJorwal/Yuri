import { test } from "node:test";
import assert from "node:assert/strict";
import {
  BREATH_HZ,
  IDLE_BONE_NAMES,
  MAX_ABS_OFFSET,
  RAMP_SECONDS,
  applyIdleOffsets,
  idleOffsets,
} from "../src/idle-motion.ts";

const AXES = ["x", "y", "z"];

function* frames(seconds, fps = 60, start = 0) {
  for (let i = 0; i <= seconds * fps; i++) yield start + i / fps;
}

test("no bone ever moves past the ceiling, over ten minutes of idling", () => {
  // Spring bones jitter on big body motion; this is the guard.
  let worst = 0;
  for (const t of frames(600, 10)) {
    const o = idleOffsets(t);
    for (const name of IDLE_BONE_NAMES) for (const a of AXES) worst = Math.max(worst, Math.abs(o[name][a]));
  }
  assert.ok(worst > 0.005, "she should actually move");
  assert.ok(worst <= MAX_ABS_OFFSET, `worst ${worst}`);
});

test("intensity 0 is perfectly still, and 0.5 is half the motion", () => {
  const still = idleOffsets(37.3, 0);
  for (const name of IDLE_BONE_NAMES) for (const a of AXES) assert.equal(still[name][a], 0);
  const full = idleOffsets(37.3, 1);
  const half = idleOffsets(37.3, 0.5);
  assert.ok(Math.abs(half.chest.x - full.chest.x * 0.5) < 1e-12);
  assert.ok(Math.abs(half.head.y - full.head.y * 0.5) < 1e-12);
});

test("she eases in from rest instead of popping on boot", () => {
  const first = idleOffsets(0);
  for (const name of IDLE_BONE_NAMES) for (const a of AXES) assert.equal(first[name][a], 0);
  const mid = Math.abs(idleOffsets(RAMP_SECONDS / 2).head.y);
  const late = Math.abs(idleOffsets(RAMP_SECONDS * 10).head.y);
  assert.ok(mid <= 0.5 * 0.035 + 1e-9, "no full-size motion halfway up the ramp");
  assert.ok(late >= 0, "sanity");
});

test("motion is smooth: no frame-to-frame jump at 60 fps", () => {
  let prev = idleOffsets(0);
  let worst = 0;
  for (const t of frames(120, 60, 1 / 60)) {
    const cur = idleOffsets(t);
    for (const name of IDLE_BONE_NAMES) for (const a of AXES) worst = Math.max(worst, Math.abs(cur[name][a] - prev[name][a]));
    prev = cur;
  }
  assert.ok(worst < 0.002, `largest per-frame step ${worst} rad`);
});

test("chest breathes at the configured rate", () => {
  // Count rising zero crossings of chest pitch over 100 s after the ramp.
  let crossings = 0;
  let prev = idleOffsets(10).chest.x;
  for (const t of frames(100, 60, 10 + 1 / 60)) {
    const cur = idleOffsets(t).chest.x;
    if (prev < 0 && cur >= 0) crossings++;
    prev = cur;
  }
  const hz = crossings / 100;
  assert.ok(Math.abs(hz - BREATH_HZ) < 0.03, `measured ${hz} Hz`);
});

test("the shoulders mirror each other and the head does not repeat", () => {
  for (const t of [3, 17.2, 55.5, 120]) {
    const o = idleOffsets(t);
    assert.ok(Math.abs(o.leftShoulder.z + o.rightShoulder.z) < 1e-12, `t=${t}`);
  }
  // Unrelated frequencies: the head is not in the same place one sway period later.
  const a = idleOffsets(20).head;
  const b = idleOffsets(20 + 9.3).head;
  assert.ok(Math.abs(a.y - b.y) > 1e-4 || Math.abs(a.x - b.x) > 1e-4);
});

test("applyIdleOffsets writes every bone the model has and skips missing ones", () => {
  const written = {};
  const bone = (n) => ({ rotation: { set: (x, y, z) => (written[n] = [x, y, z]) } });
  const bones = { hips: bone("hips"), head: bone("head"), chest: null, upperChest: undefined };
  const o = idleOffsets(42);
  applyIdleOffsets(bones, o);
  assert.deepEqual(Object.keys(written).sort(), ["head", "hips"]);
  assert.deepEqual(written.head, [o.head.x, o.head.y, o.head.z]);
});
