import { test } from "node:test";
import assert from "node:assert/strict";
import {
  MOUTH_MAX_OPEN,
  MOUTH_NOISE_GATE,
  MOUTH_RMS_FOR_MAX,
  easeMouth,
  getMouthOpenValue,
  mouthTargetFromRms,
  rmsOfTimeDomain,
} from "../src/lipsync.ts";

// The real session: mouth snapped to full "aa" (fangs + tongue visible).
// The old driver was min(1, rms * 4) over a ~3 ms window with no smoothing.

test("the mouth never opens past MOUTH_MAX_OPEN, however loud", () => {
  for (const rms of [0.1, 0.2, MOUTH_RMS_FOR_MAX, 0.5, 1, 5]) {
    assert.ok(mouthTargetFromRms(rms) <= MOUTH_MAX_OPEN + 1e-9, `rms ${rms}`);
  }
  assert.ok(MOUTH_MAX_OPEN < 1, "full 'aa' must never be reachable");
});

test("old driver would have hit 1.0 for ordinary loud speech; the new one does not", () => {
  const oldDriver = (rms) => Math.min(1, rms * 4);
  assert.equal(oldDriver(0.28), 1);
  assert.ok(mouthTargetFromRms(0.28) <= MOUTH_MAX_OPEN);
});

test("silence and room tone keep the mouth shut", () => {
  assert.equal(mouthTargetFromRms(0), 0);
  assert.equal(mouthTargetFromRms(MOUTH_NOISE_GATE), 0);
  assert.equal(mouthTargetFromRms(MOUTH_NOISE_GATE / 2), 0);
});

test("louder never means a smaller opening", () => {
  let last = -1;
  for (let rms = 0; rms <= 0.4; rms += 0.01) {
    const v = mouthTargetFromRms(rms);
    assert.ok(v >= last - 1e-12, `non-monotonic at ${rms}`);
    last = v;
  }
});

test("quiet speech still visibly moves the mouth", () => {
  assert.ok(mouthTargetFromRms(0.05) > 0.08);
});

test("frame-to-frame jitter is smoothed instead of flapping 0..1", () => {
  // loud/quiet alternating every frame at 60 fps -- what raw per-frame
  // waveform reads looked like.
  let m = 0;
  const out = [];
  for (let i = 0; i < 90; i++) {
    m = easeMouth(m, mouthTargetFromRms(i % 2 ? 0.25 : 0.03), 1 / 60);
    out.push(m);
  }
  const tail = out.slice(30);
  const swing = Math.max(...tail) - Math.min(...tail);
  assert.ok(swing < 0.15, `swing ${swing}`);
});

test("mouth closes within about a third of a second after speech stops", () => {
  let m = MOUTH_MAX_OPEN;
  let frames = 0;
  while (m >= 0.01 && frames < 600) {
    m = easeMouth(m, 0, 1 / 60);
    frames++;
  }
  assert.ok(frames / 60 < 0.4, `${frames / 60}s`);
});

test("easing is frame-rate independent", () => {
  let a = 0;
  for (let i = 0; i < 30; i++) a = easeMouth(a, 0.5, 1 / 30);
  let b = 0;
  for (let i = 0; i < 120; i++) b = easeMouth(b, 0.5, 1 / 120);
  assert.ok(Math.abs(a - b) < 1e-9);
});

test("rmsOfTimeDomain: silence is 0, a full-scale square wave is ~1", () => {
  assert.equal(rmsOfTimeDomain(new Uint8Array(1024).fill(128)), 0);
  const square = new Uint8Array(1024).map((_, i) => (i % 2 ? 255 : 0));
  assert.ok(rmsOfTimeDomain(square) > 0.98);
  assert.equal(rmsOfTimeDomain(new Uint8Array(0)), 0);
});

test("with nothing playing, the shared mouth value is 0", () => {
  for (let i = 0; i < 5; i++) assert.equal(getMouthOpenValue(), 0);
});
