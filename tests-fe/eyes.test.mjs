import { test } from "node:test";
import assert from "node:assert/strict";
import {
  BLINK_INTERVAL_MAX_S,
  BlinkController,
  GAZE_MAX_PITCH_DEG,
  GAZE_MAX_YAW_DEG,
  GazeController,
  HEAD_FOLLOW_YAW,
  addHeadFollow,
  mulberry32,
} from "../src/eyes.ts";

const DT = 1 / 60;

// Runs a blink controller; returns the weight trace.
function blinkTrace(seed, seconds, speaking) {
  const b = new BlinkController(mulberry32(seed));
  const out = [];
  for (let i = 0; i < seconds / DT; i++) out.push(b.update(DT, speaking));
  return out;
}

// Rising edges (closed -> starting to close) and the run length of each blink.
function blinkRuns(trace) {
  const runs = [];
  let start = -1;
  for (let i = 0; i < trace.length; i++) {
    if (trace[i] > 0.02 && start < 0) start = i;
    if (trace[i] <= 0.02 && start >= 0) {
      runs.push({ at: start * DT, len: (i - start) * DT, peak: Math.max(...trace.slice(start, i)) });
      start = -1;
    }
  }
  return runs;
}

test("blink weight stays in 0..1 and a blink actually closes the eye", () => {
  const trace = blinkTrace(1, 300, false);
  assert.ok(Math.min(...trace) >= 0);
  assert.ok(Math.max(...trace) <= 1);
  assert.ok(Math.max(...trace) > 0.99, "a blink must reach fully shut");
});

test("resting blink rate is human: roughly 10-25 a minute", () => {
  const runs = blinkRuns(blinkTrace(7, 600, false));
  const perMin = runs.length / 10;
  assert.ok(perMin > 8 && perMin < 26, `${perMin}/min`);
});

test("she blinks less while speaking", () => {
  const idle = blinkRuns(blinkTrace(11, 600, false)).length;
  const talk = blinkRuns(blinkTrace(11, 600, true)).length;
  assert.ok(talk < idle * 0.8, `speaking ${talk} vs idle ${idle}`);
});

test("slow blinks happen only once she has been quiet a while", () => {
  const idle = blinkRuns(blinkTrace(5, 600, false));
  assert.ok(idle.some((r) => r.len > 0.4), "relaxed idling should produce slow blinks");
  const talk = blinkRuns(blinkTrace(5, 600, true));
  assert.ok(talk.every((r) => r.len < 0.45), "no slow blink while speaking");
  // And not in the first 15 s of quiet.
  const early = blinkRuns(blinkTrace(5, 14, false));
  assert.ok(early.every((r) => r.len < 0.4));
});

test("double blinks occur, and the gap between blinks never exceeds the schedule", () => {
  const runs = blinkRuns(blinkTrace(21, 600, false));
  let doubles = 0;
  for (let i = 1; i < runs.length; i++) if (runs[i].at - (runs[i - 1].at + runs[i - 1].len) < 0.2) doubles++;
  assert.ok(doubles > 0, "expected at least one double blink in ten minutes");
  for (let i = 1; i < runs.length; i++) {
    const gap = runs[i].at - (runs[i - 1].at + runs[i - 1].len);
    assert.ok(gap < BLINK_INTERVAL_MAX_S * 1.6 + 0.5, `gap ${gap}`);
  }
});

test("requestBlink blinks now when idle, and is ignored mid-blink", () => {
  const b = new BlinkController(mulberry32(3));
  for (let i = 0; i < 30; i++) b.update(DT, false); // let the opening interval pass its start
  const fresh = new BlinkController(mulberry32(3));
  fresh.requestBlink();
  let peak = 0;
  for (let i = 0; i < 20; i++) peak = Math.max(peak, fresh.update(DT, false));
  assert.ok(peak > 0.9, "a requested blink should close the eye within a third of a second");
  // A second request during that blink does not restart it.
  const c = new BlinkController(mulberry32(3));
  c.requestBlink();
  c.update(DT, false);
  const before = c.update(DT, false);
  c.requestBlink();
  const after = c.update(DT, false);
  assert.ok(after >= before, "still closing, not restarted");
});

function gazeTrace(seed, seconds, speaking) {
  const g = new GazeController(mulberry32(seed));
  const out = [];
  let blinkRequests = 0;
  for (let i = 0; i < seconds / DT; i++) {
    out.push(g.update(DT, speaking));
    if (g.consumeBlinkRequest()) blinkRequests++;
  }
  return { out, blinkRequests };
}

test("gaze never leaves the rig's range, over ten minutes", () => {
  const { out } = gazeTrace(2, 600, false);
  assert.ok(Math.max(...out.map((f) => Math.abs(f.yaw))) <= GAZE_MAX_YAW_DEG);
  assert.ok(Math.max(...out.map((f) => Math.abs(f.pitch))) <= GAZE_MAX_PITCH_DEG);
});

test("same seed, same eyes: the controllers are deterministic", () => {
  const a = gazeTrace(9, 60, false).out;
  const b = gazeTrace(9, 60, false).out;
  assert.deepEqual(a, b);
  const c = gazeTrace(10, 60, false).out;
  assert.notDeepEqual(a, c);
});

test("she glances away now and then, and is back on you most of the time", () => {
  const { out } = gazeTrace(4, 600, false);
  assert.ok(Math.max(...out.map((f) => Math.abs(f.headYaw))) > 15, "expected real glances");
  const home = out.filter((f) => Math.abs(f.headYaw) < 4).length / out.length;
  assert.ok(home > 0.6, `home ${home}`);
});

function glanceStarts(trace) {
  let n = 0;
  let away = false;
  for (const f of trace) {
    if (!away && Math.abs(f.headYaw) > 12) {
      n++;
      away = true;
    }
    if (away && Math.abs(f.headYaw) < 4) away = false;
  }
  return n;
}

test("she glances away less while speaking", () => {
  const idle = glanceStarts(gazeTrace(6, 600, false).out);
  const talk = glanceStarts(gazeTrace(6, 600, true).out);
  assert.ok(talk < idle * 0.8, `speaking ${talk} vs idle ${idle}`);
});

test("eyes move in quick steps, the head moves slowly", () => {
  const { out } = gazeTrace(8, 300, false);
  let eyeStep = 0;
  let headStep = 0;
  for (let i = 1; i < out.length; i++) {
    eyeStep = Math.max(eyeStep, Math.abs(out[i].yaw - out[i - 1].yaw));
    headStep = Math.max(headStep, Math.abs(out[i].headYaw - out[i - 1].headYaw));
  }
  // Units are lookAt input degrees (the rig maps ~9:1 down to eye rotation).
  // The worst eye frame is the first one of a 70-degree glance (rate 14/s),
  // ~14.5; the worst head frame is the same glance at rate 5/s, ~5.6.
  assert.ok(eyeStep < 16, `eye step ${eyeStep} deg/frame`);
  assert.ok(headStep < 6.2, `head step ${headStep} deg/frame`);
  assert.ok(headStep < eyeStep * 0.5, "the head lags the eyes");
});

test("some glances ask for a blink, and reading the request clears it", () => {
  const { blinkRequests } = gazeTrace(12, 600, false);
  assert.ok(blinkRequests > 5, `${blinkRequests}`);
  const g = new GazeController(mulberry32(1));
  assert.equal(g.consumeBlinkRequest(), false);
});

test("micro-saccades never reach the head", () => {
  // With glances impossible (a controller we only run for 3 s, before the
  // first glance can start at >= 4 s) the eyes still move but the head does not.
  const g = new GazeController(mulberry32(15));
  let eyeMax = 0;
  let headMax = 0;
  for (let i = 0; i < 3 / DT; i++) {
    const f = g.update(DT, false);
    eyeMax = Math.max(eyeMax, Math.abs(f.yaw));
    headMax = Math.max(headMax, Math.abs(f.headYaw));
  }
  assert.ok(eyeMax > 1, "the eyes should be moving");
  assert.equal(headMax, 0);
});

test("addHeadFollow adds on top of the idle layer and leaves z alone", () => {
  const bone = () => ({ rotation: { x: 0.01, y: 0.02, z: 0.03 } });
  const neck = bone();
  const head = bone();
  addHeadFollow({ neck, head }, { yaw: 0, pitch: 0, headYaw: 40, headPitch: 0 });
  const yaw = 40 * (Math.PI / 180) * HEAD_FOLLOW_YAW;
  assert.ok(Math.abs(neck.rotation.y - (0.02 + 0.4 * yaw)) < 1e-12);
  assert.ok(Math.abs(head.rotation.y - (0.02 + 0.6 * yaw)) < 1e-12);
  assert.equal(head.rotation.z, 0.03);
  addHeadFollow({ neck: null, head: undefined }, { yaw: 0, pitch: 0, headYaw: 40, headPitch: 10 });
});
