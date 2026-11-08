import { test } from "node:test";
import assert from "node:assert/strict";
import { lookAtHeight } from "../src/framing.ts";

// The shell's real framing: 32 degree FOV from 1.2 m, so +/-0.344 m is visible around the target.
const F = { distance: 1.2, fovDegrees: 32, headroom: 0.05, maxRise: 0.2 };
const halfVisible = 1.2 * Math.tan((32 * Math.PI) / 360);

test("a model that already fits is framed exactly as before (look at the head bone)", () => {
  assert.equal(lookAtHeight(1.4, 1.62, F), 1.4);
});

test("tall bunny ears raise the frame just enough to fit, with the headroom left over", () => {
  const head = 1.4;
  const earTip = head + 0.4; // taller than the 0.344 m the frame shows above the head bone
  const y = lookAtHeight(head, earTip, F);
  assert.ok(y > head, "camera must pan up");
  assert.ok(Math.abs(y + halfVisible - (earTip + F.headroom)) < 1e-9, "top edge sits headroom above the ears");
});

test("a freak tall mesh cannot drag the face out of the frame: the rise is capped", () => {
  assert.equal(lookAtHeight(1.4, 5, F), 1.4 + F.maxRise);
});

test("it never looks below the head bone, even for a short model", () => {
  assert.equal(lookAtHeight(1.4, 1.5, F), 1.4);
});

test("a missing or broken bounding box falls back to the head bone", () => {
  for (const bad of [Number.NaN, -Infinity, Infinity]) assert.equal(lookAtHeight(1.4, bad, F), 1.4);
  assert.equal(lookAtHeight(Number.NaN, 1.8, F), Number.NaN);
});

test("moving the frame up by the returned amount puts the ears fully inside it", () => {
  const head = 1.35;
  const earTip = head + 0.38;
  const y = lookAtHeight(head, earTip, F);
  assert.ok(y + halfVisible >= earTip, "ear tip inside the top edge");
});
