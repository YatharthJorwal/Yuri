import { test } from "node:test";
import assert from "node:assert/strict";
import { buildVrmReport } from "../src/vrm-report.ts";

const expr = (expressionName, extra = {}) => ({
  expressionName,
  isBinary: false,
  overrideMouth: "none",
  overrideBlink: "none",
  overrideLookAt: "none",
  ...extra,
});

function fakeVrm() {
  const names = ["aa", "ih", "ou", "ee", "oh", "blink", "happy", "angry", "sad", "relaxed", "surprised", "pout"];
  return {
    meta: { metaVersion: "1", name: "Luna" },
    expressionManager: {
      expressions: names.map((n) => expr(n, n === "happy" ? { overrideMouth: "blend" } : {})),
    },
    lookAt: { autoUpdate: true, applier: { constructor: { name: "VRMLookAtBoneApplier" } } },
    humanoid: { humanBones: { hips: {}, spine: {}, chest: {}, neck: {}, head: {}, leftEye: {}, rightEye: {}, jaw: undefined } },
    springBoneManager: { joints: new Set([1, 2, 3, 4]) },
  };
}

test("reports expressions, custom ones and which standard presets are absent", () => {
  const r = buildVrmReport(fakeVrm());
  assert.match(r.text, /Luna \(VRM spec 1\)/);
  assert.match(r.text, /custom expressions: pout/);
  assert.match(r.text, /NOT present: .*blinkLeft/);
  assert.match(r.text, /happy\(mouth:blend/);
});

test("reports gaze driver, spring joints and missing bones", () => {
  const r = buildVrmReport(fakeVrm());
  assert.match(r.text, /VRMLookAtBoneApplier, autoUpdate=true/);
  assert.match(r.text, /spring-bone joints.*: 4/);
  assert.match(r.text, /missing: .*jaw/);
  assert.match(r.text, /missing: .*upperChest/);
});

test("never throws on an empty or hostile object", () => {
  assert.doesNotThrow(() => buildVrmReport({}));
  const hostile = new Proxy({}, { get() { throw new Error("boom"); } });
  assert.doesNotThrow(() => buildVrmReport(hostile));
  assert.match(buildVrmReport({}).text, /expressions \(0\): none/);
});
