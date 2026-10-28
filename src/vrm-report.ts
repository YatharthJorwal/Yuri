// One-time capability report for the loaded VRM, printed to the devtools
// console at boot (and kept on `window.__lunaVrmReport` so it can be copied
// after the fact). It changes nothing about how she looks or moves.
//
// Why it exists: the "living avatar" phase (docs/ROADMAP.md, Phase 13) wants
// breathing, gaze, yawns, pouts and expression cycles, and what is possible
// depends entirely on what THIS model exports -- which expressions exist
// (does it have a pout? extra visemes?), whether eyes are driven by bones or
// by expressions, whether there are eye/jaw/chest bones to move, and how many
// spring-bone joints (hair) will react to body motion. The dev sandbox has no
// copy of the model, so nobody can answer that from code alone; this answers
// it on the real machine in one paste.

import type { VRM } from "@pixiv/three-vrm";

export interface VrmReport {
  text: string;
  data: Record<string, unknown>;
}

function safe<T>(fn: () => T, fallback: T): T {
  try {
    return fn();
  } catch {
    return fallback;
  }
}

/** Never throws: a report problem must not be able to stop the avatar booting. */
export function buildVrmReport(vrm: VRM): VrmReport {
  const meta = safe(() => vrm.meta as unknown as Record<string, unknown> | undefined, undefined);
  const specVersion = safe(() => String(meta?.metaVersion ?? "?"), "?");
  const name = safe(() => String(meta?.name ?? meta?.title ?? "?"), "?");

  const expressions = safe(
    () =>
      (vrm.expressionManager?.expressions ?? []).map((e) => ({
        name: e.expressionName,
        binary: e.isBinary,
        overrideMouth: e.overrideMouth,
        overrideBlink: e.overrideBlink,
        overrideLookAt: e.overrideLookAt,
      })),
    [] as { name: string; binary: boolean; overrideMouth: string; overrideBlink: string; overrideLookAt: string }[],
  );

  const lookAt = safe(() => {
    const la = vrm.lookAt;
    if (!la) return "none";
    const applier = (la as unknown as { applier?: { constructor?: { name?: string } } }).applier;
    return `${applier?.constructor?.name ?? "unknown applier"}, autoUpdate=${String(la.autoUpdate)}`;
  }, "unknown");

  const bones = safe(() => {
    const table = vrm.humanoid?.humanBones as unknown as Record<string, unknown> | undefined;
    return Object.keys(table ?? {})
      .filter((k) => table?.[k])
      .sort();
  }, [] as string[]);

  const springJoints = safe(() => vrm.springBoneManager?.joints.size ?? 0, 0);

  const wanted = ["spine", "chest", "upperChest", "neck", "head", "leftEye", "rightEye", "jaw", "leftShoulder", "rightShoulder"];
  const missingBones = wanted.filter((b) => !bones.includes(b));

  const exprNames = expressions.map((e) => e.name);
  const preset = ["aa", "ih", "ou", "ee", "oh", "blink", "blinkLeft", "blinkRight", "happy", "angry", "sad", "relaxed", "surprised", "neutral", "lookUp", "lookDown", "lookLeft", "lookRight"];
  const missingPresets = preset.filter((p) => !exprNames.includes(p));
  const custom = exprNames.filter((n) => !preset.includes(n));

  const lines = [
    `[luna] VRM report -- ${name} (VRM spec ${specVersion})`,
    `  expressions (${exprNames.length}): ${exprNames.join(", ") || "none"}`,
    `  custom expressions: ${custom.join(", ") || "none"}`,
    `  standard presets NOT present: ${missingPresets.join(", ") || "none"}`,
    `  mouth/blink/gaze overrides: ${
      expressions
        .filter((e) => e.overrideMouth !== "none" || e.overrideBlink !== "none" || e.overrideLookAt !== "none")
        .map((e) => `${e.name}(mouth:${e.overrideMouth} blink:${e.overrideBlink} look:${e.overrideLookAt})`)
        .join(", ") || "none"
    }`,
    `  lookAt: ${lookAt}`,
    `  humanoid bones present (${bones.length}); of the ones a living idle wants, missing: ${missingBones.join(", ") || "none"}`,
    `  spring-bone joints (hair/clothes physics): ${springJoints}`,
  ];
  return {
    text: lines.join("\n"),
    data: { name, specVersion, expressions, lookAt, bones, springJoints, missingBones, missingPresets, custom },
  };
}

/** Logs the report and stores it on window. Safe to call once after load. */
export function logVrmReport(vrm: VRM): void {
  try {
    const report = buildVrmReport(vrm);
    console.log(report.text);
    (window as unknown as { __lunaVrmReport?: VrmReport }).__lunaVrmReport = report;
  } catch (err) {
    console.warn("[luna] VRM report failed (harmless):", err);
  }
}
