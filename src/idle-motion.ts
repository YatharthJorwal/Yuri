// Procedural idle motion for the shell body (Phase 13, step 1): breathing,
// a slow weight shift, and a drifting head, so she stops looking like a
// mannequin between replies. Pure math -- no three.js, no DOM -- so the
// shape of the motion is tested under `npm run test:fe`; main.ts only reads
// the offsets once a frame and writes them onto the VRM's normalized bones.
//
// Every bone driven here is one applyIdlePose() (main.ts) never touches, so
// offsets are absolute Euler rotations on a rest-pose (identity) bone, not
// deltas stacked on a pose. If a later step drives these bones from a VRMA
// clip, switch to additive quaternions at that point.
//
// Amplitudes are deliberately tiny (radians; 0.02 rad is ~1 degree). Body
// motion also drives the hair/clothes spring bones, and a big or fast sway
// makes them jitter -- the ceiling is MAX_ABS_OFFSET and a test enforces it.
//
// Signs follow the VRM normalized-bone convention (she faces +Z; +X pitch
// tips the chest forward). UNVERIFIED on a real render: if she slumps on the
// inhale instead of lifting, flip BREATH_SIGN.

export const BREATH_HZ = 0.26; // ~3.8 s per breath, a calm resting rate
export const BREATH_SIGN = -1; // inhale = chest tips back / up
export const RAMP_SECONDS = 1.5; // ease in from rest so she doesn't pop on boot
export const MAX_ABS_OFFSET = 0.05; // radians, ~2.9 degrees; tested ceiling

export interface BoneOffset {
  x: number;
  y: number;
  z: number;
}

export const IDLE_BONE_NAMES = [
  "hips",
  "spine",
  "chest",
  "upperChest",
  "neck",
  "head",
  "leftShoulder",
  "rightShoulder",
] as const;

export type IdleBoneName = (typeof IDLE_BONE_NAMES)[number];
export type IdleOffsets = Record<IdleBoneName, BoneOffset>;

const TAU = Math.PI * 2;

function smoothstep01(x: number): number {
  const c = Math.min(1, Math.max(0, x));
  return c * c * (3 - 2 * c);
}

/** A smooth, non-repeating-looking wobble in [-1, 1]: three sines at
 * unrelated frequencies. Deterministic, so it is testable, and there is no
 * per-frame randomness to read as twitchiness. */
function drift(t: number, hz: [number, number, number], phase: [number, number, number]): number {
  return (
    0.5 * Math.sin(TAU * hz[0] * t + phase[0]) +
    0.3 * Math.sin(TAU * hz[1] * t + phase[1]) +
    0.2 * Math.sin(TAU * hz[2] * t + phase[2])
  );
}

function zero(): BoneOffset {
  return { x: 0, y: 0, z: 0 };
}

/**
 * Where every idle bone should be `timeSeconds` after boot. `intensity` is
 * a 0..1 master scale (0 = perfectly still).
 */
export function idleOffsets(timeSeconds: number, intensity = 1): IdleOffsets {
  const out = {} as IdleOffsets;
  for (const name of IDLE_BONE_NAMES) out[name] = zero();

  const k = Math.max(0, Math.min(1, intensity)) * smoothstep01(timeSeconds / RAMP_SECONDS);
  if (k === 0) return out;
  const t = timeSeconds;

  // Breathing: a slow sine with a gentle depth variation so no two breaths
  // are identical. Chest and upper chest tip, the spine barely, and the
  // shoulders lift a hair on the inhale (left/right are mirrored: lowering
  // the left arm is -Z, so raising the left shoulder is +Z).
  const depth = 1 + 0.2 * Math.sin(TAU * 0.037 * t + 1.3);
  const breath = Math.sin(TAU * BREATH_HZ * t) * depth;
  const chestPitch = BREATH_SIGN * 0.014 * breath * k;
  const upperChestPitch = BREATH_SIGN * 0.01 * breath * k;
  const spinePitch = BREATH_SIGN * 0.005 * breath * k;
  out.chest.x = chestPitch;
  out.upperChest.x = upperChestPitch;
  out.spine.x = spinePitch;
  out.leftShoulder.z = 0.012 * breath * k;
  out.rightShoulder.z = -0.012 * breath * k;

  // Weight shift: hips lean slowly side to side, spine and chest counter
  // so the head stays roughly over her feet. Two long periods that don't
  // divide each other, so the pattern doesn't loop audibly.
  const sway = 0.65 * Math.sin((TAU * t) / 9.3 + 0.7) + 0.35 * Math.sin((TAU * t) / 15.1 + 2.1);
  const turn = Math.sin((TAU * t) / 15.1 + 2.1);
  out.hips.z = 0.016 * sway * k;
  out.hips.y = 0.01 * turn * k;
  out.spine.z = -0.01 * sway * k;
  out.chest.z = -0.006 * sway * k;

  // Head drift: small, slow, three axes on unrelated frequencies. The neck
  // takes 40% of it and the head 60%. Breathing pitch is partly cancelled
  // here so her gaze doesn't bob with every breath.
  const nx = drift(t, [0.11, 0.17, 0.29], [0.4, 1.9, 3.1]) * 0.02;
  const ny = drift(t, [0.07, 0.13, 0.23], [2.2, 0.6, 4.0]) * 0.035;
  const nz = drift(t, [0.09, 0.19, 0.31], [5.1, 2.7, 1.1]) * 0.012;
  const comp = -0.3 * (chestPitch + upperChestPitch + spinePitch);
  out.neck.x = 0.4 * nx * k + comp;
  out.neck.y = 0.4 * ny * k;
  out.neck.z = 0.4 * nz * k;
  out.head.x = 0.6 * nx * k + comp;
  out.head.y = 0.6 * ny * k;
  out.head.z = 0.6 * nz * k;

  return out;
}

/** Anything with a three.js-style `rotation.set(x, y, z)` -- a normalized
 * VRM bone node, or a test double. Null/undefined for a bone the model
 * lacks, which is skipped. */
export type RotatableBone = { rotation: { set(x: number, y: number, z: number): unknown } } | null | undefined;

export function applyIdleOffsets(
  bones: Partial<Record<IdleBoneName, RotatableBone>>,
  offsets: IdleOffsets,
): void {
  for (const name of IDLE_BONE_NAMES) {
    const bone = bones[name];
    if (!bone) continue;
    const o = offsets[name];
    bone.rotation.set(o.x, o.y, o.z);
  }
}
