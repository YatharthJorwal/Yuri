/**
 * Where the camera should look so the whole model fits, ears and all.
 *
 * The camera used to look at the head bone, which sits at the base of the head, so
 * anything tall above it (bunny ears, a hat, a ponytail) was cropped by the top of
 * the frame. Instead: look at the head bone unless the model's highest point would
 * fall outside the frame, and in that case raise the look-at point just enough to
 * leave `headroom` metres above the tallest point. `maxRise` stops a freak mesh (a
 * halo, a mis-scaled prop) from dragging the face out of the bottom of the frame.
 *
 * Pure maths, no three.js, so it is covered by tests-fe/framing.test.mjs.
 */

export interface Framing {
  /** Camera distance from the look-at point, in metres. */
  distance: number;
  /** Vertical field of view, in degrees. */
  fovDegrees: number;
  /** Clear space to leave above the model's tallest point, in metres. */
  headroom: number;
  /** The most the look-at point may be raised above the head bone, in metres. */
  maxRise: number;
}

export function lookAtHeight(headY: number, modelTopY: number, framing: Framing): number {
  if (!Number.isFinite(modelTopY) || !Number.isFinite(headY)) return headY;
  const halfVisible = framing.distance * Math.tan((framing.fovDegrees * Math.PI) / 360);
  const needed = modelTopY + framing.headroom - halfVisible;
  return Math.min(Math.max(headY, needed), headY + framing.maxRise);
}
