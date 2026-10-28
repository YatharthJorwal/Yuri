/**
 * The post chain.
 *
 * An honest note on "ray tracing", because it was asked for by name: real
 * ray/path tracing at interactive rates isn't available here. This is WebGL2
 * through three.js, and the machine also has to run a VRM with spring bones,
 * an LLM bridge and a TTS pipeline at the same time. What *is* available is
 * the set of screen-space approximations that the offline-render look mostly
 * comes from, and those are what this file assembles:
 *
 *   - GTAO (ground-truth ambient occlusion) -- contact darkening in corners,
 *     under furniture, where a wall meets a floor. This is the single effect
 *     that most reads as "raytraced" to people, because unoccluded ambient
 *     light is the most obviously CG thing about a naive render.
 *   - Bloom on the emissive lamps and screens, driven by the time of day, so
 *     night reads as lamps glowing rather than lamps being pale shapes.
 *   - ACES Filmic tone mapping with per-mode exposure (set on the renderer,
 *     not here) so highlights roll off instead of clipping to white.
 *   - SMAA, because the composer bypasses the renderer's MSAA.
 *
 * Everything is behind a quality switch. GTAO in particular is the expensive
 * pass, and on a 3060 sharing the card with the rest of the stack it's worth
 * being able to drop it without losing bloom and tone mapping.
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/examples/jsm/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

export type Quality = 'high' | 'medium' | 'low';
export const QUALITIES: Quality[] = ['high', 'medium', 'low'];

export interface PostFX {
  render(dt: number): void;
  setSize(w: number, h: number): void;
  setQuality(q: Quality): void;
  quality(): Quality;
  /** Follow the apartment's current mode. */
  setBloom(strength: number): void;
  dispose(): void;
}

export function createPostFX(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  initial: Quality = 'high',
): PostFX {
  let quality: Quality = initial;
  let w = window.innerWidth;
  let h = window.innerHeight;

  const composer = new EffectComposer(renderer);
  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);

  const gtao = new GTAOPass(scene, camera, w, h);
  // Tuned for a domestic interior in metres: the interesting occlusion is at
  // skirting/table-leg scale, so the sample radius is small. A large radius
  // here is what makes GTAO look like a dirty smudge instead of contact shade.
  gtao.output = GTAOPass.OUTPUT.Default;
  const gtaoParams = {
    radius: 0.28,
    distanceExponent: 1.2,
    thickness: 0.4,
    scale: 1.0,
    samples: 16,
    distanceFallOff: 1.0,
    screenSpaceRadius: false,
  };
  gtao.updateGtaoMaterial(gtaoParams);
  composer.addPass(gtao);

  // Round 13: radius was 0.7, which on this apartment's small bright props
  // (light fixtures, screens) was producing wide, ringed halos rather than
  // a tight glow -- classic UnrealBloomPass over-blur on a small very-bright
  // source. Narrowed to 0.35 and threshold nudged up (0.92 -> 0.96) so only
  // genuinely blown-out pixels bloom, not merely bright ones. Per-mode
  // strength (setBloom, below) still does the day/night intensity work.
  const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.3, 0.35, 0.96);
  composer.addPass(bloom);

  const smaa = new SMAAPass();
  composer.addPass(smaa);

  const output = new OutputPass();
  composer.addPass(output);

  function applyQuality(): void {
    gtao.enabled = quality === 'high';
    smaa.enabled = quality !== 'low';
    bloom.enabled = true;
    const scale = quality === 'low' ? 0.75 : 1;
    composer.setPixelRatio(Math.min(window.devicePixelRatio, quality === 'high' ? 2 : 1.25) * scale);
    if (quality === 'high') {
      gtao.updateGtaoMaterial({ ...gtaoParams, samples: 16 });
    } else if (quality === 'medium') {
      gtao.updateGtaoMaterial({ ...gtaoParams, samples: 8 });
    }
  }
  applyQuality();

  return {
    render() {
      composer.render();
    },
    setSize(nw, nh) {
      w = nw;
      h = nh;
      composer.setSize(nw, nh);
      gtao.setSize(nw, nh);
      bloom.setSize(nw, nh);
    },
    setQuality(q) {
      quality = q;
      applyQuality();
    },
    quality: () => quality,
    setBloom(strength) {
      bloom.strength = strength;
    },
    dispose() {
      composer.dispose();
      gtao.dispose();
      bloom.dispose();
    },
  };
}
