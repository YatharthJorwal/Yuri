/**
 * Two ways to be in the apartment.
 *
 * SPECTATOR -- the existing free-fly camera, unchanged in spirit: WASD flies,
 * right-drag looks, scroll changes speed, no collision, can leave the flat
 * and look in from outside. It stays the default because it's the right tool
 * for checking on the place.
 *
 * VISITOR -- first person, feet on the floor. Eye height, head bob, no
 * flying, and crucially *clamped to the same walkable rectangles Luna uses*,
 * so you're in the rooms with her rather than drifting through walls. This
 * is the "stuck in that plane" mode: the apartment reads completely
 * differently at 1.65m with a real ceiling above you than it does from a
 * drone shot, which is most of what makes the reference game feel like a
 * place rather than a diorama.
 *
 * Pointer lock is used for looking around in visitor mode, so you can turn
 * past the edge of the window. Esc releases it, and releasing it doesn't
 * leave the mode -- you just stop steering.
 */

import * as THREE from 'three';

export type CameraMode = 'spectator' | 'visitor';

/** What the walkable-area clamp needs to expose. Keeps this file decoupled
 * from whichever navmesh implementation sandbox.ts is using. */
export interface Clampable {
  clamp(x: number, z: number): { x: number; z: number };
  contains(x: number, z: number): boolean;
}

const EYE_HEIGHT = 1.5; // was 1.62 -- brought down a bit per request
const CROUCH_HEIGHT = 1.05;
const WALK_SPEED = 2.1;
const RUN_SPEED = 3.6;
const ACCEL = 14;
const FRICTION = 11;

export interface CameraRig {
  camera: THREE.PerspectiveCamera;
  mode(): CameraMode;
  setMode(m: CameraMode, opts?: { at?: THREE.Vector3; yaw?: number }): void;
  update(dt: number): void;
  /** Ground position of the visitor, or null when spectating. */
  visitorPosition(): THREE.Vector3 | null;
  /** How fast the visitor is moving, for footstep/bob purposes. */
  visitorSpeed(): number;
  dispose(): void;
}

export function createCameraRig(
  camera: THREE.PerspectiveCamera,
  dom: HTMLElement,
  walkable: Clampable,
  spawn: { x: number; z: number; yaw: number },
): CameraRig {
  let mode: CameraMode = 'spectator';

  // Shared look state, so switching modes doesn't snap your view around.
  let yaw = spawn.yaw;
  let pitch = -0.02;

  // Spectator state
  const flyPos = new THREE.Vector3(camera.position.x, camera.position.y, camera.position.z);
  let flySpeed = 2.6;

  // Visitor state
  const feet = new THREE.Vector3(spawn.x, 0, spawn.z);
  const vel = new THREE.Vector3();
  let bobPhase = 0;
  let crouch = 0;
  let speedNow = 0;

  const keys = new Set<string>();
  let looking = false;
  let locked = false;

  // --- input ---------------------------------------------------------------
  const onKeyDown = (e: KeyboardEvent): void => {
    const t = e.target as HTMLElement | null;
    // Don't steal WASD from the chat box.
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    keys.add(e.code);
    if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
  };
  const onKeyUp = (e: KeyboardEvent): void => {
    keys.delete(e.code);
  };
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  const onContext = (e: Event): void => e.preventDefault();
  dom.addEventListener('contextmenu', onContext);

  const onPointerDown = (e: PointerEvent): void => {
    if (mode === 'visitor') {
      if (e.button === 0 && !locked) {
        void dom.requestPointerLock?.();
      }
      return;
    }
    if (e.button === 2 || e.button === 1) {
      looking = true;
      dom.setPointerCapture(e.pointerId);
    }
  };
  const onPointerUp = (): void => {
    looking = false;
  };
  const onPointerMove = (e: PointerEvent): void => {
    const active = mode === 'visitor' ? locked : looking;
    if (!active) return;
    yaw -= e.movementX * 0.0022;
    pitch = THREE.MathUtils.clamp(pitch - e.movementY * 0.0020, -1.45, 1.45);
  };
  dom.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointerup', onPointerUp);
  window.addEventListener('pointercancel', onPointerUp);
  window.addEventListener('pointermove', onPointerMove);

  const onLockChange = (): void => {
    locked = document.pointerLockElement === dom;
  };
  document.addEventListener('pointerlockchange', onLockChange);

  const onWheel = (e: WheelEvent): void => {
    if (mode !== 'spectator') return;
    e.preventDefault();
    // Round 13 (feedback): "camera gained lightspeed in spectate mode."
    // This logic is untouched by round 13's own changes -- flySpeed has
    // always been scroll-adjustable, and scrolling while looking around is
    // an easy way to ratchet it toward the old ceiling without meaning to.
    // Still, 24 units/s crosses this building's real ~19m width in under a
    // second, which reads as "lightspeed" regardless of how it got there --
    // lowered the ceiling to something that stays fast without feeling
    // broken at this building's actual scale.
    flySpeed = THREE.MathUtils.clamp(flySpeed * (e.deltaY > 0 ? 0.88 : 1.14), 0.35, 14);
  };
  dom.addEventListener('wheel', onWheel, { passive: false });

  // --- helpers -------------------------------------------------------------
  function applyLook(): void {
    camera.rotation.set(0, 0, 0);
    camera.rotateY(yaw);
    camera.rotateX(pitch);
  }

  function wishDir(): THREE.Vector3 {
    // Forward/right in the ground plane from the current yaw. Kept flat in
    // visitor mode so looking at the ceiling doesn't launch you at it.
    const f = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
    const r = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    const d = new THREE.Vector3();
    if (keys.has('KeyW') || keys.has('ArrowUp')) d.add(f);
    if (keys.has('KeyS') || keys.has('ArrowDown')) d.sub(f);
    if (keys.has('KeyD') || keys.has('ArrowRight')) d.add(r);
    if (keys.has('KeyA') || keys.has('ArrowLeft')) d.sub(r);
    if (d.lengthSq() > 0) d.normalize();
    return d;
  }

  function updateSpectator(dt: number): void {
    const d = wishDir();
    // Spectator keeps full 6-dof: forward follows where you're looking.
    const look = new THREE.Vector3(
      -Math.sin(yaw) * Math.cos(pitch),
      Math.sin(pitch),
      -Math.cos(yaw) * Math.cos(pitch),
    );
    const move = new THREE.Vector3();
    if (keys.has('KeyW') || keys.has('ArrowUp')) move.add(look);
    if (keys.has('KeyS') || keys.has('ArrowDown')) move.sub(look);
    const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    if (keys.has('KeyD') || keys.has('ArrowRight')) move.add(right);
    if (keys.has('KeyA') || keys.has('ArrowLeft')) move.sub(right);
    if (keys.has('Space')) move.y += 1;
    if (keys.has('ShiftLeft') || keys.has('ShiftRight')) move.y -= 1;
    void d;
    if (move.lengthSq() > 0) {
      move.normalize().multiplyScalar(flySpeed * dt);
      flyPos.add(move);
    }
    camera.position.copy(flyPos);
    applyLook();
  }

  function updateVisitor(dt: number): void {
    const wish = wishDir();
    const running = keys.has('ShiftLeft') || keys.has('ShiftRight');
    const targetSpeed = running ? RUN_SPEED : WALK_SPEED;

    // Accelerate toward the wish direction, decelerate with friction.
    if (wish.lengthSq() > 0) {
      vel.x = THREE.MathUtils.lerp(vel.x, wish.x * targetSpeed, Math.min(1, ACCEL * dt));
      vel.z = THREE.MathUtils.lerp(vel.z, wish.z * targetSpeed, Math.min(1, ACCEL * dt));
    } else {
      const f = Math.max(0, 1 - FRICTION * dt);
      vel.x *= f;
      vel.z *= f;
    }

    const nx = feet.x + vel.x * dt;
    const nz = feet.z + vel.z * dt;
    // Same navmesh Luna uses. Sliding along a wall rather than stopping dead
    // is the difference between "a floor plan" and "a place you can move in",
    // so try each axis separately when the combined move is blocked.
    if (walkable.contains(nx, nz)) {
      feet.set(nx, 0, nz);
    } else if (walkable.contains(nx, feet.z)) {
      feet.x = nx;
      vel.z = 0;
    } else if (walkable.contains(feet.x, nz)) {
      feet.z = nz;
      vel.x = 0;
    } else {
      const safe = walkable.clamp(nx, nz);
      feet.set(safe.x, 0, safe.z);
      vel.set(0, 0, 0);
    }

    speedNow = Math.hypot(vel.x, vel.z);

    // Crouch on Ctrl / C -- useful for looking under things and for being at
    // her eye level when she's sitting.
    const wantCrouch = keys.has('ControlLeft') || keys.has('KeyC') ? 1 : 0;
    crouch = THREE.MathUtils.lerp(crouch, wantCrouch, Math.min(1, 9 * dt));
    const standH = THREE.MathUtils.lerp(EYE_HEIGHT, CROUCH_HEIGHT, crouch);

    // Head bob, scaled by how fast you're actually going.
    bobPhase += speedNow * dt * 5.4;
    const bobY = Math.sin(bobPhase * 2) * 0.022 * Math.min(1, speedNow / WALK_SPEED);
    const bobX = Math.cos(bobPhase) * 0.016 * Math.min(1, speedNow / WALK_SPEED);

    camera.position.set(
      feet.x + Math.cos(yaw) * bobX,
      standH + bobY,
      feet.z - Math.sin(yaw) * bobX,
    );
    applyLook();
    // A touch of roll while strafing sells the weight of a body.
    camera.rotateZ(-bobX * 0.35);
  }

  return {
    camera,
    mode: () => mode,

    setMode(m, opts) {
      if (m === mode) return;
      if (m === 'visitor') {
        const at = opts?.at;
        const start = at ?? new THREE.Vector3(spawn.x, 0, spawn.z);
        const safe = walkable.clamp(start.x, start.z);
        feet.set(safe.x, 0, safe.z);
        vel.set(0, 0, 0);
        bobPhase = 0;
        if (opts?.yaw !== undefined) yaw = opts.yaw;
        pitch = THREE.MathUtils.clamp(pitch, -0.9, 0.9);
      } else {
        // Step back out of the body: keep the current viewpoint so the switch
        // doesn't teleport you.
        flyPos.copy(camera.position);
        if (document.pointerLockElement === dom) document.exitPointerLock();
      }
      mode = m;
    },

    update(dt) {
      const d = Math.min(dt, 0.05);
      if (mode === 'spectator') updateSpectator(d);
      else updateVisitor(d);
    },

    visitorPosition: () => (mode === 'visitor' ? feet.clone() : null),
    visitorSpeed: () => (mode === 'visitor' ? speedNow : 0),

    dispose() {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
      window.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('pointerlockchange', onLockChange);
      dom.removeEventListener('pointerdown', onPointerDown);
      dom.removeEventListener('contextmenu', onContext);
      dom.removeEventListener('wheel', onWheel);
    },
  };
}
