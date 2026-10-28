// Phase 5 -- webcam capture for the capture_camera tool (see
// orchestrator/camera.py and tools/__init__.py). Same "plain Web APIs, no
// Tauri-specific plugin" spirit as mic.ts -- WebView2 supports
// getUserMedia() for video the same as it does for audio.
//
// Deliberately NOT continuous/streaming to the backend --
// docs/ARCHITECTURE.md's "Vision tools -- pull, not push" spec: the
// camera stream stays open locally once armed (so a capture can be
// grabbed instantly when asked, no per-request permission friction), but
// nothing is sent anywhere until the backend actually asks for a frame
// (request_camera_frame, see ws-client.ts). No live preview is ever shown
// to the user either, per that same spec ("even though nothing is
// displayed back to the user") -- the only visible sign the camera is
// armed is the tray icon (src-tauri/src/lib.rs's set_camera_indicator),
// not anything in this window.

import { invoke } from "@tauri-apps/api/core";

export interface CameraInputOptions {
  onArmedChange: (armed: boolean) => void;
  onError: (err: unknown) => void;
}

export class CameraInput {
  private opts: CameraInputOptions;
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement | null = null;

  constructor(opts: CameraInputOptions) {
    this.opts = opts;
  }

  get isArmed(): boolean {
    return this.stream !== null;
  }

  /** Requests camera access (triggers the browser's own one-time
   * permission prompt the first time this origin has asked) and keeps
   * the stream open until disarm() is called. No-op if already armed --
   * safe to call from a menu item without worrying whether it's already
   * been clicked once this session. */
  async arm(): Promise<void> {
    if (this.isArmed) return;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch (err) {
      // Permission denied, or no camera present -- nothing to recover
      // from here; same shape as mic.ts's own start() error handling.
      this.opts.onError(err);
      return;
    }

    // Hidden video element the stream actually plays into -- never
    // attached to the DOM or shown, per this module's own top comment on
    // why there's no preview. A canvas alongside it is what actually
    // grabs pixels out when captureFrame() is called.
    this.video = document.createElement("video");
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play();

    this.canvas = document.createElement("canvas");

    // If the OS/browser stops the stream out from under us (camera
    // unplugged, access revoked at the OS level), treat it the same as
    // an explicit disarm rather than leaving isArmed lying about the
    // real state.
    this.stream.getVideoTracks()[0]?.addEventListener("ended", () => {
      void this.disarm();
    });

    await this.setIndicator(true);
    this.opts.onArmedChange(true);
  }

  /** Stops the stream and releases the camera. No-op if already
   * disarmed. */
  async disarm(): Promise<void> {
    if (!this.isArmed) return;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video = null;
    this.canvas = null;
    await this.setIndicator(false);
    this.opts.onArmedChange(false);
  }

  /** Grabs the current frame as bare base64 JPEG (no `data:` prefix), or
   * null if not armed or the video isn't ready yet.
   *
   * Plain 2D canvas, not WebGL, on purpose: there's no pixel processing
   * happening here (no filters, no live preview being rendered) -- just
   * one still frame handed off as-is, and a 2D context does that exact
   * job with less setup than a WebGL one would, for identical speed. The
   * actual "fast" part of this pipeline (sub-millisecond to a few ms) is
   * the browser's own video-decode-to-canvas path either way; WebGL
   * would only earn its keep here if something needed real GPU-side
   * image processing, which grabbing a single still frame doesn't. */
  captureFrame(): string | null {
    if (!this.stream || !this.video || !this.canvas) return null;
    const { videoWidth, videoHeight } = this.video;
    if (!videoWidth || !videoHeight) return null;

    this.canvas.width = videoWidth;
    this.canvas.height = videoHeight;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(this.video, 0, 0, videoWidth, videoHeight);

    // toDataURL includes a "data:image/jpeg;base64," prefix -- strip it;
    // the orchestrator side (llm.describe_image) expects bare base64,
    // same convention capture_screen's own base64 PNG already uses.
    const dataUrl = this.canvas.toDataURL("image/jpeg", 0.85);
    const commaIndex = dataUrl.indexOf(",");
    return commaIndex === -1 ? null : dataUrl.slice(commaIndex + 1);
  }

  private async setIndicator(active: boolean): Promise<void> {
    try {
      await invoke("set_camera_indicator", { active });
    } catch (err) {
      // Non-fatal -- the actual camera arm/disarm above already
      // succeeded or failed on its own merits; the tray icon merely
      // failing to update shouldn't block anything or surface as a
      // camera error to the user.
      console.error("[luna] failed to update tray camera indicator", err);
    }
  }
}
