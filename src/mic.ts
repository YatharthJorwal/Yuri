// Captures short voice clips from the user's mic via the browser's
// MediaRecorder API and hands each finished clip back as a Blob, ready to
// base64-encode and send to the orchestrator as a `user_audio` message
// (see ws-client.ts). No Tauri-specific mic plugin needed -- WebView2 (the
// webview Tauri uses on Windows) supports getUserMedia()/MediaRecorder like
// any Chromium-based browser, so this is plain Web APIs throughout, same
// spirit as lipsync.ts using plain Web Audio instead of a Tauri audio API.

// Toggle-to-record (mouse) and push-to-talk (F9 hotkey, wired up in
// main.ts via a Tauri global-shortcut event -- see src-tauri/src/lib.rs)
// both drive the same start()/stop() pair below, so there's one recording
// state machine to get right, not two. Toggle is the click-friendly default
// for the mic button; push-to-talk exists because holding a key while
// talking to a small, draggable, always-on-top overlay is easier than
// aiming a click at it, especially with another window (a game) focused --
// F9 works globally regardless of which window has focus.
//
// The mic button itself is still click-to-start/click-to-stop rather than
// press-and-hold: a *mouse* press-and-hold on this tiny window risks losing
// the mouseup entirely if the cursor drifts off it before releasing, which
// would leave the mic stuck recording with no way to stop it from the UI.
// A physical key doesn't have that failure mode -- keyup fires wherever the
// cursor is -- so push-to-talk is safe for the hotkey even though it isn't
// for the mouse.
const MAX_RECORDING_MS = 30_000; // safety net if the user forgets to click stop

// Chromium (WebView2's engine) supports opus-in-webm; fall back gracefully
// in case a future webview swap changes that. The orchestrator doesn't need
// to be told which of these it got -- faster-whisper decodes via PyAV,
// which sniffs the container/codec from the bytes themselves.
function pickMimeType(): string {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
  for (const type of candidates) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return ""; // let the browser pick its own default
}

export interface MicInputOptions {
  onRecordingChange: (recording: boolean) => void;
  onClip: (blob: Blob) => void;
  onError: (err: unknown) => void;
}

export class MicInput {
  private opts: MicInputOptions;
  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: BlobPart[] = [];
  private activeMimeType = "";
  private maxDurationTimer: number | undefined;

  constructor(opts: MicInputOptions) {
    this.opts = opts;
  }

  get isRecording(): boolean {
    return this.recorder?.state === "recording";
  }

  /** Starts recording if idle; no-op if already recording -- safe to call
   * from both the mic button and the F9 hotkey without them fighting each
   * other, and safe against a stray duplicate "pressed" event. */
  async start(): Promise<void> {
    if (this.isRecording) return;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      // Most likely mic permission denied, or no input device present --
      // nothing to recover from here; the user can grant permission (or
      // plug in a mic) and try again.
      this.opts.onError(err);
      return;
    }

    this.activeMimeType = pickMimeType();
    this.recorder = this.activeMimeType
      ? new MediaRecorder(this.stream, { mimeType: this.activeMimeType })
      : new MediaRecorder(this.stream);
    this.chunks = [];

    this.recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    });
    // Both stop() below and the max-duration safety net go through
    // recorder.stop() -> this "stop" event, not a direct call --
    // MediaRecorder can still have a last "dataavailable" pending right up
    // to that event, so reading this.chunks any earlier could miss the
    // tail of the clip.
    this.recorder.addEventListener("stop", () => this.handleStopped());

    this.recorder.start();
    this.opts.onRecordingChange(true);
    this.maxDurationTimer = window.setTimeout(() => this.stop(), MAX_RECORDING_MS);
  }

  /** Stops recording if active; no-op if already idle -- same reasoning
   * as start() above. */
  stop(): void {
    window.clearTimeout(this.maxDurationTimer);
    if (this.recorder && this.recorder.state !== "inactive") {
      this.recorder.stop();
    }
  }

  /** Click-to-start/click-to-stop for the mouse: mic button. */
  async toggle(): Promise<void> {
    if (this.isRecording) {
      this.stop();
    } else {
      await this.start();
    }
  }

  private handleStopped(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.recorder = null;
    this.opts.onRecordingChange(false);

    if (this.chunks.length) {
      const blob = new Blob(this.chunks, { type: this.activeMimeType || "audio/webm" });
      this.chunks = [];
      this.opts.onClip(blob);
    }
  }
}

export async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]!);
  return btoa(binary);
}
