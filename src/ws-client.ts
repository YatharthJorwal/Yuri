// Talks to the Python orchestrator (orchestrator/app.py) over a small JSON
// protocol. Phase 1 only needed two message shapes; Phase 2.5 adds voice
// input (user_audio / transcript); a later round adds a "stop" button;
// Phase 8 adds `emotion` on `turn_end`; the full-body sandbox round adds a
// `surface` identifier on connect and a `surface_status` reply, so two
// windows (the desktop shell and the sandbox) talking to the same
// orchestrator at once don't both try to drive the same conversation --
// see that message's own comment below and docs/DECISIONS.md:
//
//   -> { type: "user_text", text: string }
//   -> { type: "user_audio", audio_b64: string }
//   -> { type: "stop" }
//   -> { type: "get_log" }
//   -> { type: "clear_log" }
//   -> { type: "set_temp_mode", enabled: boolean }
//   -> { type: "set_ocr_watch", enabled: boolean }
//   -> { type: "camera_frame", image_b64: string | null }
//   -> { type: "user_file", filename: string, kind: "image" | "text", content: string, caption?: string | null }
//   <- { type: "speak", text: string, audio_b64: string, mime: string }
//   <- { type: "transcript", text: string }
//   <- { type: "turn_end", emotion?: string }
//   <- { type: "surface_status", role: "driver" | "observer" }
//   <- { type: "log", turns: { user: string, assistant: string, ts: string }[], user_name: string }
//   <- { type: "request_camera_frame" }
//
// `request_camera_frame` (Phase 5's capture_camera tool, see
// orchestrator/camera.py) is the one message the orchestrator sends
// specifically expecting an answer back over this same connection rather
// than just broadcasting state -- the frontend replies with `camera_frame`
// as soon as it can (a captured frame's base64, or null if the camera
// isn't armed/capture failed), and the orchestrator's camera.py is the one
// that actually waits for it (bounded by a timeout there, not anything
// enforced here).
//
// `get_log`/`clear_log` (Phase 9's persistent conversation-log panel) are
// answered with the same `log` message either way -- `clear_log` just
// empties the underlying table first, so the response naturally comes
// back with `turns: []` rather than needing a separate ack shape. Unlike
// `user_text`/`user_audio`, these aren't gated by driver/observer status
// (see below) -- the log is one shared, persistent record, not something
// tied to a single connection's in-memory `history`, so either window can
// read or clear it freely. Server-side: orchestrator/memory/store.py's
// `transcript_log` table, written once per turn from `app.py`'s
// `_run_turn` (see its own comment for exactly what gets logged and what
// doesn't).
//
// `transcript` is sent once per `user_audio` message, always -- an empty
// `text` means the orchestrator heard nothing intelligible (silence, noise),
// which is distinct from a connection problem. `turn_end` is sent once a
// reply is fully done generating, whether it finished normally or was cut
// short by a `stop` -- the frontend uses it to know when to re-enable input
// and hide the stop button, since queued audio finishing playback isn't the
// same signal (more chunks could still be on the way when audio catches up).
// `emotion`, when present, is one of the six VRM expression names main.ts's
// emotion-blend system knows about (see orchestrator/persona.py's
// VALID_EMOTIONS) -- omitted (not a guessed default) whenever the LLM's own
// reply didn't produce a recognizable tag, or the turn never got that far
// (LLM unreachable, stopped mid-generation) -- the frontend leaves whatever
// expression she's already wearing alone in that case, rather than snapping
// to something arbitrary. Delivered once per whole turn, not per `speak`
// chunk -- an early design considered tagging every sentence individually,
// but that's a much heavier compliance ask for a small model for very
// little benefit, since the blend itself is what makes expression changes
// look gradual, not the tagging granularity.
//
// `surface_status`: whichever window connects first becomes the "driver"
// (the one actually allowed to start turns); a second window connecting
// while the first is still open becomes an "observer" -- sent a
// `surface_status` with role "observer" right after connecting, and gets a
// canned in-character decline (not silence) if it tries to send
// `user_text`/`user_audio` anyway, rather than either window's TTS audio
// silently overlapping the other's. If the driver disconnects, the
// orchestrator promotes the next still-open connection (if any) and sends
// it a fresh `surface_status` with role "driver" -- see app.py's
// `ws_endpoint` for the actual promotion logic. Each connection still has
// its own conversation history (unchanged since Phase 2), so a promoted
// observer starts a fresh conversation rather than inheriting the old
// driver's mid-conversation state -- a known, accepted limitation, not an
// oversight; see docs/DECISIONS.md.

const ORCHESTRATOR_URL = "ws://127.0.0.1:8765/ws";
const RECONNECT_DELAY_MS = 2000;

export type SpeakMessage = {
  type: "speak";
  text: string;
  audio_b64: string;
  mime: string;
};

export type TranscriptMessage = {
  type: "transcript";
  text: string;
};

export type TurnEndMessage = {
  type: "turn_end";
  emotion?: string;
};

export type LogTurn = {
  user: string;
  assistant: string;
  ts: string;
};

export type LogMessage = {
  type: "log";
  turns: LogTurn[];
  /** From config.yaml's session.user_name -- how the frontend labels the
   * user's own lines ("Yatharth: hello"). Falls back to "You" client-side
   * if this is ever missing (shouldn't happen, but the config value is
   * user-edited free text, not a validated enum). */
  user_name: string;
};

export type SurfaceRole = "driver" | "observer";

export type SurfaceStatusMessage = {
  type: "surface_status";
  role: SurfaceRole;
};

export type RequestCameraFrameMessage = {
  type: "request_camera_frame";
};

export type ConnectionState = "offline" | "connected" | "listening";

interface WsClientOptions {
  /** Identifies which window this connection is from -- "shell" for the
   * Tauri desktop pet (src/main.ts), "sandbox" for the full-body sandbox
   * (src/sandbox.ts). Purely informational on the wire (a query param the
   * orchestrator can log/key off of); the driver/observer decision itself
   * is "whoever connected first," not based on which surface name this is
   * -- see this file's own top-of-file comment. */
  surface: "shell" | "sandbox";
  onSpeak: (msg: SpeakMessage) => void;
  onTranscript: (msg: TranscriptMessage) => void;
  onTurnEnd: (emotion?: string) => void;
  onStateChange: (state: ConnectionState) => void;
  /** Fired for both get_log's response and clear_log's (the latter always
   * arrives with an empty `turns` array) -- see this file's own protocol
   * comment. Optional for the same reason onSurfaceStatus is: a caller
   * that never sends get_log/clear_log (there isn't one today besides
   * main.ts's log panel) doesn't need to handle it. */
  onLog?: (msg: LogMessage) => void;
  /** Optional since it's new -- a caller that doesn't care about
   * driver/observer status (there isn't one today, but keeping this
   * optional rather than required avoids forcing every future caller to
   * handle a concern that may not apply to it) can simply omit it. */
  onSurfaceStatus?: (role: SurfaceRole) => void;
  /** Phase 5's capture_camera tool asking this connection for a frame --
   * see this file's own protocol comment on request_camera_frame/
   * camera_frame. The caller is expected to answer with sendCameraFrame
   * (below) as soon as it reasonably can; there's no enforcement here if
   * it doesn't, camera.py's own timeout is what actually bounds the
   * wait. Optional: a caller with no camera module (sandbox.ts, at least
   * for now) can simply omit it and camera.py's request will time out
   * and surface as "camera isn't armed" to the model, same as if it
   * genuinely weren't armed. */
  onRequestCameraFrame?: () => void;
  /** Fired every time the socket opens, first connect and every reconnect.
   * The shell uses it to resend per-connection UI state (Temp Chat) and the
   * Continuous OCR toggle: sendSetTempMode/sendSetOcrWatch silently drop
   * their message when the socket isn't OPEN, so a toggle clicked while the
   * orchestrator was still starting (or reconnecting) showed "On" in the
   * menu while the backend never heard about it. */
  onOpen?: () => void;
}

export class WsClient {
  private socket: WebSocket | null = null;
  private opts: WsClientOptions;
  private reconnectTimer: number | undefined;

  constructor(opts: WsClientOptions) {
    this.opts = opts;
  }

  connect(): void {
    const url = `${ORCHESTRATOR_URL}?surface=${encodeURIComponent(this.opts.surface)}`;
    this.socket = new WebSocket(url);

    this.socket.addEventListener("open", () => {
      this.opts.onStateChange("connected");
      this.opts.onOpen?.();
    });

    this.socket.addEventListener("message", (event) => {
      let parsed:
        | SpeakMessage
        | TranscriptMessage
        | TurnEndMessage
        | SurfaceStatusMessage
        | LogMessage
        | RequestCameraFrameMessage;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        console.warn("[luna] ignoring malformed message from orchestrator");
        return;
      }
      if (parsed.type === "speak") {
        this.opts.onStateChange("connected");
        this.opts.onSpeak(parsed);
      } else if (parsed.type === "transcript") {
        this.opts.onTranscript(parsed);
      } else if (parsed.type === "turn_end") {
        this.opts.onTurnEnd(parsed.emotion);
      } else if (parsed.type === "surface_status") {
        this.opts.onSurfaceStatus?.(parsed.role);
      } else if (parsed.type === "log") {
        this.opts.onLog?.(parsed);
      } else if (parsed.type === "request_camera_frame") {
        this.opts.onRequestCameraFrame?.();
      }
    });

    this.socket.addEventListener("close", () => {
      this.opts.onStateChange("offline");
      this.scheduleReconnect();
    });

    this.socket.addEventListener("error", () => {
      this.socket?.close();
    });
  }

  sendUserText(text: string): void {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      console.warn("[luna] not connected to orchestrator yet");
      return;
    }
    this.opts.onStateChange("listening");
    this.socket.send(JSON.stringify({ type: "user_text", text }));
  }

  sendUserAudio(audioB64: string): void {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      console.warn("[luna] not connected to orchestrator yet");
      return;
    }
    this.opts.onStateChange("listening");
    this.socket.send(JSON.stringify({ type: "user_audio", audio_b64: audioB64 }));
  }

  /** Quick-action menu's Upload Image/File item. `kind` tells the
   * orchestrator how to treat `content` -- "image" means base64 (no
   * data: prefix, same convention every other image payload in this
   * project uses), described through the vision model before she reacts
   * to it; "text" means content is the file's own text, read directly,
   * no vision step needed. `caption`, when given, is whatever the user
   * typed in the normal input box alongside the attachment (see
   * main.ts's submitTurn) -- optional, since sending a file with no
   * comment is a real, common case, not a degraded one. Starts a real
   * turn the same way sendUserText does (onStateChange("listening")) --
   * from the turn-handling side, an upload is just another way the user
   * said something, not a separate mechanism. */
  sendUserFile(filename: string, kind: "image" | "text", content: string, caption?: string): void {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      console.warn("[luna] not connected to orchestrator yet");
      return;
    }
    this.opts.onStateChange("listening");
    this.socket.send(JSON.stringify({ type: "user_file", filename, kind, content, caption: caption || null }));
  }

  /**
   * Push ambient context about where Luna is and what's around her.
   *
   * Deliberately fire-and-forget and deliberately *not* a turn: it never
   * changes connection state, never prompts a reply, and is silently
   * dropped if the socket isn't up. The orchestrator is free to ignore the
   * message type entirely -- an older `app.py` that doesn't know about it
   * will just not match it, which is why this doesn't wait for an ack.
   * See docs/DECISIONS.md on the scene-state channel.
   */
  sendSceneState(text: string): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ type: "scene_state", text }));
  }

  /** Tells the orchestrator to cancel whatever's currently generating (if
   * anything -- a no-op on its side if nothing is). Client-side audio
   * playback is stopped separately and immediately by main.ts's
   * SpeakQueue.stopAll(), since waiting on a round-trip to the
   * orchestrator and back would make the stop button feel laggy for the
   * part that actually matters visually/audibly. */
  sendStop(): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ type: "stop" }));
  }

  /** Requests the full persistent conversation log -- answered with a
   * `log` message via onLog. Not gated by driver/observer status (see
   * this file's protocol comment), so this is safe to call even from an
   * observer window. */
  sendGetLog(): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ type: "get_log" }));
  }

  /** Wipes the persistent conversation log server-side. Also answered
   * with a `log` message (turns: []) via onLog, so callers can just
   * re-render off the same handler rather than assuming success. */
  sendClearLog(): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ type: "clear_log" }));
  }

  /** Quick-action menu's Temp Chat toggle -- while enabled, the
   * orchestrator skips reading or writing any persistent memory for this
   * connection's turns (see app.py's temp_mode/temp_turn_flags). Fire-
   * and-forget like sendSceneState -- there's no meaningful failure mode
   * to react to, and the frontend already reflects its own toggle state
   * optimistically the instant the user clicks it. Known gap, not
   * handled here: temp_mode lives server-side per *connection*, so a
   * reconnect (this class's own scheduleReconnect below) starts a fresh
   * connection with temp_mode back at false, even if the UI still shows
   * it on. Fixed via the onOpen option: the shell resends its Temp Chat
   * state every time the socket opens. */
  sendSetTempMode(enabled: boolean): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ type: "set_temp_mode", enabled }));
  }

  /** Quick-action menu's Continuous OCR toggle -- ambient screen-watching
   * (see orchestrator/ocr_watch.py), independent of Task Guide Mode.
   * Fire-and-forget, same reasoning as sendSetTempMode above. */
  sendSetOcrWatch(enabled: boolean): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ type: "set_ocr_watch", enabled }));
  }

  /** Answers a request_camera_frame (see onRequestCameraFrame in
   * WsClientOptions above) -- image_b64 is null when the camera isn't
   * armed, permission was denied, or the capture otherwise failed;
   * camera.py on the other end turns that into the same "camera isn't
   * armed" message the model sees either way, so the caller doesn't need
   * to distinguish those cases before calling this. Fire-and-forget, same
   * as every other send* method here -- if the socket isn't open, the
   * request will simply time out orchestrator-side rather than error
   * here. */
  sendCameraFrame(image_b64: string | null): void {
    if (this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify({ type: "camera_frame", image_b64 }));
  }

  private scheduleReconnect(): void {
    window.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = window.setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
  }
}
