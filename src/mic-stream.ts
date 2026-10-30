// Streaming microphone: raw 20 ms PCM16 frames at 16 kHz, straight to the
// orchestrator's VAD (orchestrator/vad.py). Replaces the old MediaRecorder
// whole-clip path, which could only react after the user released the mic.
//
// Echo cancellation is requested here, in the browser, because that is where a
// usable AEC implementation actually lives. Barge-in depends on her own voice
// not coming back through the mic as "the user started talking".

export interface MicStreamOptions {
  onFrame: (pcmB64: string) => void;
  onActiveChange: (active: boolean) => void;
  onError: (err: unknown) => void;
}

const TARGET_RATE = 16000;
const FRAME_SAMPLES = 320; // 20 ms at 16 kHz

export class MicStream {
  private opts: MicStreamOptions;
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private sink: GainNode | null = null;
  private pending: number[] = [];
  private active = false;

  constructor(opts: MicStreamOptions) {
    this.opts = opts;
  }

  get isActive(): boolean {
    return this.active;
  }

  async start(): Promise<void> {
    if (this.active) return;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
    } catch (err) {
      this.opts.onError(err);
      return;
    }

    this.ctx = new AudioContext({ sampleRate: TARGET_RATE });
    await this.ctx.resume().catch(() => undefined);

    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.processor = this.ctx.createScriptProcessor(1024, 1, 1);
    // A zero gain node is how a ScriptProcessor gets driven without playing the
    // mic back through the speakers (which would also feed her own TTS).
    this.sink = this.ctx.createGain();
    this.sink.gain.value = 0;
    this.processor.onaudioprocess = (event) => this.handleAudio(event.inputBuffer);
    this.source.connect(this.processor);
    this.processor.connect(this.sink);
    this.sink.connect(this.ctx.destination);

    this.pending = [];
    this.active = true;
    this.opts.onActiveChange(true);
  }

  async stop(): Promise<void> {
    if (!this.active) return;
    this.active = false;
    this.processor?.disconnect();
    this.sink?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    const ctx = this.ctx;
    this.ctx = null;
    this.processor = null;
    this.sink = null;
    this.source = null;
    this.stream = null;
    this.pending = [];
    if (ctx) await ctx.close().catch(() => undefined);
    this.opts.onActiveChange(false);
  }

  private handleAudio(buffer: AudioBuffer): void {
    const data = buffer.getChannelData(0);
    const ratio = (this.ctx?.sampleRate ?? TARGET_RATE) / TARGET_RATE;
    if (Math.abs(ratio - 1) < 0.01) {
      for (let i = 0; i < data.length; i++) this.pending.push(data[i]!);
    } else {
      const outLength = Math.floor(data.length / ratio);
      for (let i = 0; i < outLength; i++) {
        this.pending.push(data[Math.floor(i * ratio)] ?? 0);
      }
    }

    while (this.pending.length >= FRAME_SAMPLES) {
      const chunk = this.pending.splice(0, FRAME_SAMPLES);
      const pcm = new Int16Array(FRAME_SAMPLES);
      for (let i = 0; i < FRAME_SAMPLES; i++) {
        const sample = Math.max(-1, Math.min(1, chunk[i] ?? 0));
        pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      this.opts.onFrame(bytesToBase64(new Uint8Array(pcm.buffer)));
    }
  }
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}
