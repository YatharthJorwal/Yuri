// Quick-action menu's Upload Image/File item. Scoped deliberately narrow
// this round: images (any image/* MIME type) go through the same vision
// pipeline capture_screen/capture_camera already use, and plain text/code
// files get their content read directly -- no vision step needed for
// those. PDFs, Word docs, and video (the "mp4" half of the original
// "image/file/mp4" ask) are explicitly NOT handled here; each needs its
// own real parsing/frame-extraction work this round didn't attempt. See
// docs/DECISIONS.md for the full scoping reasoning.

/** Hard ceiling before even trying to read a file -- rejects absurd
 * uploads (a multi-hundred-MB video someone picked by mistake, say)
 * before spending any time on them. Well above what any real image or
 * text file used with this feature should need. */
const MAX_FILE_BYTES = 15 * 1024 * 1024;

/** Longest side an uploaded image gets resized to before encoding, via
 * canvas (see resizeImageToBase64Jpeg below) -- matches what a vision
 * model actually needs; sending a full 4000px phone photo would just be
 * wasted bandwidth and processing time for no extra understanding. */
const MAX_IMAGE_DIMENSION = 1024;

/** Text files get truncated to this many characters (not tokens -- a
 * rough but simple proxy) before being sent, so one large file can't
 * blow out the model's context window on its own. Truncation is called
 * out explicitly in what gets sent (see readTextFile below) rather than
 * silently cutting content off. */
const MAX_TEXT_CHARS = 6000;

/** Extensions treated as text even when the browser reports a generic or
 * empty MIME type for them (common for code files) -- checked in
 * addition to, not instead of, a `text/*` MIME type. Not exhaustive;
 * covers the languages/formats likely to actually come up here. */
const TEXT_EXTENSIONS = new Set([
  "txt",
  "md",
  "markdown",
  "csv",
  "log",
  "json",
  "yaml",
  "yml",
  "toml",
  "ini",
  "xml",
  "html",
  "css",
  "js",
  "ts",
  "jsx",
  "tsx",
  "py",
  "rs",
  "go",
  "java",
  "c",
  "cpp",
  "h",
  "hpp",
  "cs",
  "rb",
  "php",
  "sh",
  "sql",
]);

export type ClassifiedFile = { kind: "image" | "text" } | { kind: "unsupported"; reason: string };

/** Decides how (or whether) a picked file can be handled at all --
 * before any reading/resizing work happens, so an unsupported pick (a
 * PDF, a video, a spreadsheet) fails fast with a clear reason instead of
 * partway through trying to read it as text and producing garbage. */
export function classifyFile(file: File): ClassifiedFile {
  if (file.size > MAX_FILE_BYTES) {
    return { kind: "unsupported", reason: `File's too big (over ${MAX_FILE_BYTES / 1024 / 1024}MB).` };
  }
  if (file.type.startsWith("image/")) {
    return { kind: "image" };
  }
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (file.type.startsWith("text/") || TEXT_EXTENSIONS.has(ext)) {
    return { kind: "text" };
  }
  return {
    kind: "unsupported",
    reason: "That file type isn't supported yet -- images and text/code files only for now.",
  };
}

/** Draws the image onto an off-screen canvas (downscaled if needed) and
 * returns bare base64 JPEG -- no `data:` prefix, same convention
 * camera.ts's captureFrame() and every other image payload in this
 * project already use. Plain 2D canvas, same reasoning as camera.ts's
 * own choice: this is one still image with no filters or live rendering
 * involved, so WebGL wouldn't add anything here either. */
export async function resizeImageToBase64Jpeg(file: File): Promise<string> {
  const objectUrl = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("Couldn't decode that image."));
      el.src = objectUrl;
    });

    const scale = Math.min(1, MAX_IMAGE_DIMENSION / Math.max(img.naturalWidth, img.naturalHeight));
    const width = Math.max(1, Math.round(img.naturalWidth * scale));
    const height = Math.max(1, Math.round(img.naturalHeight * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable.");
    ctx.drawImage(img, 0, 0, width, height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    const commaIndex = dataUrl.indexOf(",");
    if (commaIndex === -1) throw new Error("Failed to encode image.");
    return dataUrl.slice(commaIndex + 1);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

/** Reads a text/code file's content directly, truncated to
 * MAX_TEXT_CHARS with an explicit marker if it was cut -- the model
 * should know content is incomplete rather than silently reason about a
 * partial file as if it were the whole thing. */
export async function readTextFile(file: File): Promise<string> {
  const text = await file.text();
  if (text.length <= MAX_TEXT_CHARS) return text;
  return `${text.slice(0, MAX_TEXT_CHARS)}\n\n[... truncated, file continues beyond this point ...]`;
}
