// `npm run dev`: the page AND the backend, so a plain browser tab works end to end.
//
//   npm run dev        backend (orchestrator/app.py) + vite on http://localhost:1420
//   npm run dev:ui     vite only. This is what `npm run tauri dev` runs: the Tauri shell starts
//                      the backend itself, and a second copy would fight it for port 8765.
//
// If a backend is already listening on 8765 it is reused, not duplicated. Ctrl+C asks the backend
// to shut down gracefully (it finishes the session-end memory pass), then force-kills it if it
// takes more than a few seconds. Backend output is also written to logs/orchestrator.log.
import { spawn, spawnSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { portOpen } from "./doctor-lib.mjs";
import { backendPlan, lineSplitter, venvPython } from "./dev-lib.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const BACKEND_PORT = 8765;
const GRACEFUL_WAIT_MS = 8000;
let stopping = false;
const say = (msg) => console.log(`\x1b[35m[dev]\x1b[0m ${msg}`);

const python = venvPython(root);
const plan = backendPlan({ alreadyRunning: await portOpen(BACKEND_PORT), pythonExists: existsSync(python) });

let backend = null;
if (plan === "reuse") {
  say(`a backend is already running on port ${BACKEND_PORT}; using that one`);
} else if (plan === "skip") {
  say(`no Python at ${python}: starting the page only. Create the venv (see README) or run \`npm run doctor\`.`);
} else {
  mkdirSync(join(root, "logs"), { recursive: true });
  const logFile = createWriteStream(join(root, "logs", "orchestrator.log"), { flags: "w" });
  const started = Date.now();
  backend = spawn(python, ["app.py"], {
    cwd: join(root, "orchestrator"),
    env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  for (const stream of [backend.stdout, backend.stderr]) {
    stream.setEncoding("utf8");
    const lines = lineSplitter((line) => console.log(`\x1b[36m[backend]\x1b[0m ${line}`));
    stream.on("data", (chunk) => {
      logFile.write(chunk);
      lines.push(chunk);
    });
    stream.on("end", () => lines.flush());
  }
  backend.on("error", (err) => say(`could not start the backend: ${err.message}`));
  backend.on("exit", (code) => {
    if (!stopping) say(`the backend exited (code ${code}). See logs/orchestrator.log. The page keeps running; restart with \`npm run dev\`.`);
  });
  // Tell the user when the page can actually talk to something.
  (async () => {
    while (backend.exitCode === null && Date.now() - started < 90_000) {
      if (await portOpen(BACKEND_PORT, 400)) {
        say(`backend ready in ${((Date.now() - started) / 1000).toFixed(1)}s; the page connects on its own`);
        return;
      }
      await new Promise((r) => setTimeout(r, 300));
    }
  })();
}

const viteBin = join(root, "node_modules", "vite", "bin", "vite.js");
const vite = spawn(process.execPath, [viteBin, ...process.argv.slice(2)], { cwd: root, stdio: "inherit" });

function killTree(child) {
  if (!child || child.exitCode !== null || !child.pid) return;
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
  else child.kill("SIGKILL");
}

async function stopAll(code = 0) {
  if (stopping) return;
  stopping = true;
  if (backend && backend.exitCode === null) {
    say("stopping the backend (finishing the session-end memory pass)...");
    try {
      await fetch(`http://127.0.0.1:${BACKEND_PORT}/shutdown`, { method: "POST", signal: AbortSignal.timeout(2000) });
    } catch {
      /* it may already be closing; the wait below covers it */
    }
    const deadline = Date.now() + GRACEFUL_WAIT_MS;
    while (backend.exitCode === null && Date.now() < deadline) await new Promise((r) => setTimeout(r, 100));
    killTree(backend);
  }
  if (vite.exitCode === null) vite.kill();
  process.exit(code);
}

process.on("SIGINT", () => stopAll(0));
process.on("SIGTERM", () => stopAll(0));
vite.on("exit", (code) => stopAll(code ?? 0));
