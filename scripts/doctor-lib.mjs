// The checks behind `npm run doctor` (see doctor.mjs). Core Node modules only. Every check is
// wrapped so the doctor itself can never be the thing that breaks `npm run dev` or `tauri dev`.
import { spawnSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import net from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const win = process.platform === "win32";
const DEV_PORT = 1420; // fixed: src-tauri/tauri.conf.json devUrl and vite strictPort
const BACKEND_PORT = 8765;
const OLLAMA_PORT = 11434;

/** True if nothing is listening on `port` for any of the addresses `localhost` can resolve to. */
export async function portFree(port, hosts = ["127.0.0.1", "::1"]) {
  const results = await Promise.all(
    hosts.map(
      (host) =>
        new Promise((resolve) => {
          const srv = net.createServer();
          srv.once("error", (err) => resolve(err.code !== "EADDRINUSE")); // e.g. no IPv6 => fine
          srv.once("listening", () => srv.close(() => resolve(true)));
          srv.listen(port, host);
        }),
    ),
  );
  return results.every(Boolean);
}

/** True if something accepts TCP connections on 127.0.0.1:port within `ms`. */
export function portOpen(port, ms = 700) {
  return new Promise((resolve) => {
    const sock = net.connect({ port, host: "127.0.0.1" });
    const done = (ok) => {
      sock.destroy();
      resolve(ok);
    };
    sock.setTimeout(ms, () => done(false));
    sock.once("connect", () => done(true));
    sock.once("error", () => done(false));
  });
}

function freePortHelp(port) {
  return win
    ? `find the owner with:  netstat -ano | findstr :${port}   then:  taskkill /PID <pid> /F   (a leftover "tauri dev" or vite is the usual culprit)`
    : `find the owner with:  lsof -i :${port}   then stop it`;
}

export async function runChecks({ brief = false } = {}) {
  const out = [];
  const add = (level, title, fix) => out.push({ level, title, fix });
  const guard = async (fn) => {
    try {
      await fn();
    } catch (err) {
      add("warn", `a doctor check crashed (${err?.message ?? err}) -- ignoring it`);
    }
  };

  await guard(() => {
    const major = Number(process.versions.node.split(".")[0]);
    if (major < 18) add("fail", `Node ${process.versions.node} is too old`, "install Node 18 or newer");
  });

  await guard(() => {
    if (!existsSync(join(root, "node_modules", "vite", "package.json")))
      add("fail", "dependencies are not installed (vite is missing)", "run:  npm install");
  });

  await guard(async () => {
    if (!(await portFree(DEV_PORT)))
      add(
        "fail",
        `port ${DEV_PORT} is already in use, so the dev server cannot start (it must be exactly ${DEV_PORT})`,
        freePortHelp(DEV_PORT),
      );
  });

  await guard(() => {
    const py = win ? join(root, ".venv", "Scripts", "python.exe") : join(root, ".venv", "bin", "python");
    if (!existsSync(py))
      add(
        "warn",
        "no Python venv at .venv, so the shell cannot start the backend (the window will sit on \"connecting\")",
        "from the repo root:  python -m venv .venv   then:  .venv\\Scripts\\pip install -r orchestrator\\requirements-dev.txt",
      );
  });

  await guard(() => {
    const dir = join(root, "public", "vrm");
    const models = existsSync(dir) ? readdirSync(dir).filter((f) => f.toLowerCase().endsWith(".vrm")) : [];
    if (!existsSync(join(dir, "yuri.vrm")))
      add(
        "warn",
        models.length
          ? `public/vrm/yuri.vrm is missing (found: ${models.join(", ")}), so no avatar will appear`
          : "public/vrm/yuri.vrm is missing, so no avatar will appear",
        models.length ? "rename your model to yuri.vrm, or set window.__YURI_MODEL_PATH__" : "put your avatar at public/vrm/yuri.vrm (see public/vrm/README.txt)",
      );
  });

  if (!brief) {
    await guard(async () => {
      if (!(await portOpen(BACKEND_PORT)))
        add(
          "warn",
          `the backend is not running on port ${BACKEND_PORT}, so the page would sit on "connecting"`,
          "run `npm run dev` (starts the backend and the page together) or `npm run tauri dev`",
        );
    });
    await guard(async () => {
      if (!(await portOpen(OLLAMA_PORT)))
        add("warn", `Ollama is not reachable on port ${OLLAMA_PORT}, so replies and memory will not work`, "start Ollama");
    });
    await guard(() => {
      const r = spawnSync("cargo", ["--version"], { encoding: "utf8", shell: win });
      if (r.error || r.status !== 0) add("warn", "Rust (cargo) not found, which `tauri dev` needs", "install Rust from https://rustup.rs");
    });
  }
  return out;
}

export function formatReport(results, { brief = false } = {}) {
  if (results.length === 0) return brief ? "" : "[yuri] doctor: everything looks fine.";
  const mark = { fail: "✗", warn: "!" };
  const lines = [brief ? "[yuri] heads-up before the dev server starts:" : "[yuri] doctor:"];
  for (const r of results) {
    lines.push(`  ${mark[r.level] ?? "-"} ${r.title}`);
    if (r.fix) lines.push(`      fix: ${r.fix}`);
  }
  return lines.join("\n");
}
