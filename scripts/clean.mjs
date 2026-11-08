// Deletes regenerable build/test leftovers so the project folder stays small.
//   npm run clean        dist/, test caches, __pycache__
//   npm run clean:all    all of the above plus src-tauri/target (the Rust build, usually the
//                        biggest thing on disk; the next `tauri dev` rebuilds it, a few minutes)
// Never touches: .venv, node_modules, data/ (memory.db), logs/, config, public/vrm.
import { existsSync, readdirSync, rmSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const all = process.argv.includes("--all");

function sizeOf(path) {
  const st = statSync(path);
  if (!st.isDirectory()) return st.size;
  let total = 0;
  for (const name of readdirSync(path)) total += sizeOf(join(path, name));
  return total;
}

function* pycaches(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const full = join(dir, entry.name);
    if (entry.name === "__pycache__") yield full;
    else yield* pycaches(full);
  }
}

const targets = [
  join(root, "dist"),
  join(root, ".pytest_cache"),
  join(root, "orchestrator", ".pytest_cache"),
  join(root, "node_modules", ".cache"),
  ...(existsSync(join(root, "orchestrator")) ? pycaches(join(root, "orchestrator")) : []),
];
if (all) targets.push(join(root, "src-tauri", "target"));

let freed = 0;
for (const target of targets) {
  if (!existsSync(target)) continue;
  const bytes = sizeOf(target);
  rmSync(target, { recursive: true, force: true });
  freed += bytes;
  console.log(`removed ${target.slice(root.length + 1)} (${(bytes / 1048576).toFixed(1)} MB)`);
}
console.log(`freed ${(freed / 1048576).toFixed(1)} MB${all ? "" : " (use `npm run clean:all` to include the Rust build)"}`);
