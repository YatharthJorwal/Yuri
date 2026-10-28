// Runs the frontend's pure-logic tests with NO new dependencies: esbuild
// (already installed as a dependency of vite) bundles each tests-fe/*.test.mjs
// -- which may import ../src/*.ts directly -- and Node's built-in test runner
// executes the result. Pure logic only (math, schedulers, report builders):
// there is no browser or renderer here, so anything that needs WebGL, the
// DOM or an AudioContext is verified by eye on the real machine instead.
//
//   npm run test:fe

import { build } from "esbuild";
import { spawnSync } from "node:child_process";
import { mkdirSync, readdirSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const testsDir = join(root, "tests-fe");
const outDir = join(root, "node_modules", ".cache", "luna-fe-tests");

const files = readdirSync(testsDir).filter((f) => f.endsWith(".test.mjs"));
if (files.length === 0) {
  console.error("no tests found in tests-fe/");
  process.exit(1);
}

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

await build({
  entryPoints: files.map((f) => join(testsDir, f)),
  outdir: outDir,
  bundle: true,
  platform: "node",
  format: "esm",
  outExtension: { ".js": ".mjs" },
  logLevel: "error",
});

const result = spawnSync(
  process.execPath,
  ["--test", ...files.map((f) => join(outDir, f))],
  { stdio: "inherit" },
);
process.exit(result.status ?? 1);
