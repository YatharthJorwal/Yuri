import { test } from "node:test";
import assert from "node:assert/strict";
import { backendPlan, lineSplitter, venvPython } from "../scripts/dev-lib.mjs";

test("the venv python path matches each platform (and what the Tauri shell uses on Windows)", () => {
  assert.match(venvPython("/r", "win32"), /\.venv[\\/]Scripts[\\/]python\.exe$/);
  assert.match(venvPython("/r", "linux"), /\.venv[\\/]bin[\\/]python$/);
});

test("lines split across chunks are reassembled, and CRLF is trimmed", () => {
  const lines = [];
  const s = lineSplitter((l) => lines.push(l));
  s.push("[yuri] star");
  s.push("ting\r\nsecond line\nthi");
  s.push("rd");
  assert.deepEqual(lines, ["[yuri] starting", "second line"]);
  s.flush();
  assert.deepEqual(lines, ["[yuri] starting", "second line", "third"]);
  s.flush(); // nothing left: no empty line
  assert.equal(lines.length, 3);
});

test("a backend that is already up is reused, never duplicated", () => {
  assert.equal(backendPlan({ alreadyRunning: true, pythonExists: true }), "reuse");
  assert.equal(backendPlan({ alreadyRunning: true, pythonExists: false }), "reuse");
});

test("with no venv the page still starts; otherwise the backend does", () => {
  assert.equal(backendPlan({ alreadyRunning: false, pythonExists: false }), "skip");
  assert.equal(backendPlan({ alreadyRunning: false, pythonExists: true }), "start");
});
