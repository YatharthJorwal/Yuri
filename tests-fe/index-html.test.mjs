import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// The tests are bundled into node_modules/.cache, so locate the repo by cwd (npm runs them from the root).
const html = readFileSync(join(process.cwd(), "index.html"), "utf8");
const block = (id) => {
  const start = html.indexOf(`id="${id}"`);
  assert.ok(start >= 0, `#${id} exists`);
  return html.slice(start, html.indexOf("</svg>", start));
};

test("the status indicator is a heart: fill, track and arc paths, no circles", () => {
  const orb = block("status-orb");
  for (const cls of ["orb-fill", "orb-track", "orb-arc"]) assert.match(orb, new RegExp(`<path class="${cls}"`));
  assert.doesNotMatch(orb, /<circle/);
});

test("the attachment remove button keeps its own x icon (the heart edit once clobbered it)", () => {
  const remove = block("pending-attachment-remove");
  assert.match(remove, /d="M6 6l12 12M18 6 6 18"/);
  assert.doesNotMatch(remove, /orb-/);
});

test("all three heart paths draw the same shape", () => {
  const orb = block("status-orb");
  const shapes = [...orb.matchAll(/ d="([^"]+)"/g)].map((m) => m[1]);
  assert.equal(shapes.length, 3);
  assert.equal(new Set(shapes).size, 1);
});
