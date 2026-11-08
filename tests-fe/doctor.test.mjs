import { test } from "node:test";
import assert from "node:assert/strict";
import net from "node:net";
import { formatReport, portFree, portOpen } from "../scripts/doctor-lib.mjs";

function listen(port = 0) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.listen(port, "127.0.0.1", () => resolve(srv));
  });
}

test("a port something is listening on is reported as taken, a free one as free", async () => {
  const srv = await listen();
  const { port } = srv.address();
  assert.equal(await portFree(port), false);
  assert.equal(await portOpen(port), true);
  await new Promise((r) => srv.close(r));
  assert.equal(await portFree(port), true);
  assert.equal(await portOpen(port, 300), false);
});

test("a healthy report is empty in brief mode, so a good start prints nothing", () => {
  assert.equal(formatReport([], { brief: true }), "");
  assert.match(formatReport([], { brief: false }), /everything looks fine/);
});

test("each problem is printed with its fix", () => {
  const text = formatReport(
    [
      { level: "fail", title: "port 1420 is already in use", fix: "taskkill it" },
      { level: "warn", title: "no avatar" },
    ],
    { brief: true },
  );
  assert.match(text, /✗ port 1420 is already in use\n\s+fix: taskkill it/);
  assert.match(text, /! no avatar/);
  assert.match(text, /heads-up before the dev server starts/);
});
