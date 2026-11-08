// Checks the things that make the dev window "not load", and says how to fix each.
//   npm run doctor      full report (also checks the backend, Ollama and the Rust toolchain)
//   npm run dev         runs this first in --brief mode (via the `predev` hook): prints only
//                       problems, never blocks, so a healthy start stays silent.
import { formatReport, runChecks } from "./doctor-lib.mjs";

const brief = process.argv.includes("--brief");
let results = [];
try {
  results = await runChecks({ brief });
} catch (err) {
  console.error(`[yuri] doctor failed unexpectedly: ${err?.message ?? err}`);
}
const text = formatReport(results, { brief });
if (text) console.log(text);
// Brief mode runs ahead of the dev server and must never block it.
process.exit(!brief && results.some((r) => r.level === "fail") ? 1 : 0);
