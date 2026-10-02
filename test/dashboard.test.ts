import assert from "node:assert/strict";
import test from "node:test";
import { OBSERVABILITY_HTML } from "../src/dashboard.ts";

test("observability dashboard only renders called-model outcome stats", () => {
  assert.match(OBSERVABILITY_HTML, /Called models/);
  assert.match(OBSERVABILITY_HTML, /Model calls/);
  assert.match(OBSERVABILITY_HTML, /Successful calls/);
  assert.match(OBSERVABILITY_HTML, /Failed calls/);
  assert.match(OBSERVABILITY_HTML, /\['Model','Calls','Successes','Failures'\]/);
  assert.doesNotMatch(OBSERVABILITY_HTML, /<h2>Providers<\/h2>/);
  assert.doesNotMatch(OBSERVABILITY_HTML, /<h2>Requested models<\/h2>/);
});
