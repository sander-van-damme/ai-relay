import assert from "node:assert/strict";
import test from "node:test";
import { OBSERVABILITY_HTML } from "../src/dashboard.ts";

test("observability dashboard renders relay and generic provider-contract data", () => {
  assert.match(OBSERVABILITY_HTML, /Providers/);
  assert.match(OBSERVABILITY_HTML, /Called models/);
  assert.match(OBSERVABILITY_HTML, /Registered models/);
  assert.match(OBSERVABILITY_HTML, /Requested models/);
  assert.match(OBSERVABILITY_HTML, /Routing input tokens/);
  assert.match(OBSERVABILITY_HTML, /Upstream input tokens/);
  assert.match(OBSERVABILITY_HTML, /Upstream output tokens/);
  assert.match(OBSERVABILITY_HTML, /Upstream total tokens/);
  assert.match(OBSERVABILITY_HTML, /Input capacity/);
  assert.match(OBSERVABILITY_HTML, /Terminal failures/);
  assert.match(OBSERVABILITY_HTML, /Cancellations/);
  assert.match(OBSERVABILITY_HTML, /Failed attempts/);
  assert.match(OBSERVABILITY_HTML, /registered_models/);
  assert.match(OBSERVABILITY_HTML, /routingInputTokens/);
  assert.match(OBSERVABILITY_HTML, /upstreamInputTokens/);
});
