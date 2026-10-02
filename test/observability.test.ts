import assert from "node:assert/strict";
import test from "node:test";
import { Observability } from "../src/observability.ts";
import type { Provider } from "../src/providers/index.ts";

const provider = {
  id: "test",
  priority: 1,
  isConfigured: () => true,
  listModels: () => [{ id: "test/one", providerId: "test", inputCapacityTokens: 100 }],
  status: () => ({ id: "test", configured: true, blockedUntil: null, models: [{ id: "test/one", active: 0, blockedUntil: null, overflowBlockedUntil: null }] }),
} as unknown as Provider;

test("observability separates terminal requests from failed failover attempts", () => {
  const stats = new Observability([provider]);
  stats.request("auto", 1);
  stats.attempt("test", "test/one", 10);
  stats.failedAttempt("test", "test/one");
  stats.attempt("test", "test/one", 12);
  stats.success("test", "test/one", 12, { outputTokens: 5, totalTokens: 17 });
  stats.request("test/one", 2);
  stats.terminalFailure();
  stats.cancellation();

  const snapshot = stats.snapshot(0) as any;
  assert.deepEqual(snapshot.queue, { current_depth: 0, peak_depth: 2 });
  assert.equal(snapshot.totals.requests, 2);
  assert.equal(snapshot.totals.successes, 1);
  assert.equal(snapshot.totals.terminalFailures, 1);
  assert.equal(snapshot.totals.failedAttempts, 1);
  assert.equal(snapshot.totals.attempts, 2);
  assert.equal(snapshot.totals.inputTokens, 22);
  assert.equal(snapshot.totals.outputTokens, 5);
  assert.equal(snapshot.totals.totalTokens, 27);
  assert.deepEqual(snapshot.requested_models.map((row: any) => [row.model, row.kind, row.requests]), [["auto", "auto", 1], ["test/one", "explicit", 1]]);
  assert.equal(snapshot.providers.length, 1);
  assert.equal(snapshot.models.length, 1);
});
