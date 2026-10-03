import assert from "node:assert/strict";
import test from "node:test";
import { Observability } from "../src/observability.ts";
import type { Provider } from "../src/providers/index.ts";

const provider = {
  id: "test",
  priority: 1,
  isConfigured: () => true,
  listModels: () => [
    { id: "test/one", providerId: "test", inputCapacityTokens: 100 },
    { id: "test/unused", providerId: "test", inputCapacityTokens: 200 },
  ],
  status: () => ({
    id: "test",
    configured: true,
    blockedUntil: null,
    models: [
      { id: "test/one", active: 1, blockedUntil: null, overflowBlockedUntil: null },
      { id: "test/unused", active: 0, blockedUntil: null, overflowBlockedUntil: null },
    ],
  }),
} as unknown as Provider;

test("observability separates registered provider-contract state from called-model stats", () => {
  const stats = new Observability([provider]);

  const initial = stats.snapshot(0) as any;
  assert.deepEqual(initial.models, []);
  assert.deepEqual(initial.requested_models, []);
  assert.deepEqual(
    initial.registered_models.map((row: any) => [row.id, row.provider, row.input_capacity_tokens, row.active]),
    [["test/one", "test", 100, 1], ["test/unused", "test", 200, 0]],
  );
  assert.equal(initial.providers[0].priority, 1);
  assert.equal(initial.providers[0].configured, true);

  stats.request("auto", 1);
  stats.attempt("test", "test/one", 10);
  stats.failedAttempt("test", "test/one");
  stats.attempt("test", "test/one", 12);
  stats.success("test", "test/one", { inputTokens: 11, outputTokens: 5, totalTokens: 16 });
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
  assert.equal(snapshot.totals.routingInputTokens, 22);
  assert.equal(snapshot.totals.upstreamInputTokens, 11);
  assert.equal(snapshot.totals.upstreamOutputTokens, 5);
  assert.equal(snapshot.totals.upstreamTotalTokens, 16);
  assert.deepEqual(
    snapshot.requested_models.map((row: any) => [row.model, row.kind, row.requests]),
    [["auto", "auto", 1], ["test/one", "explicit", 1]],
  );
  assert.equal(snapshot.providers.length, 1);
  assert.deepEqual(
    snapshot.models.map((row: any) => [
      row.id,
      row.provider,
      row.input_capacity_tokens,
      row.attempts,
      row.successes,
      row.failedAttempts,
    ]),
    [["test/one", "test", 100, 2, 1, 1]],
  );
});


test("observability keeps missing upstream usage unavailable", () => {
  const stats = new Observability([provider]);
  stats.attempt("test", "test/one", 12);
  stats.success("test", "test/one");

  const snapshot = stats.snapshot(0) as any;
  assert.equal(snapshot.totals.routingInputTokens, 12);
  assert.equal(snapshot.totals.upstreamInputTokens, null);
  assert.equal(snapshot.totals.upstreamOutputTokens, null);
  assert.equal(snapshot.totals.upstreamTotalTokens, null);
});
