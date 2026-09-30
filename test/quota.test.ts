import assert from "node:assert/strict";
import test from "node:test";
import { estimateInputTokens, parseRetryAfterMs, quotaDelayMs, reserveQuota } from "../src/quota.ts";
import type { ModelConfig, ModelRuntimeState } from "../src/types.ts";

const model: ModelConfig = {
  id: "google/test",
  provider: "google",
  upstreamModel: "test",
  enabled: true,
  maxConcurrent: 1,
  limits: {
    requestsPerMinute: 2,
    inputTokensPerMinute: 100,
    requestsPerDay: 10,
    minimumSpacingMs: 0,
  },
};

function state(): ModelRuntimeState {
  return { active: 0, blockedUntil: 0, lastStartedAt: 0, events: [] };
}

test("input token estimate ignores relay model and stream fields", () => {
  const a = estimateInputTokens({ model: "a", stream: true, messages: [{ role: "user", content: "hello" }] });
  const b = estimateInputTokens({ model: "b", stream: false, messages: [{ role: "user", content: "hello" }] });
  assert.equal(a, b);
  assert.ok(a > 0);
});

test("request-per-minute window delays the third request", () => {
  const runtime = state();
  reserveQuota(runtime, 10, 1_000);
  runtime.active = 0;
  reserveQuota(runtime, 10, 2_000);
  runtime.active = 0;
  assert.equal(quotaDelayMs(model, runtime, 10, 2_500), 58_500);
});

test("token window waits until enough token usage expires", () => {
  const runtime = state();
  runtime.events = [
    { at: 1_000, inputTokens: 60 },
    { at: 20_000, inputTokens: 30 },
  ];
  assert.equal(quotaDelayMs({ ...model, limits: { ...model.limits, requestsPerMinute: null } }, runtime, 20, 30_000), 31_000);
});

test("oversized token request is not dispatchable", () => {
  const runtime = state();
  assert.equal(quotaDelayMs(model, runtime, 101, 1_000), Number.POSITIVE_INFINITY);
});

test("Retry-After seconds are converted to milliseconds", () => {
  assert.equal(parseRetryAfterMs("3", 5000), 3000);
});
