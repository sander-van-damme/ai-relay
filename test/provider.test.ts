import assert from "node:assert/strict";
import test from "node:test";
import type { QuotaPolicy } from "../src/providers/shared/quota.ts";
import { OpenAICompatibleProvider } from "../src/providers/shared/openai-compatible.ts";

function policy(tpm: number): QuotaPolicy {
  return {
    maxConcurrent: null,
    dailyWindow: { type: "rolling" },
    limits: {
      requestsPerMinute: null,
      inputTokensPerMinute: tpm,
      requestsPerDay: null,
      minimumSpacingMs: 0,
    },
  };
}

function provider() {
  return new OpenAICompatibleProvider({
    id: "test",
    priority: 1,
    credentialEnv: "TEST_PROVIDER_KEY",
    baseUrl: "https://example.test/v1",
    defaultRetryMs: 5_000,
    providerFailureCooldownMs: 15_000,
    models: [
      { id: "test/small", upstreamModel: "small", contextWindowTokens: 1_000, quota: policy(100) },
      { id: "test/large", upstreamModel: "large", contextWindowTokens: 1_000, quota: policy(1_000) },
    ],
  });
}

const autoRequest = {
  requestedModel: "auto",
  estimatedInputTokens: 50,
  maxOptimizationWaitMs: 15_000,
  excludedModelIds: new Set<string>(),
};

test("provider uses the smallest effective capacity that fits within the wait cutoff", async () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  const originalFetch = globalThis.fetch;
  process.env.TEST_PROVIDER_KEY = "test";
  globalThis.fetch = async () => new Response('{"ok":true}', { status: 200 });
  try {
    const p = provider();
    const base = Date.now();
    const first = p.getBestOffer({ ...autoRequest, estimatedInputTokens: 100 }, base);
    assert.equal(first?.modelId, "test/small");
    const result = await p.execute(first!, { messages: [] }, false, 100, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status === "success") result.release();

    const soon = p.getBestOffer(autoRequest, base + 1_000);
    assert.equal(soon?.modelId, "test/large");

    const patient = p.getBestOffer({ ...autoRequest, maxOptimizationWaitMs: 61_000 }, base + 1_000);
    assert.equal(patient?.modelId, "test/small");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});

test("network failure puts the provider behind a cooldown", async () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  const originalFetch = globalThis.fetch;
  process.env.TEST_PROVIDER_KEY = "test";
  globalThis.fetch = async () => { throw new Error("down"); };
  try {
    const p = provider();
    const now = Date.now();
    const offer = p.getBestOffer(autoRequest, now)!;
    const result = await p.execute(offer, { messages: [] }, false, 50, new AbortController().signal);
    assert.equal(result.status, "retryable");
    if (result.status === "retryable") assert.equal(result.scope, "provider");
    const next = p.getBestOffer(autoRequest, now + 1);
    assert.ok(next);
    assert.ok((next?.availableAt ?? 0) >= now + 15_000);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});

test("authentication rejection is provider-scoped", async () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  const originalFetch = globalThis.fetch;
  process.env.TEST_PROVIDER_KEY = "test";
  globalThis.fetch = async () => new Response('{"error":"bad key"}', { status: 401 });
  try {
    const p = provider();
    const offer = p.getBestOffer(autoRequest, Date.now())!;
    const result = await p.execute(offer, { messages: [] }, false, 50, new AbortController().signal);
    assert.equal(result.status, "rejected");
    if (result.status === "rejected") assert.equal(result.scope, "provider");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});
