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
  offerKind: "standard" as const,
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

test("overflow offer is available only after an opted-in daily quota is exhausted", async () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  const originalFetch = globalThis.fetch;
  process.env.TEST_PROVIDER_KEY = "test";
  globalThis.fetch = async () => new Response('{"ok":true}', { status: 200 });
  try {
    const dailyPolicy: QuotaPolicy = {
      maxConcurrent: null,
      dailyWindow: { type: "rolling" },
      limits: {
        requestsPerMinute: null,
        inputTokensPerMinute: 1_000,
        requestsPerDay: 1,
        minimumSpacingMs: 0,
      },
    };
    const p = new OpenAICompatibleProvider({
      id: "test",
      priority: 1,
      credentialEnv: "TEST_PROVIDER_KEY",
      baseUrl: "https://example.test/v1",
      defaultRetryMs: 5_000,
      providerFailureCooldownMs: 15_000,
      overflowProbe: {
        hardCapTtlMs: 24 * 60 * 60 * 1000,
        limits: ["requestsPerDay"],
      },
      models: [
        { id: "test/daily", upstreamModel: "daily", contextWindowTokens: 2_000, quota: dailyPolicy },
      ],
    });

    const initial = p.getBestOffer({ ...autoRequest, estimatedInputTokens: 50 }, Date.now())!;
    assert.equal(initial.kind, "standard");
    const result = await p.execute(initial, { messages: [] }, false, 50, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status === "success") result.release();

    const now = Date.now();
    const standard = p.getBestOffer({ ...autoRequest, estimatedInputTokens: 50 }, now)!;
    assert.equal(standard.kind, "standard");
    assert.ok(standard.availableAt > now + 60_000);

    const overflow = p.getBestOffer(
      { ...autoRequest, offerKind: "overflow", estimatedInputTokens: 50 },
      now,
    )!;
    assert.equal(overflow.kind, "overflow");
    assert.equal(overflow.availableAt, now);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});

test("overflow 429 suppresses more overflow probes for that model", async () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  const originalFetch = globalThis.fetch;
  process.env.TEST_PROVIDER_KEY = "test";
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return calls === 1
      ? new Response('{"ok":true}', { status: 200 })
      : new Response('{"error":"quota"}', { status: 429 });
  };
  try {
    const dailyPolicy: QuotaPolicy = {
      maxConcurrent: null,
      dailyWindow: { type: "rolling" },
      limits: {
        requestsPerMinute: null,
        inputTokensPerMinute: 1_000,
        requestsPerDay: 1,
        minimumSpacingMs: 0,
      },
    };
    const p = new OpenAICompatibleProvider({
      id: "test",
      priority: 1,
      credentialEnv: "TEST_PROVIDER_KEY",
      baseUrl: "https://example.test/v1",
      defaultRetryMs: 5_000,
      providerFailureCooldownMs: 15_000,
      overflowProbe: {
        hardCapTtlMs: 24 * 60 * 60 * 1000,
        limits: ["requestsPerDay"],
      },
      models: [
        { id: "test/daily", upstreamModel: "daily", contextWindowTokens: 2_000, quota: dailyPolicy },
      ],
    });

    const first = p.getBestOffer({ ...autoRequest, estimatedInputTokens: 50 }, Date.now())!;
    const firstResult = await p.execute(first, { messages: [] }, false, 50, new AbortController().signal);
    if (firstResult.status === "success") firstResult.release();

    const overflow = p.getBestOffer(
      { ...autoRequest, offerKind: "overflow", estimatedInputTokens: 50 },
      Date.now(),
    )!;
    const overflowResult = await p.execute(
      overflow,
      { messages: [] },
      false,
      50,
      new AbortController().signal,
    );
    assert.equal(overflowResult.status, "retryable");
    if (overflowResult.status === "retryable") {
      assert.equal(overflowResult.scope, "model");
      assert.equal(overflowResult.reason, "overflow_limit_confirmed");
    }

    const blocked = p.getBestOffer(
      { ...autoRequest, offerKind: "overflow", estimatedInputTokens: 50 },
      Date.now() + 60_000,
    );
    assert.equal(blocked, null);

    const status = p.status();
    assert.ok((status.models[0]?.overflowBlockedUntil ?? 0) > Date.now() + 23 * 60 * 60 * 1000);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});

test("overflow never bypasses hard single-request capacity", () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  process.env.TEST_PROVIDER_KEY = "test";
  try {
    const dailyPolicy: QuotaPolicy = {
      maxConcurrent: null,
      dailyWindow: { type: "rolling" },
      limits: {
        requestsPerMinute: null,
        inputTokensPerMinute: 100,
        requestsPerDay: 1,
        minimumSpacingMs: 0,
      },
    };
    const p = new OpenAICompatibleProvider({
      id: "test",
      priority: 1,
      credentialEnv: "TEST_PROVIDER_KEY",
      baseUrl: "https://example.test/v1",
      defaultRetryMs: 5_000,
      providerFailureCooldownMs: 15_000,
      overflowProbe: {
        hardCapTtlMs: 24 * 60 * 60 * 1000,
        limits: ["requestsPerDay"],
      },
      models: [
        { id: "test/small", upstreamModel: "small", contextWindowTokens: 1_000, quota: dailyPolicy },
      ],
    });

    const offer = p.getBestOffer(
      { ...autoRequest, offerKind: "overflow", estimatedInputTokens: 200 },
      Date.now(),
    );
    assert.equal(offer, null);
  } finally {
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});

test("overflow can continue with another exhausted model after one model confirms its cap", async () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  const originalFetch = globalThis.fetch;
  process.env.TEST_PROVIDER_KEY = "test";
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return calls <= 2
      ? new Response('{"ok":true}', { status: 200 })
      : new Response('{"error":"quota"}', { status: 429 });
  };
  try {
    const dailyPolicy: QuotaPolicy = {
      maxConcurrent: null,
      dailyWindow: { type: "rolling" },
      limits: {
        requestsPerMinute: null,
        inputTokensPerMinute: 1_000,
        requestsPerDay: 1,
        minimumSpacingMs: 0,
      },
    };
    const p = new OpenAICompatibleProvider({
      id: "test",
      priority: 1,
      credentialEnv: "TEST_PROVIDER_KEY",
      baseUrl: "https://example.test/v1",
      defaultRetryMs: 5_000,
      providerFailureCooldownMs: 15_000,
      overflowProbe: {
        hardCapTtlMs: 24 * 60 * 60 * 1000,
        limits: ["requestsPerDay"],
      },
      models: [
        { id: "test/a", upstreamModel: "a", contextWindowTokens: 2_000, quota: dailyPolicy },
        { id: "test/b", upstreamModel: "b", contextWindowTokens: 2_000, quota: dailyPolicy },
      ],
    });

    const first = p.getBestOffer(autoRequest, Date.now())!;
    const firstResult = await p.execute(first, { messages: [] }, false, 50, new AbortController().signal);
    if (firstResult.status === "success") firstResult.release();

    const second = p.getBestOffer(autoRequest, Date.now())!;
    assert.equal(second.modelId, "test/b");
    const secondResult = await p.execute(second, { messages: [] }, false, 50, new AbortController().signal);
    if (secondResult.status === "success") secondResult.release();

    const overflowA = p.getBestOffer(
      { ...autoRequest, offerKind: "overflow" },
      Date.now(),
    )!;
    assert.equal(overflowA.modelId, "test/a");
    const failed = await p.execute(
      overflowA,
      { messages: [] },
      false,
      50,
      new AbortController().signal,
    );
    assert.equal(failed.status, "retryable");

    const overflowB = p.getBestOffer(
      { ...autoRequest, offerKind: "overflow" },
      Date.now() + 60_000,
    );
    assert.equal(overflowB?.modelId, "test/b");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});

test("overflow upstream failure does not create a hard-cap observation", async () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  const originalFetch = globalThis.fetch;
  process.env.TEST_PROVIDER_KEY = "test";
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return calls === 1
      ? new Response('{"ok":true}', { status: 200 })
      : new Response('{"error":"temporary"}', { status: 503 });
  };
  try {
    const dailyPolicy: QuotaPolicy = {
      maxConcurrent: null,
      dailyWindow: { type: "rolling" },
      limits: {
        requestsPerMinute: null,
        inputTokensPerMinute: 1_000,
        requestsPerDay: 1,
        minimumSpacingMs: 0,
      },
    };
    const p = new OpenAICompatibleProvider({
      id: "test",
      priority: 1,
      credentialEnv: "TEST_PROVIDER_KEY",
      baseUrl: "https://example.test/v1",
      defaultRetryMs: 5_000,
      providerFailureCooldownMs: 15_000,
      overflowProbe: {
        hardCapTtlMs: 24 * 60 * 60 * 1000,
        limits: ["requestsPerDay"],
      },
      models: [
        { id: "test/daily", upstreamModel: "daily", contextWindowTokens: 2_000, quota: dailyPolicy },
      ],
    });

    const first = p.getBestOffer(autoRequest, Date.now())!;
    const firstResult = await p.execute(first, { messages: [] }, false, 50, new AbortController().signal);
    if (firstResult.status === "success") firstResult.release();

    const overflow = p.getBestOffer(
      { ...autoRequest, offerKind: "overflow" },
      Date.now(),
    )!;
    const failed = await p.execute(
      overflow,
      { messages: [] },
      false,
      50,
      new AbortController().signal,
    );
    assert.equal(failed.status, "retryable");
    if (failed.status === "retryable") assert.equal(failed.scope, "provider");
    assert.equal(p.status().models[0]?.overflowBlockedUntil, null);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});

test("provider without overflow configuration never advertises overflow", () => {
  const originalKey = process.env.TEST_PROVIDER_KEY;
  process.env.TEST_PROVIDER_KEY = "test";
  try {
    const p = provider();
    const offer = p.getBestOffer(
      { ...autoRequest, offerKind: "overflow" },
      Date.now(),
    );
    assert.equal(offer, null);
  } finally {
    if (originalKey === undefined) delete process.env.TEST_PROVIDER_KEY;
    else process.env.TEST_PROVIDER_KEY = originalKey;
  }
});
