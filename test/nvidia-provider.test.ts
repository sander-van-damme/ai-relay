import assert from "node:assert/strict";
import test from "node:test";
import { createNvidiaProvider, NVIDIA_MODELS } from "../src/providers/nvidia/index.ts";

const body = {
  messages: [{ role: "user", content: "hello" }],
};

const baseRequest = {
  offerKind: "standard" as const,
  body,
  requestedModel: "auto",
  maxOptimizationWaitMs: 15_000,
  excludedModelIds: new Set<string>(),
};

function jsonResponse(value: unknown, status = 200, headers?: HeadersInit): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

function installKey(): () => void {
  const originalKey = process.env.NVIDIA_API_KEY;
  process.env.NVIDIA_API_KEY = "test";
  return () => {
    if (originalKey === undefined) delete process.env.NVIDIA_API_KEY;
    else process.env.NVIDIA_API_KEY = originalKey;
  };
}

test("NVIDIA catalog preserves all candidates but exposes only the verified model", () => {
  assert.equal(NVIDIA_MODELS.length, 15);
  assert.deepEqual(
    NVIDIA_MODELS.filter((model) => model.enabled).map((model) => model.id),
    ["nvidia/openai/gpt-oss-20b"],
  );
  const provider = createNvidiaProvider();
  assert.deepEqual(provider.listModels(), [{
    id: "nvidia/openai/gpt-oss-20b",
    providerId: "nvidia",
    inputCapacityTokens: 128_000,
  }]);
});

test("NVIDIA requires an API key before probing token endpoints", async () => {
  const originalKey = process.env.NVIDIA_API_KEY;
  const originalFetch = globalThis.fetch;
  delete process.env.NVIDIA_API_KEY;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    throw new Error("unexpected fetch");
  };
  try {
    const result = await createNvidiaProvider().getBestOffer(baseRequest, Date.now());
    assert.deepEqual(result, {
      status: "no_offer",
      providerId: "nvidia",
      reason: "provider_not_configured",
    });
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.NVIDIA_API_KEY;
    else process.env.NVIDIA_API_KEY = originalKey;
  }
});

test("NVIDIA probes render and tokenize, selects render count, and caches the result", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  const calls = new Map<string, number>();
  globalThis.fetch = async (input) => {
    const url = String(input);
    calls.set(url, (calls.get(url) ?? 0) + 1);
    if (url.endsWith("/v1/chat/completions/render")) {
      return jsonResponse({ token_ids: [1, 2, 3, 4] });
    }
    if (url.endsWith("/tokenize")) {
      return jsonResponse({ count: 4, max_model_len: 128_000, tokens: [1, 2, 3, 4] });
    }
    throw new Error(`unexpected URL: ${url}`);
  };

  try {
    const provider = createNvidiaProvider();
    const first = await provider.getBestOffer(baseRequest, Date.now());
    assert.equal(first.status, "offer");
    if (first.status === "offer") assert.equal(first.offer.inputTokens, 4);

    const second = await provider.getBestOffer(baseRequest, Date.now());
    assert.equal(second.status, "offer");
    assert.equal(calls.get("https://integrate.api.nvidia.com/v1/chat/completions/render"), 1);
    assert.equal(calls.get("https://integrate.api.nvidia.com/tokenize"), 1);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA falls back to tokenize when hosted chat render is unavailable", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.endsWith("/v1/chat/completions/render")) {
      return jsonResponse({ error: "not found" }, 404);
    }
    if (url.endsWith("/tokenize")) {
      return jsonResponse({ count: 7, max_model_len: 128_000, tokens: [1, 2, 3, 4, 5, 6, 7] });
    }
    throw new Error(`unexpected URL: ${url}`);
  };

  try {
    const result = await createNvidiaProvider().getBestOffer(baseRequest, Date.now());
    assert.equal(result.status, "offer");
    if (result.status === "offer") assert.equal(result.offer.inputTokens, 7);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA refuses to advertise an offer when both authoritative token probes fail", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse({ error: "unsupported" }, 404);

  try {
    const result = await createNvidiaProvider().getBestOffer(baseRequest, Date.now());
    assert.equal(result.status, "no_offer");
    if (result.status === "no_offer") {
      assert.equal(result.reason, "token_count_failed");
      assert.match(result.detail ?? "", /chat_render=404/);
      assert.match(result.detail ?? "", /tokenize=404/);
    }
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA treats 429 as model-scoped and respects Retry-After", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.endsWith("/v1/chat/completions/render")) return jsonResponse({ token_ids: [1, 2, 3] });
    if (url.endsWith("/tokenize")) return jsonResponse({ count: 3, tokens: [1, 2, 3], max_model_len: 128_000 });
    if (url.endsWith("/v1/chat/completions")) {
      return jsonResponse({ error: { message: "rate limited" } }, 429, { "retry-after": "2" });
    }
    throw new Error(`unexpected URL: ${url}`);
  };

  try {
    const provider = createNvidiaProvider();
    const offerResult = await provider.getBestOffer(baseRequest, Date.now());
    assert.equal(offerResult.status, "offer");
    if (offerResult.status !== "offer") return;

    const failedAt = Date.now();
    const result = await provider.execute(
      offerResult.offer,
      body,
      false,
      new AbortController().signal,
    );
    assert.equal(result.status, "retryable");
    if (result.status === "retryable") {
      assert.equal(result.scope, "model");
      assert.equal(result.reason, "rate_limit");
      assert.ok(result.retryAt >= failedAt + 2_000);
    }
    const modelStatus = provider.status().models[0];
    assert.ok((modelStatus?.blockedUntil ?? 0) >= failedAt + 2_000);
    assert.equal(provider.status().blockedUntil, null);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA successful execution releases model concurrency state", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    const url = String(input);
    if (url.endsWith("/v1/chat/completions/render")) return jsonResponse({ token_ids: [1, 2] });
    if (url.endsWith("/tokenize")) return jsonResponse({ count: 2, tokens: [1, 2], max_model_len: 128_000 });
    if (url.endsWith("/v1/chat/completions")) return jsonResponse({ choices: [] });
    throw new Error(`unexpected URL: ${url}`);
  };

  try {
    const provider = createNvidiaProvider();
    const offerResult = await provider.getBestOffer(baseRequest, Date.now());
    assert.equal(offerResult.status, "offer");
    if (offerResult.status !== "offer") return;

    const result = await provider.execute(
      offerResult.offer,
      body,
      false,
      new AbortController().signal,
    );
    assert.equal(result.status, "success");
    if (result.status !== "success") return;
    assert.equal(provider.status().models[0]?.active, 1);
    result.release();
    assert.equal(provider.status().models[0]?.active, 0);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA does not expose disabled catalog models or speculative overflow offers", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    throw new Error("unexpected fetch");
  };

  try {
    const provider = createNvidiaProvider();
    const disabled = await provider.getBestOffer({
      ...baseRequest,
      requestedModel: "nvidia/deepseek-ai/deepseek-v4.1-flash",
    }, Date.now());
    assert.equal(disabled.status, "no_offer");
    if (disabled.status === "no_offer") assert.equal(disabled.reason, "no_eligible_model");

    const overflow = await provider.getBestOffer({
      ...baseRequest,
      offerKind: "overflow" as const,
    }, Date.now());
    assert.equal(overflow.status, "no_offer");
    if (overflow.status === "no_offer") assert.equal(overflow.reason, "no_eligible_model");
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});
