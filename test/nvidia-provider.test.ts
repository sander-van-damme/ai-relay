import assert from "node:assert/strict";
import test from "node:test";
import { createNvidiaProvider, NVIDIA_MODELS } from "../src/providers/nvidia/index.ts";
import { countNvidiaInputTokens } from "../src/providers/nvidia/token-count.ts";

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

test("NVIDIA catalog preserves all candidates, tokenizer specs, and only enables GPT-OSS", () => {
  assert.equal(NVIDIA_MODELS.length, 15);
  assert.deepEqual(
    NVIDIA_MODELS.filter((model) => model.enabled).map((model) => model.id),
    ["nvidia/openai/gpt-oss-20b"],
  );
  assert.ok(NVIDIA_MODELS.every((model) => model.tokenizer !== undefined));
  assert.equal(
    NVIDIA_MODELS.find((model) => model.id === "nvidia/openai/gpt-oss-20b")?.tokenizer.kind,
    "gpt-oss-20b",
  );
  assert.equal(
    NVIDIA_MODELS.filter((model) => model.tokenizer.kind === "huggingface").length,
    14,
  );

  const provider = createNvidiaProvider();
  assert.deepEqual(provider.listModels(), [{
    id: "nvidia/openai/gpt-oss-20b",
    providerId: "nvidia",
    inputCapacityTokens: 131_072,
  }]);
});

test("NVIDIA requires an API key before evaluating offers", async () => {
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

test("GPT-OSS token counting is local and produces a stable positive count", async () => {
  const spec = NVIDIA_MODELS.find((model) => model.id === "nvidia/openai/gpt-oss-20b")!.tokenizer;
  const first = await countNvidiaInputTokens(body, spec);
  const second = await countNvidiaInputTokens(body, spec);
  assert.ok(Number.isSafeInteger(first));
  assert.ok(first > 0);
  assert.equal(second, first);
});

test("NVIDIA offer evaluation for GPT-OSS does not make a network request", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    throw new Error("unexpected fetch");
  };

  try {
    const provider = createNvidiaProvider();
    const first = await provider.getBestOffer(baseRequest, Date.now());
    assert.equal(first.status, "offer");
    if (first.status === "offer") assert.ok(first.offer.inputTokens > 0);

    const second = await provider.getBestOffer(baseRequest, Date.now());
    assert.equal(second.status, "offer");
    if (first.status === "offer" && second.status === "offer") {
      assert.equal(second.offer.inputTokens, first.offer.inputTokens);
    }
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA treats 429 as model-scoped and respects Retry-After", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async (input) => {
    const url = String(input);
    calls += 1;
    assert.equal(url, "https://integrate.api.nvidia.com/v1/chat/completions");
    return jsonResponse({ error: { message: "rate limited" } }, 429, { "retry-after": "2" });
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

    assert.equal(calls, 1);
    const modelStatus = provider.status().models[0];
    assert.ok((modelStatus?.blockedUntil ?? 0) >= failedAt + 2_000);
    assert.equal(provider.status().blockedUntil, null);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA successful execution calls chat completions and releases concurrency state", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  let calls = 0;
  let expectedPromptTokens = 0;
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    calls += 1;
    assert.equal(url, "https://integrate.api.nvidia.com/v1/chat/completions");
    const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
    assert.equal(request.model, "openai/gpt-oss-20b");
    return jsonResponse({
      choices: [],
      usage: { prompt_tokens: expectedPromptTokens, completion_tokens: 5, total_tokens: expectedPromptTokens + 5 },
    });
  };

  try {
    const provider = createNvidiaProvider();
    const offerResult = await provider.getBestOffer(baseRequest, Date.now());
    assert.equal(offerResult.status, "offer");
    if (offerResult.status !== "offer") return;
    expectedPromptTokens = offerResult.offer.inputTokens;

    const result = await provider.execute(
      offerResult.offer,
      body,
      false,
      new AbortController().signal,
    );
    assert.equal(result.status, "success");
    if (result.status !== "success") return;

    assert.equal(calls, 1);
    assert.deepEqual(await result.usage, { inputTokens: expectedPromptTokens, outputTokens: 5, totalTokens: expectedPromptTokens + 5 });
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
