import assert from "node:assert/strict";
import test from "node:test";
import { createProviders } from "../src/providers/index.ts";
import { GROQ_MODELS, GroqProvider, type GroqModel } from "../src/providers/groq/index.ts";
import { countGroqInputTokens } from "../src/providers/groq/token-count.ts";

const body = { messages: [{ role: "user", content: "hello" }] };
const baseRequest = {
  offerKind: "standard" as const,
  body,
  requestedModel: "auto",
  maxOptimizationWaitMs: 15_000,
  excludedModelIds: new Set<string>(),
};

function jsonResponse(value: unknown, status = 200, headers?: HeadersInit): Response {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json", ...headers } });
}

function installKey(): () => void {
  const original = process.env.GROQ_API_KEY;
  process.env.GROQ_API_KEY = "test";
  return () => original === undefined ? delete process.env.GROQ_API_KEY : void (process.env.GROQ_API_KEY = original);
}

function gptModel(id: string, limits: Partial<GroqModel["quota"]["limits"]> = {}, tokensPerDay = 200_000): GroqModel {
  const base = GROQ_MODELS.find((model) => model.upstreamModel === "openai/gpt-oss-20b")!;
  return {
    ...base,
    id,
    quota: { ...base.quota, limits: { ...base.quota.limits, ...limits } },
    tokensPerDay,
  };
}

function gptOnly(): GroqProvider {
  return new GroqProvider(GROQ_MODELS.filter((model) => model.tokenizer.kind === "gpt-oss"));
}

async function offer(provider: GroqProvider, modelId = "groq/openai/gpt-oss-20b", requestBody = body) {
  const result = await provider.getBestOffer({ ...baseRequest, requestedModel: modelId, body: requestBody }, Date.now());
  assert.equal(result.status, "offer");
  if (result.status !== "offer") throw new Error("expected offer");
  return result.offer;
}

test("Groq is registered and exposes the current free general-chat catalog", () => {
  assert.ok(createProviders().some((provider) => provider.id === "groq"));
  assert.deepEqual(GROQ_MODELS.map((model) => model.id), [
    "groq/openai/gpt-oss-120b",
    "groq/qwen/qwen3.8-27b",
    "groq/openai/gpt-oss-20b",
  ]);
  assert.ok(GROQ_MODELS.every((model) => model.quota.limits.requestsPerMinute === 30));
  assert.ok(GROQ_MODELS.every((model) => model.quota.limits.requestsPerDay === 1_000));
  assert.ok(GROQ_MODELS.every((model) => model.quota.limits.inputTokensPerMinute === 8_000));
  assert.ok(GROQ_MODELS.every((model) => model.tokensPerDay === 200_000));
  assert.ok(new GroqProvider().listModels().every((model) => model.inputCapacityTokens === 8_000));
});

test("Groq requires GROQ_API_KEY", async () => {
  const original = process.env.GROQ_API_KEY;
  delete process.env.GROQ_API_KEY;
  try {
    assert.deepEqual(await new GroqProvider().getBestOffer(baseRequest, Date.now()), {
      status: "no_offer", providerId: "groq", reason: "provider_not_configured",
    });
  } finally {
    if (original === undefined) delete process.env.GROQ_API_KEY; else process.env.GROQ_API_KEY = original;
  }
});

test("GPT-OSS token counting is local and stable", async () => {
  const first = await countGroqInputTokens(body, { kind: "gpt-oss" });
  const second = await countGroqInputTokens(body, { kind: "gpt-oss" });
  assert.ok(first > 0);
  assert.equal(second, first);
});

test("explicit GPT-OSS selection stays on the requested model and auto prefers 120B at equal capacity", async () => {
  const restore = installKey();
  try {
    const provider = gptOnly();
    const explicit = await offer(provider);
    assert.equal(explicit.modelId, "groq/openai/gpt-oss-20b");
    const auto = await provider.getBestOffer(baseRequest, Date.now());
    assert.equal(auto.status, "offer");
    if (auto.status === "offer") assert.equal(auto.offer.modelId, "groq/openai/gpt-oss-120b");
  } finally { restore(); }
});

test("known unsupported requests, GPT vision requests, and overflow do not produce offers", async () => {
  const restore = installKey();
  try {
    const provider = gptOnly();
    for (const request of [
      { ...baseRequest, body: { ...body, logprobs: true } },
      { ...baseRequest, body: { ...body, store: true } },
      { ...baseRequest, body: { ...body, functions: [{ name: "legacy", parameters: { type: "object" } }] } },
      { ...baseRequest, body: { ...body, service_tier: "flex" } },
      { ...baseRequest, requestedModel: "groq/openai/gpt-oss-20b", body: { ...body, max_completion_tokens: 65_537 } },
      { ...baseRequest, requestedModel: "groq/openai/gpt-oss-20b", body: { messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "https://example.com/a.png" } }] }] } },
      { ...baseRequest, offerKind: "overflow" as const },
    ]) {
      const result = await provider.getBestOffer(request, Date.now());
      assert.equal(result.status, "no_offer");
    }
  } finally { restore(); }
});

test("hard capacity and token-count failures remain distinguishable", async () => {
  const restore = installKey();
  try {
    const model = gptModel("groq/test/tiny", { inputTokensPerMinute: 1 });
    const provider = new GroqProvider([model]);
    const capacity = await provider.getBestOffer({ ...baseRequest, requestedModel: model.id }, Date.now());
    assert.equal(capacity.status, "no_offer");
    if (capacity.status === "no_offer") assert.equal(capacity.reason, "request_exceeds_capacity");
    const counting = await provider.getBestOffer({ ...baseRequest, requestedModel: model.id, body: { messages: "bad" } }, Date.now());
    assert.equal(counting.status, "no_offer");
    if (counting.status === "no_offer") assert.equal(counting.reason, "token_count_failed");
  } finally { restore(); }
});

test("successful execution uses Groq auth/endpoint/model and reports usage", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  let promptTokens = 0;
  globalThis.fetch = async (input, init) => {
    assert.equal(String(input), "https://api.groq.com/openai/v1/chat/completions");
    assert.equal(new Headers(init?.headers).get("authorization"), "Bearer test");
    const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
    assert.equal(request.model, "openai/gpt-oss-20b");
    return jsonResponse({ choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: 5, total_tokens: promptTokens + 5, prompt_tokens_details: { cached_tokens: 0 } } });
  };
  try {
    const provider = gptOnly();
    const selected = await offer(provider);
    promptTokens = selected.inputTokens;
    const result = await provider.execute(selected, body, false, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status !== "success") return;
    assert.deepEqual(await result.usage, { inputTokens: promptTokens, outputTokens: 5, totalTokens: promptTokens + 5 });
    assert.equal(provider.status().models.find((model) => model.id === selected.modelId)?.active, 1);
    result.release();
    assert.equal(provider.status().models.find((model) => model.id === selected.modelId)?.active, 0);
  } finally { globalThis.fetch = originalFetch; restore(); }
});

test("Groq preserves partial upstream usage without fabricating missing dimensions", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  try {
    let prompt = 0;
    globalThis.fetch = async () => jsonResponse({
      choices: [],
      usage: { prompt_tokens: prompt, total_tokens: prompt + 1 },
    });
    const inputAndTotalProvider = gptOnly();
    const inputAndTotalOffer = await offer(inputAndTotalProvider);
    prompt = inputAndTotalOffer.inputTokens;
    const inputAndTotal = await inputAndTotalProvider.execute(
      inputAndTotalOffer,
      body,
      false,
      new AbortController().signal,
    );
    assert.equal(inputAndTotal.status, "success");
    if (inputAndTotal.status !== "success") return;
    assert.deepEqual(await inputAndTotal.usage, {
      inputTokens: prompt,
      totalTokens: prompt + 1,
    });
    inputAndTotal.release();

    globalThis.fetch = async () => jsonResponse({
      choices: [],
      usage: { completion_tokens: 3 },
    });
    const outputOnlyProvider = gptOnly();
    const outputOnlyOffer = await offer(outputOnlyProvider);
    const outputOnly = await outputOnlyProvider.execute(
      outputOnlyOffer,
      body,
      false,
      new AbortController().signal,
    );
    assert.equal(outputOnly.status, "success");
    if (outputOnly.status !== "success") return;
    assert.deepEqual(await outputOnly.usage, { outputTokens: 3 });
    outputOnly.release();
  } finally {
    globalThis.fetch = originalFetch;
    restore();
  }
});

test("Groq disables parallel function calls for current non-parallel models", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  const toolBody = {
    messages: [{ role: "user", content: "weather?" }],
    tools: [{ type: "function", function: { name: "weather", parameters: { type: "object", properties: {} } } }],
  };
  globalThis.fetch = async (_input, init) => {
    const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
    assert.equal(request.parallel_tool_calls, false);
    return jsonResponse({ choices: [], usage: { prompt_tokens: 1, completion_tokens: 0, total_tokens: 1 } });
  };
  try {
    const provider = gptOnly();
    const selected = await offer(provider, "groq/openai/gpt-oss-20b", toolBody);
    const result = await provider.execute(selected, toolBody, false, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status === "success") {
      await result.usage;
      result.release();
    }
  } finally { globalThis.fetch = originalFetch; restore(); }
});

test("RPM exhaustion changes availableAt instead of dropping the offer", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  const model = gptModel("groq/test/rpm", { requestsPerMinute: 1 });
  globalThis.fetch = async () => jsonResponse({ choices: [], usage: { prompt_tokens: 1, completion_tokens: 0, total_tokens: 1 } });
  try {
    const provider = new GroqProvider([model]);
    const first = await offer(provider, model.id);
    const result = await provider.execute(first, body, false, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status !== "success") return;
    await result.usage; result.release();
    const now = Date.now();
    const second = await provider.getBestOffer({ ...baseRequest, requestedModel: model.id }, now);
    assert.equal(second.status, "offer");
    if (second.status === "offer") assert.ok(second.offer.availableAt >= now + 50_000);
  } finally { globalThis.fetch = originalFetch; restore(); }
});

test("completion tokens are reconciled into Groq TPD accounting", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  const input = await countGroqInputTokens(body, { kind: "gpt-oss" });
  const model = gptModel("groq/test/tpd", {}, input + 5);
  globalThis.fetch = async () => jsonResponse({ choices: [], usage: { prompt_tokens: input, completion_tokens: 10, total_tokens: input + 10 } });
  try {
    const provider = new GroqProvider([model]);
    const first = await offer(provider, model.id);
    const result = await provider.execute(first, body, false, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status !== "success") return;
    await result.usage; result.release();
    const now = Date.now();
    const second = await provider.getBestOffer({ ...baseRequest, requestedModel: model.id }, now);
    assert.equal(second.status, "offer");
    if (second.status === "offer") assert.ok(second.offer.availableAt >= now + 23 * 60 * 60 * 1_000);
  } finally { globalThis.fetch = originalFetch; restore(); }
});

test("429 is model-scoped and honors Groq reset headers", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse({ error: { code: "rate_limit_exceeded" } }, 429, {
    "retry-after": "2", "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "7.66s",
  });
  try {
    const provider = gptOnly();
    const selected = await offer(provider);
    const failedAt = Date.now();
    const result = await provider.execute(selected, body, false, new AbortController().signal);
    assert.equal(result.status, "retryable");
    if (result.status === "retryable") {
      assert.equal(result.scope, "model");
      assert.equal(result.reason, "rate_limit");
      assert.ok(result.retryAt >= failedAt + 7_000);
    }
  } finally { globalThis.fetch = originalFetch; restore(); }
});

test("498 is model-scoped; 5xx is provider-scoped; auth/spend blocks are provider rejections", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  try {
    const cases: Array<[number, unknown, string, string]> = [
      [498, { error: { code: "capacity_exceeded" } }, "retryable", "model"],
      [503, { error: { message: "down" } }, "retryable", "provider"],
      [401, { error: { message: "bad key" } }, "rejected", "provider"],
      [400, { error: { code: "blocked_api_access" } }, "rejected", "provider"],
    ];
    for (const [status, payload, expectedStatus, expectedScope] of cases) {
      globalThis.fetch = async () => jsonResponse(payload, status);
      const provider = gptOnly();
      const result = await provider.execute(await offer(provider), body, false, new AbortController().signal);
      assert.equal(result.status, expectedStatus);
      if (result.status === "success") throw new Error("expected failure");
      assert.equal(result.scope, expectedScope);
    }
  } finally { globalThis.fetch = originalFetch; restore(); }
});

test("Groq streaming hides relay-forced usage while still observing it", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  let prompt = 0;
  globalThis.fetch = async (_input, init) => {
    const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
    assert.deepEqual(request.stream_options, { include_usage: true });
    return new Response([
      `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "hi" } }] })}\n\n`,
      `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: prompt, completion_tokens: 2, total_tokens: prompt + 2 } })}\n\n`,
      "data: [DONE]\n\n",
    ].join(""), { headers: { "content-type": "text/event-stream" } });
  };
  try {
    const provider = gptOnly();
    const streamBody = { ...body, stream_options: { include_usage: false } };
    const selected = await offer(provider, "groq/openai/gpt-oss-20b", streamBody);
    prompt = selected.inputTokens;
    const result = await provider.execute(selected, streamBody, true, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status !== "success") return;

    const downstream = await result.response.text();
    assert.match(downstream, /"content":"hi"/);
    assert.doesNotMatch(downstream, /"choices":\[\],"usage":/);
    assert.match(downstream, /data: \[DONE\]/);
    assert.deepEqual(await result.usage, { inputTokens: prompt, outputTokens: 2, totalTokens: prompt + 2 });
    result.release();
  } finally { globalThis.fetch = originalFetch; restore(); }
});

test("Groq streaming preserves usage when the client requests it", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  let prompt = 0;
  globalThis.fetch = async (_input, init) => {
    const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
    assert.deepEqual(request.stream_options, { include_usage: true });
    return new Response([
      `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: "hi" } }] })}\n\n`,
      `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: prompt, completion_tokens: 2, total_tokens: prompt + 2 } })}\n\n`,
      "data: [DONE]\n\n",
    ].join(""), { headers: { "content-type": "text/event-stream" } });
  };
  try {
    const provider = gptOnly();
    const streamBody = { ...body, stream_options: { include_usage: true } };
    const selected = await offer(provider, "groq/openai/gpt-oss-20b", streamBody);
    prompt = selected.inputTokens;
    const result = await provider.execute(selected, streamBody, true, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status !== "success") return;

    const downstream = await result.response.text();
    assert.match(downstream, /"choices":\[\],"usage":/);
    assert.match(downstream, /data: \[DONE\]/);
    assert.deepEqual(await result.usage, { inputTokens: prompt, outputTokens: 2, totalTokens: prompt + 2 });
    result.release();
  } finally { globalThis.fetch = originalFetch; restore(); }
});


test("Groq strips replayed GPT-OSS reasoning fields before sending the next turn", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  const replayBody = {
    messages: [
      { role: "user", content: "first" },
      {
        role: "assistant",
        content: "visible answer",
        reasoning: "hidden provider reasoning",
        reasoning_content: "alternate hidden reasoning",
      },
      { role: "user", content: "continue" },
    ],
  };

  globalThis.fetch = async (_input, init) => {
    const request = JSON.parse(String(init?.body)) as {
      messages: Array<Record<string, unknown>>;
    };
    assert.deepEqual(request.messages, [
      { role: "user", content: "first" },
      { role: "assistant", content: "visible answer" },
      { role: "user", content: "continue" },
    ]);
    return jsonResponse({ choices: [], usage: { prompt_tokens: 1, completion_tokens: 0, total_tokens: 1 } });
  };

  try {
    const provider = gptOnly();
    const selected = await offer(provider, "groq/openai/gpt-oss-20b", replayBody);
    const result = await provider.execute(selected, replayBody, false, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status === "success") {
      await result.usage;
      result.release();
    }
  } finally {
    globalThis.fetch = originalFetch;
    restore();
  }
});

test("Groq treats GPT-OSS Harmony parser crashes as model-scoped rejection", async () => {
  const restore = installKey();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse({
    error: {
      message: 'openai_harmony.HarmonyError: unexpected tokens remaining in message header: Some("to=functions.bash")',
    },
  }, 500);

  try {
    const provider = gptOnly();
    const result = await provider.execute(
      await offer(provider),
      body,
      false,
      new AbortController().signal,
    );
    assert.equal(result.status, "rejected");
    if (result.status === "rejected") {
      assert.equal(result.scope, "model");
      assert.equal(result.httpStatus, 500);
    }
    assert.equal(provider.status().blockedUntil, null);
  } finally {
    globalThis.fetch = originalFetch;
    restore();
  }
});
