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

test("NVIDIA catalog exposes all candidates with tokenizer specs and capacities", () => {
  assert.equal(NVIDIA_MODELS.length, 15);
  assert.ok(NVIDIA_MODELS.every((model) => model.enabled));
  assert.ok(NVIDIA_MODELS.every(
    (model) => typeof model.contextWindowTokens === "number" && model.contextWindowTokens > 0,
  ));
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
  assert.equal(provider.listModels().length, NVIDIA_MODELS.length);
  assert.deepEqual(
    provider.listModels().map((model) => model.id),
    NVIDIA_MODELS.map((model) => model.id),
  );
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
    const gptOssRequest = { ...baseRequest, requestedModel: "nvidia/openai/gpt-oss-20b" };
    const first = await provider.getBestOffer(gptOssRequest, Date.now());
    assert.equal(first.status, "offer");
    if (first.status === "offer") assert.ok(first.offer.inputTokens > 0);

    const second = await provider.getBestOffer(gptOssRequest, Date.now());
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
    const offerResult = await provider.getBestOffer({
      ...baseRequest,
      requestedModel: "nvidia/openai/gpt-oss-20b",
    }, Date.now());
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
    const modelStatus = provider.status().models.find(
      (model) => model.id === "nvidia/openai/gpt-oss-20b",
    );
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
    assert.equal(request.stream, false);
    assert.equal(request.stream_options, undefined);
    return jsonResponse({
      choices: [],
      usage: { prompt_tokens: expectedPromptTokens, completion_tokens: 5, total_tokens: expectedPromptTokens + 5 },
    });
  };

  try {
    const provider = createNvidiaProvider();
    const offerResult = await provider.getBestOffer({
      ...baseRequest,
      requestedModel: "nvidia/openai/gpt-oss-20b",
    }, Date.now());
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
    assert.equal(provider.status().models.find((model) => model.id === "nvidia/openai/gpt-oss-20b")?.active, 1);
    result.release();
    assert.equal(provider.status().models.find((model) => model.id === "nvidia/openai/gpt-oss-20b")?.active, 0);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});


test("NVIDIA streaming hides relay-forced usage when the client disables it", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  let expectedPromptTokens = 0;
  let observedRequest: Record<string, unknown> | undefined;

  globalThis.fetch = async (_input, init) => {
    observedRequest = JSON.parse(String(init?.body)) as Record<string, unknown>;
    const usage = {
      prompt_tokens: expectedPromptTokens,
      completion_tokens: 7,
      total_tokens: expectedPromptTokens + 7,
    };
    const sse = [
      'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n',
      `data: ${JSON.stringify({ id: "chatcmpl-test", choices: [], usage })}\n\n`,
      "data: [DONE]\n\n",
    ].join("");
    const bytes = new TextEncoder().encode(sse);
    const splitAt = Math.floor(bytes.byteLength / 2);

    return new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, splitAt));
        controller.enqueue(bytes.slice(splitAt));
        controller.close();
      },
    }), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  };

  try {
    const provider = createNvidiaProvider();
    const streamBody = {
      ...body,
      stream_options: { include_usage: false },
    };
    const offerResult = await provider.getBestOffer({
      ...baseRequest,
      body: streamBody,
      requestedModel: "nvidia/openai/gpt-oss-20b",
    }, Date.now());
    assert.equal(offerResult.status, "offer");
    if (offerResult.status !== "offer") return;
    expectedPromptTokens = offerResult.offer.inputTokens;

    const result = await provider.execute(
      offerResult.offer,
      streamBody,
      true,
      new AbortController().signal,
    );
    assert.equal(result.status, "success");
    if (result.status !== "success") return;

    assert.equal(observedRequest?.stream, true);
    assert.deepEqual(observedRequest?.stream_options, { include_usage: true });

    const downstream = await result.response.text();
    assert.equal(downstream, [
      'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n',
      "data: [DONE]\n\n",
    ].join(""));
    assert.deepEqual(await result.usage, {
      inputTokens: expectedPromptTokens,
      outputTokens: 7,
      totalTokens: expectedPromptTokens + 7,
    });

    result.release();
    assert.equal(provider.status().models.find((model) => model.id === "nvidia/openai/gpt-oss-20b")?.active, 0);
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA streaming preserves usage when the client requests it", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  let expectedPromptTokens = 0;

  globalThis.fetch = async (_input, init) => {
    const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
    assert.deepEqual(request.stream_options, { include_usage: true });
    const usage = {
      prompt_tokens: expectedPromptTokens,
      completion_tokens: 3,
      total_tokens: expectedPromptTokens + 3,
    };
    return new Response([
      'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n',
      `data: ${JSON.stringify({ id: "chatcmpl-test", choices: [], usage })}\n\n`,
      "data: [DONE]\n\n",
    ].join(""), { headers: { "content-type": "text/event-stream" } });
  };

  try {
    const provider = createNvidiaProvider();
    const streamBody = { ...body, stream_options: { include_usage: true } };
    const offerResult = await provider.getBestOffer({
      ...baseRequest,
      body: streamBody,
      requestedModel: "nvidia/openai/gpt-oss-20b",
    }, Date.now());
    assert.equal(offerResult.status, "offer");
    if (offerResult.status !== "offer") return;
    expectedPromptTokens = offerResult.offer.inputTokens;

    const result = await provider.execute(offerResult.offer, streamBody, true, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status !== "success") return;

    const downstream = await result.response.text();
    assert.match(downstream, /"choices":\[\],"usage":/);
    assert.match(downstream, /data: \[DONE\]/);
    assert.deepEqual(await result.usage, {
      inputTokens: expectedPromptTokens,
      outputTokens: 3,
      totalTokens: expectedPromptTokens + 3,
    });
    result.release();
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA streaming leaves usage unavailable when no usage event is returned", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (_input, init) => {
    const request = JSON.parse(String(init?.body)) as Record<string, unknown>;
    assert.equal(request.stream, true);
    assert.deepEqual(request.stream_options, { include_usage: true });
    return new Response(
      'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n',
      { status: 200, headers: { "content-type": "text/event-stream" } },
    );
  };

  try {
    const provider = createNvidiaProvider();
    const offerResult = await provider.getBestOffer({
      ...baseRequest,
      requestedModel: "nvidia/openai/gpt-oss-20b",
    }, Date.now());
    assert.equal(offerResult.status, "offer");
    if (offerResult.status !== "offer") return;

    const result = await provider.execute(
      offerResult.offer,
      body,
      true,
      new AbortController().signal,
    );
    assert.equal(result.status, "success");
    if (result.status !== "success") return;

    const downstream = await result.response.text();
    assert.match(downstream, /data: \[DONE\]/);
    assert.equal(await result.usage, undefined);
    result.release();
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA does not expose speculative overflow offers", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    throw new Error("unexpected fetch");
  };

  try {
    const provider = createNvidiaProvider();
    assert.equal(provider.listModels().length, 15);

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


test("NVIDIA strips replayed GPT-OSS reasoning fields before sending the next turn", async () => {
  const restoreKey = installKey();
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
    const provider = createNvidiaProvider();
    const offerResult = await provider.getBestOffer({
      ...baseRequest,
      body: replayBody,
      requestedModel: "nvidia/openai/gpt-oss-20b",
    }, Date.now());
    assert.equal(offerResult.status, "offer");
    if (offerResult.status !== "offer") return;

    const result = await provider.execute(
      offerResult.offer,
      replayBody,
      false,
      new AbortController().signal,
    );
    assert.equal(result.status, "success");
    if (result.status === "success") {
      await result.usage;
      result.release();
    }
  } finally {
    globalThis.fetch = originalFetch;
    restoreKey();
  }
});

test("NVIDIA treats GPT-OSS Harmony parser crashes as model-scoped rejection", async () => {
  const restoreKey = installKey();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => jsonResponse({
    error: {
      message: 'openai_harmony.HarmonyError: unexpected tokens remaining in message header: Some("to=functions.bash")',
    },
  }, 500);

  try {
    const provider = createNvidiaProvider();
    const offerResult = await provider.getBestOffer({
      ...baseRequest,
      requestedModel: "nvidia/openai/gpt-oss-20b",
    }, Date.now());
    assert.equal(offerResult.status, "offer");
    if (offerResult.status !== "offer") return;

    const result = await provider.execute(
      offerResult.offer,
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
    restoreKey();
  }
});
