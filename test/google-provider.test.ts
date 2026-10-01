import assert from "node:assert/strict";
import test from "node:test";
import type { GoogleGenAI } from "@google/genai";
import {
  GOOGLE_MODELS,
  GoogleProvider,
  googleInteractionToOpenAI,
  type GoogleModel,
} from "../src/providers/google/index.ts";

const TEST_MODEL: GoogleModel = {
  id: "google/test-model",
  upstreamModel: "test-model",
  contextWindowTokens: 1_000,
  quota: {
    maxConcurrent: null,
    dailyWindow: { type: "rolling" },
    limits: {
      requestsPerMinute: 10,
      inputTokensPerMinute: 1_000,
      requestsPerDay: 1,
      minimumSpacingMs: 0,
    },
  },
};

const standardRequest = {
  offerKind: "standard" as const,
  body: { messages: [{ role: "user", content: "Hello" }] },
  requestedModel: TEST_MODEL.id,
  maxOptimizationWaitMs: 15_000,
  excludedModelIds: new Set<string>(),
};

function fakeClient(create: (request: unknown) => unknown | Promise<unknown>): GoogleGenAI {
  return {
    models: {
      countTokens: async () => ({ totalTokens: 10 }),
      generateContent: async (request: unknown) => create(request),
      generateContentStream: async (request: unknown) => create(request),
    },
    interactions: {
      create: async (request: unknown) => create(request),
    },
  } as unknown as GoogleGenAI;
}

function completedInteraction(id: string, text = "Hello back") {
  return {
    id,
    status: "completed",
    created: "2026-10-01T08:00:00Z",
    steps: [
      { type: "model_output", content: [{ type: "text", text }] },
    ],
    usage: {
      total_input_tokens: 10,
      total_output_tokens: 3,
      total_tokens: 13,
    },
  };
}

test("Google catalog contains every non-zero text-output model from the AI Studio limits", () => {
  assert.deepEqual(GOOGLE_MODELS.map((model) => model.id), [
    "google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b-it",
    "google/gemini-robotics-er-2-preview",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.1-flash-lite",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3.8-flash",
    "google/gemini-3.7-flash",
    "google/gemini-3.6-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
  ]);

  const robotics = GOOGLE_MODELS.find((model) => model.id === "google/gemini-robotics-er-2-preview")!;
  assert.equal(robotics.contextWindowTokens, 131_072);
  assert.equal(robotics.quota.limits.requestsPerMinute, 5);
  assert.equal(robotics.quota.limits.inputTokensPerMinute, 250_000);
  assert.equal(robotics.quota.limits.requestsPerDay, 20);
  assert.equal(robotics.transport, "generate-content");
  assert.deepEqual(robotics.thinkingLevels, ["minimal", "low", "medium", "high"]);

  const flashLite = GOOGLE_MODELS.find((model) => model.id === "google/gemini-3.5-flash-lite")!;
  assert.equal(flashLite.quota.limits.requestsPerMinute, 15);
  assert.equal(flashLite.quota.limits.inputTokensPerMinute, 250_000);
  assert.equal(flashLite.quota.limits.requestsPerDay, 500);

  const gemma = GOOGLE_MODELS.find((model) => model.id === "google/gemma-4-26b-a4b-it")!;
  assert.equal(gemma.contextWindowTokens, 262_144);
  assert.equal(gemma.quota.limits.requestsPerMinute, 30);
  assert.equal(gemma.quota.limits.inputTokensPerMinute, 16_000);
  assert.equal(gemma.quota.limits.requestsPerDay, 14_400);
});

test("Google interaction response is translated to OpenAI chat-completion shape", () => {
  const response = googleInteractionToOpenAI({
    id: "interaction-1",
    status: "requires_action",
    created: "2026-10-01T08:00:00Z",
    steps: [
      { type: "model_output", content: [{ type: "text", text: "Checking." }] },
      { type: "function_call", id: "call_1", name: "weather", arguments: { city: "Ghent" } },
    ],
    usage: { total_input_tokens: 20, total_output_tokens: 5, total_tokens: 25 },
  }, "google/gemini-3.8-flash", 20);

  assert.equal(response.id, "chatcmpl-interaction-1");
  assert.equal(response.model, "google/gemini-3.8-flash");
  const choices = response.choices as Array<Record<string, unknown>>;
  assert.equal(choices[0]?.finish_reason, "tool_calls");
  assert.deepEqual(choices[0]?.message, {
    role: "assistant",
    content: "Checking.",
    tool_calls: [
      {
        id: "call_1",
        type: "function",
        function: { name: "weather", arguments: "{\"city\":\"Ghent\"}" },
      },
    ],
  });
  assert.deepEqual(response.usage, {
    prompt_tokens: 20,
    completion_tokens: 5,
    total_tokens: 25,
  });
});

test("Google execution uses the official Interactions API and reuses stored continuation", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  const requests: unknown[] = [];
  let call = 0;
  const client = fakeClient((request) => {
    requests.push(request);
    call += 1;
    return completedInteraction(`interaction-${call}`, call === 1 ? "Hello back" : "Second answer");
  });

  try {
    const provider = new GoogleProvider(() => client, [TEST_MODEL]);
    const firstBody = { messages: [{ role: "user", content: "Hello" }] };
    const firstOffer = (await provider.getBestOffer({ ...standardRequest, body: firstBody }, Date.now()))!;
    const firstResult = await provider.execute(firstOffer, firstBody, false, new AbortController().signal);
    assert.equal(firstResult.status, "success");
    if (firstResult.status !== "success") return;
    const firstPayload = await firstResult.response.json() as Record<string, unknown>;
    assert.equal(firstPayload.model, TEST_MODEL.id);
    firstResult.release();

    const secondBody = {
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hello back" },
        { role: "user", content: "And again?" },
      ],
    };
    const secondOffer = (await provider.getBestOffer({ ...standardRequest, body: secondBody }, Date.now()))!;
    const secondResult = await provider.execute(secondOffer, secondBody, false, new AbortController().signal);
    assert.equal(secondResult.status, "success");
    if (secondResult.status === "success") secondResult.release();

    const firstRequest = requests[0] as Record<string, unknown>;
    const secondRequest = requests[1] as Record<string, unknown>;
    assert.equal(firstRequest.model, "test-model");
    assert.equal(firstRequest.store, true);
    assert.equal(secondRequest.previous_interaction_id, "interaction-1");
    assert.deepEqual(secondRequest.input, [
      { type: "user_input", content: [{ type: "text", text: "And again?" }] },
    ]);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("Robotics uses native generateContent and preserves Google content for continuation", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  const robotics = GOOGLE_MODELS.find((model) => model.id === "google/gemini-robotics-er-2-preview")!;
  const requests: Array<Record<string, unknown>> = [];
  let call = 0;
  const client = fakeClient((request) => {
    requests.push(request as Record<string, unknown>);
    call += 1;
    return {
      responseId: `robot-${call}`,
      createTime: "2026-10-01T08:00:00Z",
      candidates: [{
        finishReason: "STOP",
        content: {
          role: "model",
          parts: [{ text: call === 1 ? "Move complete" : "Second move", thoughtSignature: `sig-${call}` }],
        },
      }],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 2,
        totalTokenCount: 12,
      },
    };
  });

  try {
    const provider = new GoogleProvider(() => client, [robotics]);
    const firstBody = { messages: [{ role: "user", content: "Move once" }] };
    const firstOffer = (await provider.getBestOffer({
      ...standardRequest,
      body: firstBody,
      requestedModel: robotics.id,
    }, Date.now()))!;
    const firstResult = await provider.execute(firstOffer, firstBody, false, new AbortController().signal);
    assert.equal(firstResult.status, "success");
    if (firstResult.status !== "success") return;
    const firstPayload = await firstResult.response.json() as Record<string, unknown>;
    const firstChoices = firstPayload.choices as Array<Record<string, unknown>>;
    assert.deepEqual(firstChoices[0]?.message, { role: "assistant", content: "Move complete" });
    firstResult.release();

    const secondBody = {
      messages: [
        { role: "user", content: "Move once" },
        { role: "assistant", content: "Move complete" },
        { role: "user", content: "Move again" },
      ],
    };
    const secondOffer = (await provider.getBestOffer({
      ...standardRequest,
      body: secondBody,
      requestedModel: robotics.id,
    }, Date.now()))!;
    const secondResult = await provider.execute(secondOffer, secondBody, false, new AbortController().signal);
    assert.equal(secondResult.status, "success");
    if (secondResult.status === "success") secondResult.release();

    assert.equal(requests[0]?.model, "gemini-robotics-er-2-preview");
    const secondContents = requests[1]?.contents as Array<Record<string, unknown>>;
    assert.equal(secondContents.length, 3);
    const savedModel = secondContents[1] as Record<string, unknown>;
    const savedParts = savedModel.parts as Array<Record<string, unknown>>;
    assert.equal(savedParts[0]?.thoughtSignature, "sig-1");
    assert.deepEqual(secondContents[2], { role: "user", parts: [{ text: "Move again" }] });
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("Google streaming is translated to OpenAI SSE chunks", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";

  async function* stream() {
    yield {
      event_type: "interaction.created",
      interaction: { id: "stream-1", status: "in_progress", created: "2026-10-01T08:00:00Z" },
    };
    yield { event_type: "step.delta", index: 0, delta: { type: "text", text: "Hello" } };
    yield {
      event_type: "interaction.completed",
      interaction: {
        id: "stream-1",
        status: "completed",
        usage: { total_input_tokens: 10, total_output_tokens: 1, total_tokens: 11 },
      },
    };
  }

  try {
    const client = fakeClient(() => stream());
    const provider = new GoogleProvider(() => client, [TEST_MODEL]);
    const body = {
      messages: [{ role: "user", content: "Hello" }],
      stream_options: { include_usage: true },
    };
    const offer = (await provider.getBestOffer({ ...standardRequest, body }, Date.now()))!;
    const result = await provider.execute(offer, body, true, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status !== "success") return;
    const text = await result.response.text();
    result.release();

    assert.match(text, /"role":"assistant"/);
    assert.match(text, /"content":"Hello"/);
    assert.match(text, /"finish_reason":"stop"/);
    assert.match(text, /"prompt_tokens":10/);
    assert.match(text, /data: \[DONE\]/);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("Google overflow only bypasses RPD and a 429 blocks that model for exactly 24 hours", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  let createCalls = 0;
  const client = fakeClient(() => {
    createCalls += 1;
    if (createCalls === 1) return completedInteraction("first");
    throw Object.assign(new Error("quota exhausted"), {
      statusCode: 429,
      headers: new Headers(),
      body: '{"error":"quota exhausted"}',
    });
  });

  try {
    const provider = new GoogleProvider(() => client, [TEST_MODEL]);
    const firstOffer = (await provider.getBestOffer(standardRequest, Date.now()))!;
    const firstResult = await provider.execute(firstOffer, standardRequest.body, false, new AbortController().signal);
    assert.equal(firstResult.status, "success");
    if (firstResult.status === "success") firstResult.release();

    const now = Date.now();
    const standard = (await provider.getBestOffer(standardRequest, now))!;
    assert.ok(standard.availableAt > now + 60_000);

    const overflow = (await provider.getBestOffer({ ...standardRequest, offerKind: "overflow" }, now))!;
    assert.equal(overflow.kind, "overflow");
    assert.ok(overflow.availableAt <= Date.now());

    const beforeFailure = Date.now();
    const overflowResult = await provider.execute(
      overflow,
      standardRequest.body,
      false,
      new AbortController().signal,
    );
    assert.equal(overflowResult.status, "retryable");
    if (overflowResult.status === "retryable") {
      assert.equal(overflowResult.scope, "model");
      assert.equal(overflowResult.reason, "overflow_limit_confirmed");
    }

    const blockedUntil = provider.status().models[0]?.overflowBlockedUntil;
    assert.ok(blockedUntil !== null && blockedUntil !== undefined);
    assert.ok(blockedUntil! >= beforeFailure + 24 * 60 * 60 * 1000);
    assert.ok(blockedUntil! <= Date.now() + 24 * 60 * 60 * 1000);
    assert.equal(provider.status(blockedUntil! - 1).models[0]?.overflowBlockedUntil, blockedUntil);
    assert.equal(provider.status(blockedUntil!).models[0]?.overflowBlockedUntil, null);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("non-quota Google failures do not create an overflow hard-cap observation", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  let createCalls = 0;
  const client = fakeClient(() => {
    createCalls += 1;
    if (createCalls === 1) return completedInteraction("first");
    throw Object.assign(new Error("temporary"), {
      statusCode: 503,
      headers: new Headers(),
      body: '{"error":"temporary"}',
    });
  });

  try {
    const provider = new GoogleProvider(() => client, [TEST_MODEL]);
    const firstOffer = (await provider.getBestOffer(standardRequest, Date.now()))!;
    const firstResult = await provider.execute(firstOffer, standardRequest.body, false, new AbortController().signal);
    if (firstResult.status === "success") firstResult.release();

    const overflow = (await provider.getBestOffer(
      { ...standardRequest, offerKind: "overflow" },
      Date.now(),
    ))!;
    const failed = await provider.execute(overflow, standardRequest.body, false, new AbortController().signal);
    assert.equal(failed.status, "retryable");
    if (failed.status === "retryable") assert.equal(failed.scope, "provider");
    assert.equal(provider.status().models[0]?.overflowBlockedUntil, null);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});