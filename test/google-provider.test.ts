import assert from "node:assert/strict";
import test from "node:test";
import type { GoogleGenAI } from "@google/genai";
import {
  GOOGLE_MODELS,
  GoogleProvider,
  googleInteractionToOpenAI,
  type GoogleModel,
} from "../src/providers/google/index.ts";
import { calendarDayBounds } from "../src/providers/shared/quota.ts";

const TEST_MODEL: GoogleModel = {
  id: "google/test-model",
  upstreamModel: "test-model",
  contextWindowTokens: 1_000,
  preference: 100,
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

async function bestOffer(
  provider: GoogleProvider,
  request: Parameters<GoogleProvider["getBestOffer"]>[0],
  now?: number,
) {
  const result = await provider.getBestOffer(request, now);
  return result.status === "offer" ? result.offer : null;
}

function fakeClient(
  create: (request: unknown) => unknown | Promise<unknown>,
  count?: (request: unknown) => void,
): GoogleGenAI {
  return {
    models: {
      countTokens: async (request: unknown) => {
        count?.(request);
        return { totalTokens: 10 };
      },
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

test("Google catalog contains every eligible route from the AI Studio limits", () => {
  assert.deepEqual(GOOGLE_MODELS.map((model) => model.id), [
    "google/gemini-3.8-flash",
    "google/antigravity-preview-09-2026",
    "google/gemini-3.7-flash",
    "google/gemini-3.6-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3-flash-preview",
    "google/gemini-robotics-er-2-preview",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemma-4-31b-it",
    "google/gemma-4-26b-a4b-it",
  ]);

  const antigravity = GOOGLE_MODELS.find((model) => model.id === "google/antigravity-preview-09-2026")!;
  assert.equal(antigravity.upstreamAgent, "antigravity-preview-09-2026");
  assert.equal(antigravity.upstreamModel, "gemini-3.8-flash");
  assert.equal(antigravity.contextWindowTokens, 1_048_576);
  assert.equal(antigravity.quota.limits.requestsPerMinute, 60);
  assert.equal(antigravity.quota.limits.inputTokensPerMinute, 100_000);
  assert.equal(antigravity.quota.limits.requestsPerDay, 100);
  assert.equal(antigravity.preference, 1_200);

  const robotics = GOOGLE_MODELS.find((model) => model.id === "google/gemini-robotics-er-2-preview")!;
  assert.equal(robotics.contextWindowTokens, 131_072);
  assert.equal(robotics.quota.limits.requestsPerMinute, 5);
  assert.equal(robotics.quota.limits.inputTokensPerMinute, 250_000);
  assert.equal(robotics.quota.limits.requestsPerDay, 20);
  assert.equal(robotics.transport, "interactions");
  assert.deepEqual(robotics.thinkingLevels, ["minimal", "low", "medium", "high"]);

  const flashLite = GOOGLE_MODELS.find((model) => model.id === "google/gemini-3.5-flash-lite")!;
  assert.equal(flashLite.quota.limits.requestsPerMinute, 15);
  assert.equal(flashLite.quota.limits.inputTokensPerMinute, 250_000);
  assert.equal(flashLite.quota.limits.requestsPerDay, 500);

  const gemma31 = GOOGLE_MODELS.find((model) => model.id === "google/gemma-4-31b-it")!;
  const gemma26 = GOOGLE_MODELS.find((model) => model.id === "google/gemma-4-26b-a4b-it")!;
  assert.equal(gemma31.contextWindowTokens, 262_144);
  assert.equal(gemma31.quota.limits.requestsPerMinute, 30);
  assert.equal(gemma31.quota.limits.inputTokensPerMinute, 16_000);
  assert.equal(gemma31.quota.limits.requestsPerDay, 14_400);
  assert.ok(gemma31.preference > gemma26.preference);
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

test("Google counts system instructions and tools through the full Developer API request", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  let sdkCountCalls = 0;
  const developerCounts: Array<{
    apiKey: string;
    model: string;
    input: Record<string, unknown>;
  }> = [];

  try {
    const provider = new GoogleProvider(
      () => fakeClient(
        () => completedInteraction("unused"),
        () => { sdkCountCalls += 1; },
      ),
      [TEST_MODEL],
      async (apiKey, model, input) => {
        developerCounts.push({
          apiKey,
          model,
          input: input as unknown as Record<string, unknown>,
        });
        return 37;
      },
    );
    const body = {
      messages: [
        { role: "system", content: "Follow the task carefully." },
        { role: "user", content: "Inspect the project." },
      ],
      tools: [{
        type: "function",
        function: {
          name: "read_file",
          description: "Read a file",
          parameters: {
            type: "object",
            properties: { path: { type: "string" } },
            required: ["path"],
          },
        },
      }],
    };

    const offer = await bestOffer(provider, {
      ...standardRequest,
      body,
    }, Date.now());

    assert.equal(offer?.inputTokens, 37);
    assert.equal(sdkCountCalls, 0);
    assert.equal(developerCounts.length, 1);
    assert.equal(developerCounts[0]?.apiKey, "test");
    assert.equal(developerCounts[0]?.model, "test-model");
    const config = developerCounts[0]?.input.config as Record<string, unknown>;
    assert.match(JSON.stringify(config.systemInstruction), /Follow the task carefully/);
    assert.match(JSON.stringify(config.tools), /read_file/);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
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
    const firstOffer = (await bestOffer(provider, { ...standardRequest, body: firstBody }, Date.now()))!;
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
    const secondOffer = (await bestOffer(provider, { ...standardRequest, body: secondBody }, Date.now()))!;
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

test("Antigravity uses the agent API with Search and caller-provided functions only", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  const antigravity = GOOGLE_MODELS.find((model) => model.id === "google/antigravity-preview-09-2026")!;
  const requests: Array<Record<string, unknown>> = [];
  const countRequests: Array<Record<string, unknown>> = [];
  let call = 0;
  const client = fakeClient((request) => {
    requests.push(request as Record<string, unknown>);
    call += 1;
    return completedInteraction(`agent-${call}`, call === 1 ? "First answer" : "Second answer");
  });

  try {
    const provider = new GoogleProvider(
      () => client,
      [antigravity],
      async (_apiKey, model, input) => {
        countRequests.push({ model, config: input.config } as Record<string, unknown>);
        return 10;
      },
    );
    const tools = [{
      type: "function",
      function: {
        name: "read_file",
        description: "Read a file from the caller's workspace",
        parameters: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
    }];
    const firstBody = {
      messages: [
        { role: "system", content: "Follow the task carefully." },
        { role: "user", content: "Inspect the project." },
      ],
      tools,
      tool_choice: "auto",
      max_completion_tokens: 32_000,
    };
    const firstOffer = (await bestOffer(provider, {
      ...standardRequest,
      body: firstBody,
      requestedModel: antigravity.id,
    }, Date.now()))!;
    const firstResult = await provider.execute(firstOffer, firstBody, false, new AbortController().signal);
    assert.equal(firstResult.status, "success");
    if (firstResult.status !== "success") return;
    firstResult.release();

    assert.equal(countRequests[0]?.model, "gemini-3.8-flash");
    const countConfig = countRequests[0]?.config as Record<string, unknown>;
    assert.match(JSON.stringify(countConfig.systemInstruction), /reasoning backend for an OpenAI-compatible chat-completions interface/);
    assert.match(JSON.stringify(countConfig.tools), /googleSearch/);

    const firstRequest = requests[0]!;
    assert.equal(firstRequest.agent, "antigravity-preview-09-2026");
    assert.equal(firstRequest.model, undefined);
    assert.equal(firstRequest.environment, undefined);
    assert.deepEqual(firstRequest.agent_config, {
      type: "antigravity",
      model: "gemini-3.8-flash",
      max_total_tokens: "32010",
    });
    assert.match(String(firstRequest.system_instruction), /reasoning backend for an OpenAI-compatible chat-completions interface/);
    assert.match(String(firstRequest.system_instruction), /Follow the task carefully/);
    assert.deepEqual(firstRequest.tools, [
      { type: "google_search" },
      {
        type: "function",
        name: "read_file",
        description: "Read a file from the caller's workspace",
        parameters: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
      },
    ]);
    assert.doesNotMatch(JSON.stringify(firstRequest), /code_execution|url_context/);

    const secondBody = {
      messages: [
        { role: "system", content: "Follow the task carefully." },
        { role: "user", content: "Inspect the project." },
        { role: "assistant", content: "First answer" },
        { role: "user", content: "Continue." },
      ],
      tools,
      tool_choice: "auto",
      max_completion_tokens: 32_000,
    };
    const secondOffer = (await bestOffer(provider, {
      ...standardRequest,
      body: secondBody,
      requestedModel: antigravity.id,
    }, Date.now()))!;
    const secondResult = await provider.execute(secondOffer, secondBody, false, new AbortController().signal);
    assert.equal(secondResult.status, "success");
    if (secondResult.status === "success") secondResult.release();

    assert.equal(requests[1]?.previous_interaction_id, "agent-1");
    assert.deepEqual(requests[1]?.input, [
      { type: "user_input", content: [{ type: "text", text: "Continue." }] },
    ]);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("Antigravity still overrides default tools when the caller provides no functions", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  const antigravity = GOOGLE_MODELS.find((model) => model.id === "google/antigravity-preview-09-2026")!;
  const requests: Array<Record<string, unknown>> = [];

  try {
    const provider = new GoogleProvider(
      () => fakeClient((request) => {
        requests.push(request as Record<string, unknown>);
        return completedInteraction("agent-no-tools");
      }),
      [antigravity],
    );
    const body = { messages: [{ role: "user", content: "Hello" }] };
    const offer = (await bestOffer(provider, {
      ...standardRequest,
      body,
      requestedModel: antigravity.id,
    }, Date.now()))!;
    const result = await provider.execute(offer, body, false, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status === "success") result.release();

    assert.deepEqual(requests[0]?.tools, [{ type: "google_search" }]);
    assert.equal(requests[0]?.environment, undefined);
    assert.deepEqual(requests[0]?.agent_config, {
      type: "antigravity",
      model: "gemini-3.8-flash",
    });
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("Google preference breaks only otherwise-equivalent routing ties", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";

  const sharedQuota = {
    maxConcurrent: null,
    dailyWindow: { type: "rolling" as const },
    limits: {
      requestsPerMinute: 100,
      inputTokensPerMinute: 1_000,
      requestsPerDay: 1_000,
      minimumSpacingMs: 0,
    },
  };
  const lowPreference: GoogleModel = {
    id: "google/low-preference",
    upstreamModel: "low-preference",
    contextWindowTokens: 1_000,
    quota: sharedQuota,
    preference: 100,
  };
  const highPreference: GoogleModel = {
    id: "google/high-preference",
    upstreamModel: "high-preference",
    contextWindowTokens: 1_000,
    quota: sharedQuota,
    preference: 300,
  };
  const smallerCapacity: GoogleModel = {
    id: "google/smaller-capacity",
    upstreamModel: "smaller-capacity",
    contextWindowTokens: 500,
    quota: sharedQuota,
    preference: 1,
  };

  try {
    const provider = new GoogleProvider(
      () => fakeClient(() => completedInteraction("unused")),
      [lowPreference, highPreference],
    );
    const tied = await bestOffer(provider, {
      ...standardRequest,
      requestedModel: "auto",
    }, 1_000);
    assert.equal(tied?.modelId, highPreference.id);

    const capacityProvider = new GoogleProvider(
      () => fakeClient(() => completedInteraction("unused")),
      [highPreference, smallerCapacity],
    );
    const capacityFirst = await bestOffer(capacityProvider, {
      ...standardRequest,
      requestedModel: "auto",
    }, 1_000);
    assert.equal(capacityFirst?.modelId, smallerCapacity.id);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("generateContent fallback preserves Google content for continuation", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  const robotics = { ...GOOGLE_MODELS.find((model) => model.id === "google/gemini-robotics-er-2-preview")!, transport: "generate-content" as const };
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
    const firstOffer = (await bestOffer(provider, {
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
    const secondOffer = (await bestOffer(provider, {
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

test("generateContent fallback streaming is translated to OpenAI SSE", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  const robotics = { ...GOOGLE_MODELS.find((model) => model.id === "google/gemini-robotics-er-2-preview")!, transport: "generate-content" as const };

  async function* generatedStream() {
    yield {
      responseId: "robot-stream-1",
      createTime: "2026-10-01T08:00:00Z",
      candidates: [{ content: { role: "model", parts: [{ text: "Move" }] } }],
    };
    yield {
      responseId: "robot-stream-1",
      candidates: [{
        finishReason: "STOP",
        content: { role: "model", parts: [{ text: " done", thoughtSignature: "stream-sig" }] },
      }],
      usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 2, totalTokenCount: 12 },
    };
  }

  try {
    const client = fakeClient(() => generatedStream());
    const provider = new GoogleProvider(() => client, [robotics]);
    const body = {
      messages: [{ role: "user", content: "Move" }],
      stream_options: { include_usage: true },
    };
    const offer = (await bestOffer(provider, {
      ...standardRequest,
      body,
      requestedModel: robotics.id,
    }, Date.now()))!;
    const result = await provider.execute(offer, body, true, new AbortController().signal);
    assert.equal(result.status, "success");
    if (result.status !== "success") return;
    const text = await result.response.text();
    result.release();

    assert.match(text, /"content":"Move"/);
    assert.match(text, /"content":" done"/);
    assert.match(text, /"finish_reason":"stop"/);
    assert.match(text, /"prompt_tokens":10/);
    assert.match(text, /data: \[DONE\]/);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("Google offer filtering respects model-specific thinking levels", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  try {
    const models = GOOGLE_MODELS.filter((model) =>
      model.id === "google/gemini-3.8-flash" || model.id === "google/gemini-3.6-flash"
    );
    const provider = new GoogleProvider(() => fakeClient(() => completedInteraction("unused")), models);
    const body = {
      messages: [{ role: "user", content: "Answer quickly" }],
      reasoning_effort: "minimal",
    };

    const unsupported = await bestOffer(provider, {
      ...standardRequest,
      body,
      requestedModel: "google/gemini-3.8-flash",
    }, Date.now());
    assert.equal(unsupported, null);

    const supported = await bestOffer(provider, {
      ...standardRequest,
      body,
      requestedModel: "google/gemini-3.6-flash",
    }, Date.now());
    assert.equal(supported?.modelId, "google/gemini-3.6-flash");
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
    const offer = (await bestOffer(provider, { ...standardRequest, body }, Date.now()))!;
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

test("Google confirmed RPD overflow is blocked only until the next Pacific midnight", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  let createCalls = 0;
  const client = fakeClient(() => {
    createCalls += 1;
    if (createCalls === 1) return completedInteraction("first");
    throw Object.assign(new Error("quota exhausted"), {
      statusCode: 429,
      headers: new Headers(),
      body: '{"error":{"status":"RESOURCE_EXHAUSTED","quotaId":"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}',
    });
  });

  try {
    const provider = new GoogleProvider(() => client, [TEST_MODEL]);
    const firstOffer = (await bestOffer(provider, standardRequest, Date.now()))!;
    const firstResult = await provider.execute(firstOffer, standardRequest.body, false, new AbortController().signal);
    assert.equal(firstResult.status, "success");
    if (firstResult.status === "success") firstResult.release();

    const now = Date.now();
    const standard = (await bestOffer(provider, standardRequest, now))!;
    assert.ok(standard.availableAt > now + 60_000);

    const overflow = (await bestOffer(provider, { ...standardRequest, offerKind: "overflow" }, now))!;
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
    const expectedReset = calendarDayBounds(beforeFailure, "America/Los_Angeles").end;
    assert.equal(blockedUntil, expectedReset);
    assert.ok(blockedUntil! > beforeFailure);
    assert.ok(blockedUntil! - beforeFailure <= 24 * 60 * 60 * 1000);
    assert.equal(provider.status(blockedUntil! - 1).models[0]?.overflowBlockedUntil, blockedUntil);
    assert.equal(provider.status(blockedUntil!).models[0]?.overflowBlockedUntil, null);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});

test("a generic overflow 429 does not prove the daily overflow boundary", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  let createCalls = 0;
  const client = fakeClient(() => {
    createCalls += 1;
    if (createCalls === 1) return completedInteraction("first");
    throw Object.assign(new Error("Resource has been exhausted"), {
      statusCode: 429,
      headers: new Headers(),
      body: '{"error":{"status":"RESOURCE_EXHAUSTED","message":"Resource has been exhausted"}}',
    });
  });

  try {
    const provider = new GoogleProvider(() => client, [TEST_MODEL]);
    const firstOffer = (await bestOffer(provider, standardRequest, Date.now()))!;
    const firstResult = await provider.execute(firstOffer, standardRequest.body, false, new AbortController().signal);
    if (firstResult.status === "success") firstResult.release();

    const overflow = (await bestOffer(
      provider,
      { ...standardRequest, offerKind: "overflow" },
      Date.now(),
    ))!;
    const failed = await provider.execute(overflow, standardRequest.body, false, new AbortController().signal);
    assert.equal(failed.status, "retryable");
    if (failed.status === "retryable") {
      assert.equal(failed.scope, "model");
      assert.equal(failed.reason, "rate_limit");
    }
    assert.equal(provider.status().models[0]?.overflowBlockedUntil, null);
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
    const firstOffer = (await bestOffer(provider, standardRequest, Date.now()))!;
    const firstResult = await provider.execute(firstOffer, standardRequest.body, false, new AbortController().signal);
    if (firstResult.status === "success") firstResult.release();

    const overflow = (await bestOffer(provider, 
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

test("Google returns structured no-offer reasons for unsupported model requests", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = "test";
  try {
    const provider = new GoogleProvider(
      () => fakeClient(() => completedInteraction("unused")),
      [TEST_MODEL],
    );
    const result = await provider.getBestOffer({
      ...standardRequest,
      requestedModel: "google/not-owned",
    }, Date.now());

    assert.deepEqual(result, {
      status: "no_offer",
      providerId: "google",
      reason: "no_eligible_model",
    });
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});
