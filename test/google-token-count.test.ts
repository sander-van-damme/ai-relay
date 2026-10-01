import assert from "node:assert/strict";
import test from "node:test";
import {
  toGoogleCountInput,
  toGoogleInteractionRequest,
} from "../src/providers/google/token-count.ts";

test("Google token counting maps OpenAI chat messages and tools to native contents", () => {
  const input = toGoogleCountInput({
    messages: [
      { role: "system", content: "Be concise." },
      { role: "user", content: "What is the weather?" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "weather", arguments: "{\"city\":\"Ghent\"}" },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "{\"temperature\":18}" },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "weather",
          description: "Get weather",
          parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      },
    ],
  });

  assert.deepEqual(input.config?.systemInstruction, {
    parts: [{ text: "Be concise." }],
  });
  assert.deepEqual(input.config?.tools, [
    {
      functionDeclarations: [
        {
          name: "weather",
          description: "Get weather",
          parametersJsonSchema: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      ],
    },
  ]);
  assert.deepEqual(input.contents, [
    { role: "user", parts: [{ text: "What is the weather?" }] },
    {
      role: "model",
      parts: [{ functionCall: { id: "call_1", name: "weather", args: { city: "Ghent" } } }],
    },
    {
      role: "user",
      parts: [{ functionResponse: { id: "call_1", name: "weather", response: { temperature: 18 } } }],
    },
  ]);
});

test("Google Interactions maps chat options without the OpenAI-compatible transport", () => {
  const request = toGoogleInteractionRequest({
    messages: [
      { role: "developer", content: "Return structured weather data." },
      { role: "user", content: "Weather in Ghent" },
    ],
    max_completion_tokens: 200,
    stop: ["END"],
    reasoning_effort: "high",
    tool_choice: {
      type: "function",
      function: { name: "weather" },
    },
    tools: [
      {
        type: "function",
        function: {
          name: "weather",
          description: "Get weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      },
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "weather_response",
        strict: true,
        schema: {
          type: "object",
          properties: { temperature: { type: "number" } },
          required: ["temperature"],
        },
      },
    },
  }, "gemini-3.8-flash", false);

  assert.equal(request.model, "gemini-3.8-flash");
  assert.equal(request.stream, false);
  assert.equal(request.store, true);
  assert.equal(request.system_instruction, "Return structured weather data.");
  assert.deepEqual(request.input, [
    { type: "user_input", content: [{ type: "text", text: "Weather in Ghent" }] },
  ]);
  assert.deepEqual(request.tools, [
    {
      type: "function",
      name: "weather",
      description: "Get weather",
      parameters: { type: "object", properties: { city: { type: "string" } } },
    },
  ]);
  assert.deepEqual(request.generation_config, {
    max_output_tokens: 200,
    stop_sequences: ["END"],
    thinking_level: "high",
    tool_choice: { allowed_tools: { mode: "any", tools: ["weather"] } },
  });
  assert.deepEqual(request.response_format, {
    type: "text",
    mime_type: "application/json",
    schema: {
      type: "object",
      properties: { temperature: { type: "number" } },
      required: ["temperature"],
    },
  });
});

test("Google rejects sampling options absent from pinned Interactions v2.24", () => {
  assert.throws(
    () => toGoogleInteractionRequest({
      messages: [{ role: "user", content: "Hello" }],
      temperature: 0.2,
    }, "gemini-3.8-flash", false),
    /temperature and top_p are not supported/i,
  );
});

test("Google continuation sends only messages after the stored interaction", () => {
  const request = toGoogleInteractionRequest({
    messages: [
      { role: "user", content: "First" },
      { role: "assistant", content: "First answer" },
      { role: "user", content: "Second" },
    ],
  }, "gemini-3.8-flash", false, {
    previousInteractionId: "interaction-1",
    inputStartIndex: 2,
  });

  assert.equal(request.previous_interaction_id, "interaction-1");
  assert.deepEqual(request.input, [
    { type: "user_input", content: [{ type: "text", text: "Second" }] },
  ]);
});

test("Google token counting refuses content it cannot count authoritatively", () => {
  assert.throws(
    () => toGoogleCountInput({
      messages: [
        {
          role: "user",
          content: [{ type: "image_url", image_url: { url: "https://example.test/image.png" } }],
        },
      ],
    }),
    /cannot authoritatively count/i,
  );
});