import assert from "node:assert/strict";
import test from "node:test";
import { toGoogleCountInput } from "../src/providers/google/token-count.ts";

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
      parts: [{ functionCall: { name: "weather", args: { city: "Ghent" } } }],
    },
    {
      role: "user",
      parts: [{ functionResponse: { name: "weather", response: { temperature: 18 } } }],
    },
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
