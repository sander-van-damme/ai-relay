import assert from "node:assert/strict";
import test from "node:test";
import {
  GOOGLE_EXTERNAL_TOOL_THOUGHT_SIGNATURE,
  toAntigravityBootstrapBody,
  toGoogleCountInput,
  toGoogleDeveloperCountTokensRequest,
  toGoogleGenerateContentRequest,
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

test("Antigravity bootstrap flattens historical tool traces into plain text", () => {
  const body = toAntigravityBootstrapBody({
    messages: [
      { role: "system", content: "Work carefully." },
      { role: "user", content: "Inspect the project." },
      {
        role: "assistant",
        content: "I will read the file.",
        tool_calls: [{
          id: "call_1",
          type: "function",
          function: { name: "read_file", arguments: "{\"path\":\"a.ts\"}" },
        }],
      },
      { role: "tool", tool_call_id: "call_1", content: "export const x = 1;" },
      { role: "user", content: "Now fix it." },
    ],
    tools: [{
      type: "function",
      function: { name: "read_file", parameters: { type: "object" } },
    }],
  });

  const messages = body.messages as Array<Record<string, unknown>>;
  assert.deepEqual(messages.slice(0, 1), [{ role: "system", content: "Work carefully." }]);
  assert.equal(messages.length, 2);
  const bootstrap = messages[1]!;
  assert.equal(bootstrap.role, "user");
  assert.match(String(bootstrap.content), /historical conversation context/i);
  assert.match(String(bootstrap.content), /ASSISTANT TOOL CALL \[call_1\] read_file/);
  assert.match(String(bootstrap.content), /TOOL RESULT \[call_1\]/);
  assert.match(String(bootstrap.content), /Now fix it\./);
});

test("GenerateContent replay signs externally reconstructed function calls", () => {
  const request = toGoogleGenerateContentRequest({
    messages: [
      { role: "user", content: "Inspect the project." },
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call_1",
          type: "function",
          function: { name: "read_file", arguments: "{\"path\":\"a.ts\"}" },
        }],
      },
      { role: "tool", tool_call_id: "call_1", content: "file contents" },
      { role: "user", content: "Continue." },
    ],
    tools: [{
      type: "function",
      function: { name: "read_file", parameters: { type: "object" } },
    }],
  }, "gemini-3.8-flash", undefined, true);

  const modelContent = request.contents[1] as Record<string, unknown>;
  const parts = modelContent.parts as Array<Record<string, unknown>>;
  assert.equal(parts[0]?.thoughtSignature, GOOGLE_EXTERNAL_TOOL_THOUGHT_SIGNATURE);
  assert.match(JSON.stringify(parts[0]?.functionCall), /read_file/);
});

test("Google Developer API token counting wraps the full generation request", () => {
  const input = toGoogleCountInput({
    messages: [
      { role: "system", content: "Be concise." },
      { role: "user", content: "What is the weather?" },
    ],
    tools: [{
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
    }],
    response_format: { type: "json_object" },
  });

  assert.deepEqual(toGoogleDeveloperCountTokensRequest(input, "gemini-3.8-flash"), {
    generateContentRequest: {
      model: "models/gemini-3.8-flash",
      contents: [{ role: "user", parts: [{ text: "What is the weather?" }] }],
      systemInstruction: { parts: [{ text: "Be concise." }] },
      tools: [{
        functionDeclarations: [{
          name: "weather",
          description: "Get weather",
          parametersJsonSchema: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        }],
      }],
      generationConfig: {
        responseMimeType: "application/json",
      },
    },
  });
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

test("Google generateContent mapping supports Robotics sampling and native config", () => {
  const request = toGoogleGenerateContentRequest({
    messages: [
      { role: "system", content: "Be precise." },
      { role: "user", content: "Plan the motion." },
    ],
    temperature: 0.2,
    top_p: 0.8,
    max_completion_tokens: 120,
    reasoning_effort: "high",
    tool_choice: "required",
    tools: [
      {
        type: "function",
        function: {
          name: "move_robot",
          parameters: { type: "object", properties: { x: { type: "number" } } },
        },
      },
    ],
  }, "gemini-robotics-er-2-preview");

  assert.equal(request.model, "gemini-robotics-er-2-preview");
  assert.deepEqual(request.contents, [
    { role: "user", parts: [{ text: "Plan the motion." }] },
  ]);
  assert.equal(request.config?.temperature, 0.2);
  assert.equal(request.config?.topP, 0.8);
  assert.equal(request.config?.maxOutputTokens, 120);
  assert.deepEqual(request.config?.thinkingConfig, { thinkingLevel: "HIGH" });
  assert.deepEqual(request.config?.toolConfig, {
    functionCallingConfig: { mode: "ANY" },
  });
  assert.deepEqual(request.config?.systemInstruction, { parts: [{ text: "Be precise." }] });
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

test("Google Interactions continuation sends function results as text content blocks", () => {
  const request = toGoogleInteractionRequest({
    messages: [
      { role: "user", content: "Inspect a.ts" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "call_1",
          type: "function",
          function: { name: "read_file", arguments: "{\"path\":\"a.ts\"}" },
        }],
      },
      { role: "tool", tool_call_id: "call_1", content: "file contents" },
    ],
  }, "gemini-3.8-flash", false, {
    previousInteractionId: "interaction-with-call",
    inputStartIndex: 2,
  });

  assert.equal(request.previous_interaction_id, "interaction-with-call");
  assert.deepEqual(request.input, [{
    type: "function_result",
    name: "read_file",
    call_id: "call_1",
    result: [{ type: "text", text: "file contents" }],
  }]);
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