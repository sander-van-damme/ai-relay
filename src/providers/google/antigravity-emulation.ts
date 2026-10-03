import type { ChatCompletionRequest } from "../../types.ts";

type JsonObject = Record<string, unknown>;

export const ANTIGRAVITY_MANUAL_RESPONSE_TAG = "manual_openai_chat_completion_response";
const OPEN_TAG = `<${ANTIGRAVITY_MANUAL_RESPONSE_TAG}>`;
const CLOSE_TAG = `</${ANTIGRAVITY_MANUAL_RESPONSE_TAG}>`;

const MANUAL_EMULATION_INSTRUCTIONS = `
You are manually acting as the response-producing side of an OpenAI-compatible Chat Completions endpoint.

This is intentionally a manual emulation task.

You are NOT an actual HTTP endpoint.
You are NOT being asked to build, implement, run, or automate a Chat Completions endpoint.
Do not create a server, proxy, adapter, parser, or program to handle these requests.

For this interaction, you will receive the raw Chat Completions request that an external caller sent to AI Relay.

Your job is to inspect that raw request and manually determine what a correct OpenAI-compatible Chat Completions response should be.

Treat the entire supplied JSON object as the OUTER Chat Completions request that you are manually responding to. It may contain system, developer, user, assistant, or tool messages; conversation history; tool definitions; previous tool calls and tool results; model or generation options; or other Chat Completions request fields. Some requests contain no tools at all.

IMPORTANT: OUTER REQUEST VS. YOUR OWN ENVIRONMENT

You have your own Antigravity environment and native Antigravity capabilities. These are separate from the environment represented by the outer Chat Completions request.

Anything described inside the raw request—including tools, files, paths, repositories, commands, credentials, applications, services, or execution environments—belongs to the external caller's context unless explicitly stated otherwise. Do not assume that resources mentioned in the outer request exist inside your Antigravity environment.

TOOLS IN THE OUTER REQUEST

If the raw request contains tool definitions, those tools belong to the external runtime using the Chat Completions API. They are NOT Antigravity tools. Do not attempt to execute those outer tools using your own environment.

If the correct Chat Completions response should request one of those tools, manually return the appropriate OpenAI-compatible tool call in your final response. The external runtime will decide how to execute that tool and may later send another Chat Completions request containing the result.

Use only tools actually defined in the supplied outer request. If the request contains no tools, construct a normal assistant response without tool calls.

YOUR NATIVE ANTIGRAVITY CAPABILITIES

Your own Antigravity tools and sandbox are private aids available while deciding what response to construct. You may use them when useful to reason, calculate, inspect temporary information, search public information, or verify facts. Using them is optional. In many cases you should not need to use them at all.

Your primary job is not to perform the caller's task directly inside your own environment. Your primary job is to determine what the emulated Chat Completions endpoint should return to the external caller.

Never confuse:
1. your own Antigravity environment and native tools; and
2. the external environment and optional tools represented inside the raw Chat Completions request.

OUTER INSTRUCTIONS

System, developer, user, assistant, and tool messages contained in the raw request form the conversation that the emulated assistant is responding to. Follow those instructions when determining the response you manually construct.

Instructions referring to tools, files, paths, repositories, applications, machines, or other environmental resources describe the external caller's context. They do not imply those resources exist inside your Antigravity sandbox.

MANUAL RESPONSE

When you have determined what the OpenAI-compatible endpoint should return, manually construct that response yourself for this individual request. Do not implement software to generate it.

Your final response must appear inside exactly one envelope:

<manual_openai_chat_completion_response>
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ]
}
</manual_openai_chat_completion_response>

The JSON inside this envelope is the response AI Relay intends to return to the external caller. Anything outside this envelope is internal Antigravity activity and will not be returned as the Chat Completions response.

Focus on the semantic response: choices, assistant messages, content, tool calls, and finish reasons. You do not need to provide transport metadata such as id, object, created, model, or usage. If you do provide those fields, that is fine: AI Relay will ignore and replace them with its own authoritative metadata.

The JSON must be syntactically valid and represent a valid OpenAI-compatible Chat Completions response. If a tool call is needed, use only a tool defined in the outer request, reproduce its name exactly, and provide its arguments as the JSON string required by Chat Completions. Return the tool call; do not fabricate the result of that outer tool call.

Do not emit the final envelope until you are ready to provide the response.
`.trim();

function object(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

export function antigravityManualPrompt(body: ChatCompletionRequest): string {
  return `${MANUAL_EMULATION_INSTRUCTIONS}

RAW CHAT COMPLETIONS REQUEST

<raw_openai_chat_completion_request>
${JSON.stringify(body, null, 2)}
</raw_openai_chat_completion_request>`;
}

export function antigravityRepairPrompt(validationError: string): string {
  return `The manual Chat Completions response from your previous turn could not be returned to the external caller because AI Relay validation failed:

${validationError}

Correct the response for the SAME outer Chat Completions request. Do not restart or reinterpret the task unless the validation error requires it. Do not execute any outer tool yourself.

Produce one corrected <manual_openai_chat_completion_response> envelope containing valid JSON. Anything outside that envelope will be ignored.`;
}

export function antigravityInteractionText(interaction: unknown): string {
  const value = object(interaction);
  if (!value) return "";
  if (typeof value.output_text === "string") return value.output_text;
  if (typeof value.outputText === "string") return value.outputText;

  const steps = Array.isArray(value.steps) ? value.steps : [];
  let text = "";
  for (const rawStep of steps) {
    const step = object(rawStep);
    if (step?.type !== "model_output" || !Array.isArray(step.content)) continue;
    for (const rawContent of step.content) {
      const content = object(rawContent);
      if (content?.type === "text" && typeof content.text === "string") text += content.text;
    }
  }
  return text;
}

export function parseManualChatCompletion(text: string): JsonObject {
  const start = text.lastIndexOf(OPEN_TAG);
  if (start < 0) throw new Error(`Missing ${OPEN_TAG} envelope.`);
  const contentStart = start + OPEN_TAG.length;
  const end = text.indexOf(CLOSE_TAG, contentStart);
  if (end < 0) throw new Error(`Missing ${CLOSE_TAG} closing tag.`);

  const payloadText = text.slice(contentStart, end).trim();
  if (!payloadText) throw new Error("Manual Chat Completions response envelope is empty.");

  let parsed: unknown;
  try {
    parsed = JSON.parse(payloadText) as unknown;
  } catch (error) {
    throw new Error(`Manual Chat Completions response is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  const payload = object(parsed);
  if (!payload) throw new Error("Manual Chat Completions response must be a JSON object.");
  return payload;
}

function outerToolNames(body: ChatCompletionRequest): Set<string> {
  const names = new Set<string>();
  if (Array.isArray(body.tools)) {
    for (const rawTool of body.tools) {
      const tool = object(rawTool);
      const fn = object(tool?.function);
      if (tool?.type === "function" && typeof fn?.name === "string" && fn.name) names.add(fn.name);
    }
  }
  if (Array.isArray(body.functions)) {
    for (const rawFunction of body.functions) {
      const fn = object(rawFunction);
      if (typeof fn?.name === "string" && fn.name) names.add(fn.name);
    }
  }
  return names;
}

const FINISH_REASONS = new Set(["stop", "length", "tool_calls", "content_filter", "function_call"]);

function validateToolCalls(rawToolCalls: unknown, allowedTools: ReadonlySet<string>): JsonObject[] {
  if (!Array.isArray(rawToolCalls) || rawToolCalls.length === 0) {
    throw new Error("assistant.message.tool_calls must be a non-empty array when present.");
  }
  if (allowedTools.size === 0) {
    throw new Error("Response requested an outer tool, but the outer request defined no tools.");
  }

  return rawToolCalls.map((rawCall, index) => {
    const call = object(rawCall);
    const fn = object(call?.function);
    if (!call || call.type !== "function") {
      throw new Error(`choices[0].message.tool_calls[${index}] must have type "function".`);
    }
    if (typeof call.id !== "string" || !call.id) {
      throw new Error(`choices[0].message.tool_calls[${index}].id must be a non-empty string.`);
    }
    if (!fn || typeof fn.name !== "string" || !fn.name) {
      throw new Error(`choices[0].message.tool_calls[${index}].function.name must be a non-empty string.`);
    }
    if (!allowedTools.has(fn.name)) {
      throw new Error(`Tool "${fn.name}" was not defined by the outer Chat Completions request.`);
    }
    if (typeof fn.arguments !== "string") {
      throw new Error(`Tool "${fn.name}" arguments must be a JSON string.`);
    }
    try {
      JSON.parse(fn.arguments);
    } catch (error) {
      throw new Error(`Tool "${fn.name}" arguments are not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
    return {
      id: call.id,
      type: "function",
      function: {
        name: fn.name,
        arguments: fn.arguments,
      },
    };
  });
}

export interface ManualChatCompletionSemantic {
  choices: JsonObject[];
}

export function validateManualChatCompletion(
  payload: JsonObject,
  body: ChatCompletionRequest,
): ManualChatCompletionSemantic {
  if (!Array.isArray(payload.choices) || payload.choices.length === 0) {
    throw new Error("Manual Chat Completions response must contain a non-empty choices array.");
  }

  const allowedTools = outerToolNames(body);
  const choices = payload.choices.map((rawChoice, choiceIndex) => {
    const choice = object(rawChoice);
    const message = object(choice?.message);
    if (!choice || !message) throw new Error(`choices[${choiceIndex}].message must be an object.`);
    if (message.role !== "assistant") throw new Error(`choices[${choiceIndex}].message.role must be "assistant".`);

    const hasToolCalls = message.tool_calls !== undefined;
    const toolCalls = hasToolCalls ? validateToolCalls(message.tool_calls, allowedTools) : undefined;
    const content = message.content;
    if (content !== undefined && content !== null && typeof content !== "string") {
      throw new Error(`choices[${choiceIndex}].message.content must be a string or null.`);
    }
    if ((content === undefined || content === null) && !toolCalls) {
      throw new Error(`choices[${choiceIndex}].message must contain content or tool_calls.`);
    }

    const finishReason = choice.finish_reason;
    if (typeof finishReason !== "string" || !FINISH_REASONS.has(finishReason)) {
      throw new Error(`choices[${choiceIndex}].finish_reason is not a supported Chat Completions finish reason.`);
    }
    if (toolCalls && finishReason !== "tool_calls" && finishReason !== "function_call") {
      throw new Error(`choices[${choiceIndex}] contains tool_calls but finish_reason is "${finishReason}".`);
    }
    if (!toolCalls && (finishReason === "tool_calls" || finishReason === "function_call")) {
      throw new Error(`choices[${choiceIndex}] uses finish_reason "${finishReason}" without a tool call.`);
    }

    const normalizedMessage: JsonObject = {
      role: "assistant",
      content: content ?? (toolCalls ? null : ""),
    };
    if (toolCalls) normalizedMessage.tool_calls = toolCalls;

    return {
      index: Number.isInteger(choice.index) ? choice.index : choiceIndex,
      message: normalizedMessage,
      finish_reason: finishReason,
      logprobs: null,
    };
  });

  return { choices };
}

export function buildRelayChatCompletion(
  semantic: ManualChatCompletionSemantic,
  metadata: {
    id: string;
    model: string;
    created: number;
    usage: JsonObject;
  },
): JsonObject {
  return {
    id: metadata.id,
    object: "chat.completion",
    created: metadata.created,
    model: metadata.model,
    choices: semantic.choices,
    usage: metadata.usage,
  };
}

export function chatCompletionToSse(response: JsonObject, includeUsage: boolean): Response {
  const id = typeof response.id === "string" ? response.id : "chatcmpl-google";
  const created = typeof response.created === "number" ? response.created : Math.floor(Date.now() / 1000);
  const model = typeof response.model === "string" ? response.model : "google";
  const choices = Array.isArray(response.choices) ? response.choices : [];

  const firstChoices = choices.map((rawChoice, fallbackIndex) => {
    const choice = object(rawChoice) ?? {};
    const message = object(choice.message) ?? {};
    const delta: JsonObject = { role: "assistant" };
    if (message.content !== undefined && message.content !== null) delta.content = message.content;
    if (Array.isArray(message.tool_calls)) {
      delta.tool_calls = message.tool_calls.map((rawCall, toolIndex) => {
        const call = object(rawCall) ?? {};
        return { index: toolIndex, ...call };
      });
    }
    return {
      index: Number.isInteger(choice.index) ? choice.index : fallbackIndex,
      delta,
      finish_reason: null,
      logprobs: null,
    };
  });

  const finishChoices = choices.map((rawChoice, fallbackIndex) => {
    const choice = object(rawChoice) ?? {};
    return {
      index: Number.isInteger(choice.index) ? choice.index : fallbackIndex,
      delta: {},
      finish_reason: typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
      logprobs: null,
    };
  });

  const chunks = [
    { id, object: "chat.completion.chunk", created, model, choices: firstChoices },
    { id, object: "chat.completion.chunk", created, model, choices: finishChoices },
  ];
  if (includeUsage) {
    chunks.push({
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [],
      usage: object(response.usage) ?? {},
    } as any);
  }

  const bodyText = chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n";
  return new Response(bodyText, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
    },
  });
}
