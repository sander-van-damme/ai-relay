import {
  GoogleGenAI,
  type Content,
  type CountTokensConfig,
  type FunctionDeclaration,
  type Part,
  type Tool,
} from "@google/genai";
import type { ChatCompletionRequest } from "../../types.ts";
import type { InputTokenCounter } from "../shared/openai-compatible.ts";

type JsonObject = Record<string, unknown>;

interface GoogleCountInput {
  contents: Content[];
  config?: CountTokensConfig;
}

let client: GoogleGenAI | undefined;
let clientApiKey: string | undefined;

function object(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function textContent(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) throw new Error("Google token counting only supports string or text-part message content.");

  let text = "";
  for (const valuePart of value) {
    const part = object(valuePart);
    if (!part || (part.type !== "text" && part.type !== "input_text") || typeof part.text !== "string") {
      throw new Error("Google token counting cannot authoritatively count this message content type.");
    }
    text += part.text;
  }
  return text;
}

function parseJsonObject(value: string, label: string): JsonObject {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value) as unknown;
  } catch {
    throw new Error(`${label} must contain valid JSON.`);
  }
  const result = object(parsed);
  if (!result) throw new Error(`${label} must contain a JSON object.`);
  return result;
}

function functionDeclarations(value: unknown): Tool[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) throw new Error("tools must be an array.");

  const declarations: FunctionDeclaration[] = value.map((rawTool) => {
    const tool = object(rawTool);
    if (!tool || tool.type !== "function") {
      throw new Error("Google token counting currently supports OpenAI function tools only.");
    }
    const fn = object(tool.function);
    if (!fn || typeof fn.name !== "string") throw new Error("Function tool is missing a name.");

    const declaration: FunctionDeclaration = { name: fn.name };
    if (typeof fn.description === "string") declaration.description = fn.description;
    if (fn.parameters !== undefined) declaration.parametersJsonSchema = fn.parameters;
    return declaration;
  });

  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : undefined;
}

function functionResponseValue(value: string): JsonObject {
  try {
    const parsed = JSON.parse(value) as unknown;
    return object(parsed) ?? { result: parsed };
  } catch {
    return { result: value };
  }
}

export function toGoogleCountInput(body: ChatCompletionRequest): GoogleCountInput {
  if (!Array.isArray(body.messages)) throw new Error("messages must be an array.");

  const contents: Content[] = [];
  const systemParts: Part[] = [];
  const toolCallNames = new Map<string, string>();
  let sawConversationMessage = false;

  for (const rawMessage of body.messages) {
    const message = object(rawMessage);
    if (!message || typeof message.role !== "string") throw new Error("Invalid chat message.");

    if (message.role === "system" || message.role === "developer") {
      if (sawConversationMessage) {
        throw new Error("Google token counting requires system/developer messages before conversation messages.");
      }
      systemParts.push({ text: textContent(message.content) });
      continue;
    }

    sawConversationMessage = true;

    if (message.role === "user") {
      if (message.name !== undefined) throw new Error("Named user messages are not supported for exact Google token counting.");
      contents.push({ role: "user", parts: [{ text: textContent(message.content) }] });
      continue;
    }

    if (message.role === "assistant") {
      if (message.name !== undefined) throw new Error("Named assistant messages are not supported for exact Google token counting.");
      const parts: Part[] = [];
      const text = textContent(message.content);
      if (text) parts.push({ text });

      if (message.tool_calls !== undefined) {
        if (!Array.isArray(message.tool_calls)) throw new Error("assistant.tool_calls must be an array.");
        for (const rawCall of message.tool_calls) {
          const call = object(rawCall);
          const fn = call ? object(call.function) : null;
          if (!call || call.type !== "function" || !fn || typeof fn.name !== "string") {
            throw new Error("Only OpenAI function tool calls are supported for exact Google token counting.");
          }
          const args = typeof fn.arguments === "string"
            ? parseJsonObject(fn.arguments, `Arguments for tool ${fn.name}`)
            : object(fn.arguments);
          if (!args) throw new Error(`Arguments for tool ${fn.name} must be a JSON object.`);
          if (typeof call.id === "string") toolCallNames.set(call.id, fn.name);
          parts.push({ functionCall: { name: fn.name, args } });
        }
      }

      contents.push({ role: "model", parts });
      continue;
    }

    if (message.role === "tool") {
      if (typeof message.tool_call_id !== "string") throw new Error("Tool message is missing tool_call_id.");
      const name = toolCallNames.get(message.tool_call_id);
      if (!name) throw new Error(`Cannot resolve tool call name for ${message.tool_call_id}.`);
      const response = functionResponseValue(textContent(message.content));
      contents.push({ role: "user", parts: [{ functionResponse: { name, response } }] });
      continue;
    }

    throw new Error(`Unsupported OpenAI message role for exact Google token counting: ${message.role}`);
  }

  if (body.response_format !== undefined) {
    const responseFormat = object(body.response_format);
    if (!responseFormat || (responseFormat.type !== undefined && responseFormat.type !== "text")) {
      throw new Error("Structured response formats are not yet supported for exact Google token counting.");
    }
  }

  const tools = functionDeclarations(body.tools);
  const config: CountTokensConfig = {};
  if (systemParts.length > 0) config.systemInstruction = { parts: systemParts };
  if (tools) config.tools = tools;

  return {
    contents,
    config: Object.keys(config).length > 0 ? config : undefined,
  };
}

function googleClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) throw new Error("GEMINI_API_KEY is not configured.");
  if (!client || clientApiKey !== apiKey) {
    client = new GoogleGenAI({ apiKey });
    clientApiKey = apiKey;
  }
  return client;
}

export const countGoogleInputTokens: InputTokenCounter = async (body, model) => {
  const input = toGoogleCountInput(body);
  const config: CountTokensConfig = {
    ...input.config,
    abortSignal: AbortSignal.timeout(10_000),
  };
  const response = await googleClient().models.countTokens({
    model: model.upstreamModel,
    contents: input.contents,
    config,
  });
  if (!Number.isSafeInteger(response.totalTokens) || (response.totalTokens ?? -1) < 0) {
    throw new Error(`Google did not return a valid token count for ${model.upstreamModel}.`);
  }
  return response.totalTokens!;
};
