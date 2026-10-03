import {
  FunctionCallingConfigMode,
  ThinkingLevel,
  type Content,
  type CountTokensConfig,
  type FunctionDeclaration,
  type GenerateContentConfig,
  type Part,
  type Tool,
} from "@google/genai";
import type { ChatCompletionRequest } from "../../types.ts";

type JsonObject = Record<string, unknown>;

export interface GoogleCountInput {
  contents: Content[];
  config?: CountTokensConfig;
}

export interface GoogleDeveloperCountTokensRequest {
  generateContentRequest: {
    model: string;
    contents: Content[];
    systemInstruction?: CountTokensConfig["systemInstruction"];
    tools?: CountTokensConfig["tools"];
    generationConfig?: CountTokensConfig["generationConfig"];
  };
}

export type GoogleInteractionStep =
  | { type: "user_input"; content: Array<{ type: "text"; text: string }> }
  | { type: "model_output"; content: Array<{ type: "text"; text: string }> }
  | { type: "function_call"; id: string; name: string; arguments: JsonObject }
  | {
      type: "function_result";
      call_id: string;
      name?: string;
      result: Array<{ type: "text"; text: string }>;
    };

export interface GoogleGenerateContentRequest {
  model: string;
  contents: Content[];
  config?: GenerateContentConfig;
}

export interface GoogleInteractionRequest {
  model: string;
  input: GoogleInteractionStep[];
  store: true;
  stream: boolean;
  previous_interaction_id?: string;
  system_instruction?: string;
  tools?: Array<{
    type: "function";
    name: string;
    description?: string;
    parameters?: unknown;
  }>;
  generation_config?: {
    max_output_tokens?: number;
    seed?: number;
    stop_sequences?: string[];
    thinking_level?: "minimal" | "low" | "medium" | "high";
    tool_choice?: "auto" | "any" | "none" | "validated" | {
      allowed_tools: { mode: "auto" | "any" | "none" | "validated"; tools?: string[] };
    };
  };
  response_format?: {
    type: "text";
    mime_type: "text/plain" | "application/json";
    schema?: JsonObject;
  };
}

interface NormalizedGoogleRequest {
  countInput: GoogleCountInput;
  interaction: Omit<GoogleInteractionRequest, "model" | "stream">;
}

function object(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function textContent(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) throw new Error("Google only supports string or text-part message content in the relay chat API.");

  let text = "";
  for (const valuePart of value) {
    const part = object(valuePart);
    if (!part || (part.type !== "text" && part.type !== "input_text") || typeof part.text !== "string") {
      throw new Error("Google cannot authoritatively count this message content type.");
    }
    text += part.text;
  }
  return text;
}

export const GOOGLE_EXTERNAL_TOOL_THOUGHT_SIGNATURE = "skip_thought_signature_validator";

export function hasGoogleToolHistory(body: ChatCompletionRequest): boolean {
  if (!Array.isArray(body.messages)) return false;
  return body.messages.some((rawMessage) => {
    const message = object(rawMessage);
    return message?.role === "tool"
      || (message?.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0);
  });
}

export function hasAssistantToolCallsFrom(body: ChatCompletionRequest, startIndex: number): boolean {
  if (!Array.isArray(body.messages)) return false;
  return body.messages.slice(startIndex).some((rawMessage) => {
    const message = object(rawMessage);
    return message?.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
  });
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

function responseFormat(value: unknown): {
  interaction?: GoogleInteractionRequest["response_format"];
  mimeType?: string;
  schema?: JsonObject;
} {
  if (value === undefined) return {};
  const format = object(value);
  if (!format || typeof format.type !== "string") throw new Error("response_format must contain a type.");

  if (format.type === "text") {
    return {
      interaction: { type: "text", mime_type: "text/plain" },
      mimeType: "text/plain",
    };
  }
  if (format.type === "json_object") {
    return {
      interaction: { type: "text", mime_type: "application/json" },
      mimeType: "application/json",
    };
  }
  if (format.type === "json_schema") {
    const jsonSchema = object(format.json_schema);
    const schema = jsonSchema ? object(jsonSchema.schema) : null;
    if (!schema) throw new Error("response_format.json_schema.schema must be a JSON object.");
    return {
      interaction: { type: "text", mime_type: "application/json", schema },
      mimeType: "application/json",
      schema,
    };
  }
  throw new Error(`Unsupported Google response_format type: ${format.type}`);
}

function tools(value: unknown): {
  countTools?: Tool[];
  interactionTools?: GoogleInteractionRequest["tools"];
} {
  if (value === undefined) return {};
  if (!Array.isArray(value)) throw new Error("tools must be an array.");

  const countDeclarations: FunctionDeclaration[] = [];
  const interactionTools: NonNullable<GoogleInteractionRequest["tools"]> = [];
  for (const rawTool of value) {
    const tool = object(rawTool);
    if (!tool || tool.type !== "function") {
      throw new Error("Google currently supports OpenAI function tools only.");
    }
    const fn = object(tool.function);
    if (!fn || typeof fn.name !== "string" || !fn.name) throw new Error("Function tool is missing a name.");

    const declaration: FunctionDeclaration = { name: fn.name };
    const interactionTool: NonNullable<GoogleInteractionRequest["tools"]>[number] = {
      type: "function",
      name: fn.name,
    };
    if (typeof fn.description === "string") {
      declaration.description = fn.description;
      interactionTool.description = fn.description;
    }
    if (fn.parameters !== undefined) {
      declaration.parametersJsonSchema = fn.parameters;
      interactionTool.parameters = fn.parameters;
    }
    countDeclarations.push(declaration);
    interactionTools.push(interactionTool);
  }

  return {
    countTools: countDeclarations.length > 0 ? [{ functionDeclarations: countDeclarations }] : undefined,
    interactionTools: interactionTools.length > 0 ? interactionTools : undefined,
  };
}

function numberField(body: ChatCompletionRequest, field: string): number | undefined {
  const value = body[field];
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${field} must be a finite number.`);
  return value;
}

function integerField(body: ChatCompletionRequest, field: string): number | undefined {
  const value = numberField(body, field);
  if (value !== undefined && !Number.isSafeInteger(value)) throw new Error(`${field} must be an integer.`);
  return value;
}

function generationConfig(body: ChatCompletionRequest): GoogleInteractionRequest["generation_config"] {
  const config: NonNullable<GoogleInteractionRequest["generation_config"]> = {};
  const maxCompletionTokens = integerField(body, "max_completion_tokens");
  const maxTokens = integerField(body, "max_tokens");
  const maxOutputTokens = maxCompletionTokens ?? maxTokens;
  if (maxOutputTokens !== undefined) config.max_output_tokens = maxOutputTokens;

  const seed = integerField(body, "seed");
  if (seed !== undefined) config.seed = seed;

  if (body.stop !== undefined && body.stop !== null) {
    if (typeof body.stop === "string") config.stop_sequences = [body.stop];
    else if (Array.isArray(body.stop) && body.stop.every((item) => typeof item === "string")) {
      config.stop_sequences = body.stop as string[];
    } else {
      throw new Error("stop must be a string or an array of strings.");
    }
  }

  if (body.reasoning_effort !== undefined) {
    const effort = body.reasoning_effort;
    if (effort !== "minimal" && effort !== "low" && effort !== "medium" && effort !== "high") {
      throw new Error("reasoning_effort must be one of minimal, low, medium, or high for Google models.");
    }
    config.thinking_level = effort;
  }

  if (body.tool_choice !== undefined) {
    const choice = body.tool_choice;
    if (choice === "auto") config.tool_choice = "auto";
    else if (choice === "none") config.tool_choice = "none";
    else if (choice === "required") config.tool_choice = "any";
    else {
      const choiceObject = object(choice);
      const fn = choiceObject?.type === "function" ? object(choiceObject.function) : null;
      if (!fn || typeof fn.name !== "string" || !fn.name) throw new Error("Unsupported tool_choice for Google models.");
      config.tool_choice = { allowed_tools: { mode: "any", tools: [fn.name] } };
    }
  }

  const n = integerField(body, "n");
  if (n !== undefined && n !== 1) throw new Error("Google Interactions currently supports exactly one chat completion choice.");
  if (body.frequency_penalty !== undefined || body.presence_penalty !== undefined) {
    throw new Error("frequency_penalty and presence_penalty are not supported by the Google Interactions API.");
  }
  if (body.logprobs !== undefined || body.top_logprobs !== undefined) {
    throw new Error("logprobs are not supported by the Google Interactions API.");
  }
  if (body.parallel_tool_calls === false) {
    throw new Error("parallel_tool_calls=false is not supported by the Google Interactions API.");
  }
  if (body.modalities !== undefined) {
    if (!Array.isArray(body.modalities) || body.modalities.length !== 1 || body.modalities[0] !== "text") {
      throw new Error("This Google relay provider only supports text output.");
    }
  }
  if (body.audio !== undefined) throw new Error("This Google relay provider does not support audio output.");

  return Object.keys(config).length > 0 ? config : undefined;
}

function normalize(body: ChatCompletionRequest, inputStartIndex = 0): NormalizedGoogleRequest {
  if (!Array.isArray(body.messages)) throw new Error("messages must be an array.");

  const contents: Content[] = [];
  const steps: GoogleInteractionStep[] = [];
  const systemTexts: string[] = [];
  const toolCallNames = new Map<string, string>();
  let sawConversationMessage = false;

  for (const [messageIndex, rawMessage] of body.messages.entries()) {
    const message = object(rawMessage);
    if (!message || typeof message.role !== "string") throw new Error("Invalid chat message.");

    if (message.role === "system" || message.role === "developer") {
      if (sawConversationMessage) {
        throw new Error("Google requires system/developer messages before conversation messages.");
      }
      systemTexts.push(textContent(message.content));
      continue;
    }

    sawConversationMessage = true;
    if (message.role === "user") {
      if (message.name !== undefined) throw new Error("Named user messages are not supported by the Google relay provider.");
      const text = textContent(message.content);
      contents.push({ role: "user", parts: [{ text }] });
      if (messageIndex >= inputStartIndex) {
        steps.push({ type: "user_input", content: [{ type: "text", text }] });
      }
      continue;
    }

    if (message.role === "assistant") {
      if (message.name !== undefined) throw new Error("Named assistant messages are not supported by the Google relay provider.");
      const parts: Part[] = [];
      const text = textContent(message.content);
      if (text || message.tool_calls === undefined) {
        parts.push({ text });
        if (messageIndex >= inputStartIndex) {
          steps.push({ type: "model_output", content: [{ type: "text", text }] });
        }
      }

      if (message.tool_calls !== undefined) {
        if (!Array.isArray(message.tool_calls)) throw new Error("assistant.tool_calls must be an array.");
        for (const rawCall of message.tool_calls) {
          const call = object(rawCall);
          const fn = call ? object(call.function) : null;
          if (!call || call.type !== "function" || typeof call.id !== "string" || !fn || typeof fn.name !== "string") {
            throw new Error("Only OpenAI function tool calls with ids are supported by the Google relay provider.");
          }
          const args = typeof fn.arguments === "string"
            ? parseJsonObject(fn.arguments, `Arguments for tool ${fn.name}`)
            : object(fn.arguments);
          if (!args) throw new Error(`Arguments for tool ${fn.name} must be a JSON object.`);
          toolCallNames.set(call.id, fn.name);
          parts.push({ functionCall: { id: call.id, name: fn.name, args } });
          if (messageIndex >= inputStartIndex) {
            steps.push({ type: "function_call", id: call.id, name: fn.name, arguments: args });
          }
        }
      }
      contents.push({ role: "model", parts });
      continue;
    }

    if (message.role === "tool") {
      if (typeof message.tool_call_id !== "string") throw new Error("Tool message is missing tool_call_id.");
      const name = toolCallNames.get(message.tool_call_id);
      if (!name) throw new Error(`Cannot resolve tool call name for ${message.tool_call_id}.`);
      const result = textContent(message.content);
      let response: JsonObject;
      try {
        const parsed = JSON.parse(result) as unknown;
        response = object(parsed) ?? { result: parsed };
      } catch {
        response = { result };
      }
      contents.push({ role: "user", parts: [{ functionResponse: { id: message.tool_call_id, name, response } }] });
      if (messageIndex >= inputStartIndex) {
        steps.push({
          type: "function_result",
          call_id: message.tool_call_id,
          name,
          result: [{ type: "text", text: result }],
        });
      }
      continue;
    }

    throw new Error(`Unsupported OpenAI message role for Google: ${message.role}`);
  }

  const systemInstruction = systemTexts.length > 0 ? systemTexts.join("\n\n") : undefined;
  const mappedTools = tools(body.tools);
  const mappedResponseFormat = responseFormat(body.response_format);
  const mappedGenerationConfig = generationConfig(body);

  const countConfig: CountTokensConfig = {};
  if (systemInstruction !== undefined) countConfig.systemInstruction = { parts: [{ text: systemInstruction }] };
  if (mappedTools.countTools) countConfig.tools = mappedTools.countTools;
  if (mappedResponseFormat.mimeType) {
    countConfig.generationConfig = {
      responseMimeType: mappedResponseFormat.mimeType,
      ...(mappedResponseFormat.schema ? { responseJsonSchema: mappedResponseFormat.schema } : {}),
    };
  }

  return {
    countInput: {
      contents,
      config: Object.keys(countConfig).length > 0 ? countConfig : undefined,
    },
    interaction: {
      input: steps,
      store: true,
      ...(systemInstruction !== undefined ? { system_instruction: systemInstruction } : {}),
      ...(mappedTools.interactionTools ? { tools: mappedTools.interactionTools } : {}),
      ...(mappedGenerationConfig ? { generation_config: mappedGenerationConfig } : {}),
      ...(mappedResponseFormat.interaction ? { response_format: mappedResponseFormat.interaction } : {}),
    },
  };
}

export function toGoogleCountInput(body: ChatCompletionRequest): GoogleCountInput {
  return normalize(body).countInput;
}

export function toGoogleDeveloperCountTokensRequest(
  input: GoogleCountInput,
  model: string,
): GoogleDeveloperCountTokensRequest {
  const resourceModel = model.startsWith("models/") ? model : `models/${model}`;
  const config = input.config;
  return {
    generateContentRequest: {
      model: resourceModel,
      contents: input.contents,
      ...(config?.systemInstruction !== undefined
        ? { systemInstruction: config.systemInstruction }
        : {}),
      ...(config?.tools !== undefined ? { tools: config.tools } : {}),
      ...(config?.generationConfig !== undefined
        ? { generationConfig: config.generationConfig }
        : {}),
    },
  };
}

function contentIndexBeforeMessage(body: ChatCompletionRequest, messageIndex: number): number {
  if (!Array.isArray(body.messages)) return 0;
  let index = 0;
  for (const rawMessage of body.messages.slice(0, messageIndex)) {
    const message = object(rawMessage);
    if (message?.role !== "system" && message?.role !== "developer") index += 1;
  }
  return index;
}

function generateContentToolConfig(body: ChatCompletionRequest): GenerateContentConfig["toolConfig"] {
  if (body.tool_choice === undefined) return undefined;
  const choice = body.tool_choice;
  if (choice === "auto") return { functionCallingConfig: { mode: FunctionCallingConfigMode.AUTO } };
  if (choice === "none") return { functionCallingConfig: { mode: FunctionCallingConfigMode.NONE } };
  if (choice === "required") return { functionCallingConfig: { mode: FunctionCallingConfigMode.ANY } };

  const choiceObject = object(choice);
  const fn = choiceObject?.type === "function" ? object(choiceObject.function) : null;
  if (!fn || typeof fn.name !== "string" || !fn.name) throw new Error("Unsupported tool_choice for Google models.");
  return {
    functionCallingConfig: {
      mode: FunctionCallingConfigMode.ANY,
      allowedFunctionNames: [fn.name],
    },
  };
}

function replaySafeExternalToolContents(contents: Content[]): Content[] {
  return contents.map((content) => {
    if (content.role !== "model" || !Array.isArray(content.parts)) return content;
    let signedFunctionCall = false;
    const parts = content.parts.map((rawPart) => {
      const part = object(rawPart);
      if (!part || !object(part.functionCall) || signedFunctionCall) return rawPart;
      signedFunctionCall = true;
      if (part.thoughtSignature !== undefined) return rawPart;
      return {
        ...part,
        thoughtSignature: GOOGLE_EXTERNAL_TOOL_THOUGHT_SIGNATURE,
      } as Part;
    });
    return { ...content, parts } as Content;
  });
}

export function toGoogleGenerateContentRequest(
  body: ChatCompletionRequest,
  model: string,
  continuation?: { contents: Content[]; inputStartIndex: number },
  replayExternalToolHistory = false,
): GoogleGenerateContentRequest {
  const normalized = normalize(body);
  const interactionConfig = normalized.interaction.generation_config;
  const config: GenerateContentConfig = {};
  const countConfig = normalized.countInput.config;
  if (countConfig?.systemInstruction !== undefined) config.systemInstruction = countConfig.systemInstruction;
  if (countConfig?.tools !== undefined) config.tools = countConfig.tools;

  if (interactionConfig?.max_output_tokens !== undefined) config.maxOutputTokens = interactionConfig.max_output_tokens;
  if (interactionConfig?.seed !== undefined) config.seed = interactionConfig.seed;
  if (interactionConfig?.stop_sequences !== undefined) config.stopSequences = interactionConfig.stop_sequences;
  if (interactionConfig?.thinking_level !== undefined) {
    const levels = {
      minimal: ThinkingLevel.MINIMAL,
      low: ThinkingLevel.LOW,
      medium: ThinkingLevel.MEDIUM,
      high: ThinkingLevel.HIGH,
    } as const;
    config.thinkingConfig = { thinkingLevel: levels[interactionConfig.thinking_level] };
  }

  const temperature = numberField(body, "temperature");
  if (temperature !== undefined) config.temperature = temperature;
  const topP = numberField(body, "top_p");
  if (topP !== undefined) config.topP = topP;

  const mappedToolConfig = generateContentToolConfig(body);
  if (mappedToolConfig) config.toolConfig = mappedToolConfig;

  const responseFormat = normalized.interaction.response_format;
  if (responseFormat) {
    config.responseMimeType = responseFormat.mime_type;
    if (responseFormat.schema) config.responseJsonSchema = responseFormat.schema;
  }

  const contentStartIndex = continuation
    ? contentIndexBeforeMessage(body, continuation.inputStartIndex)
    : 0;
  const newContents = normalized.countInput.contents.slice(contentStartIndex);
  const recoveredContents = replayExternalToolHistory
    ? replaySafeExternalToolContents(newContents)
    : newContents;
  const contents = continuation ? [...continuation.contents, ...recoveredContents] : recoveredContents;

  return {
    model,
    contents,
    ...(Object.keys(config).length > 0 ? { config } : {}),
  };
}

export function toGoogleInteractionRequest(
  body: ChatCompletionRequest,
  model: string,
  stream: boolean,
  continuation?: { previousInteractionId: string; inputStartIndex: number },
): GoogleInteractionRequest {
  if (body.temperature !== undefined || body.top_p !== undefined) {
    throw new Error("temperature and top_p are not supported by the pinned Google Interactions SDK.");
  }
  const normalized = normalize(body, continuation?.inputStartIndex ?? 0);
  return {
    ...normalized.interaction,
    model,
    stream,
    ...(continuation ? { previous_interaction_id: continuation.previousInteractionId } : {}),
  };
}
