import { GoogleGenAI, type Content } from "@google/genai";
import { log } from "../../log.ts";
import type { ChatCompletionRequest } from "../../types.ts";
import {
  calendarDayBounds,
  effectiveInputCapacity,
  emptyQuotaState,
  parseRetryAfterMs,
  quotaCanEverHandle,
  quotaCanOverflow,
  quotaDelayMs,
  releaseQuota,
  reserveQuota,
  type QuotaLimitName,
  type QuotaPolicy,
  type QuotaRuntimeState,
} from "../shared/quota.ts";
import type {
  OfferRequest,
  Provider,
  ProviderExecutionResult,
  ProviderModelInfo,
  ProviderOffer,
  ProviderOfferResult,
  ProviderStatus,
} from "../shared/types.ts";
import {
  hasAssistantToolCallsFrom,
  hasGoogleToolHistory,
  toAntigravityBootstrapBody,
  toGoogleCountInput,
  toGoogleDeveloperCountTokensRequest,
  toGoogleGenerateContentRequest,
  toGoogleInteractionRequest,
  type GoogleCountInput,
  type GoogleGenerateContentRequest,
  type GoogleInteractionRequest,
} from "./token-count.ts";

const GOOGLE_DAY = { type: "calendar-day", timeZone: "America/Los_Angeles" } as const;
const DEFAULT_RETRY_MS = 5_000;
const PROVIDER_FAILURE_COOLDOWN_MS = 15_000;
const CONTINUATION_TTL_MS = 60 * 60 * 1000;
const MAX_CONTINUATIONS = 1_000;
const OVERFLOW_LIMITS = new Set<QuotaLimitName>(["requestsPerDay"]);
const ANTIGRAVITY_AGENT = "antigravity-preview-09-2026";
const ANTIGRAVITY_SYSTEM_INSTRUCTION = [
  "You are operating as the reasoning backend for an OpenAI-compatible chat-completions interface.",
  "Follow the supplied conversation and its instructions.",
  "Caller-provided function tools operate on the caller's authoritative external environment; use them when appropriate.",
  "Do not assume access to any filesystem, shell, code-execution environment, or remote environment beyond explicitly supplied tools.",
  "Google Search is available for public information.",
].join(" ");

export type GoogleTransport = "interactions" | "generate-content";
export type GoogleThinkingLevel = "minimal" | "low" | "medium" | "high";

export interface GoogleModel {
  id: string;
  upstreamModel: string;
  upstreamAgent?: string;
  contextWindowTokens: number;
  quota: QuotaPolicy;
  preference: number;
  transport?: GoogleTransport;
  thinkingLevels?: readonly GoogleThinkingLevel[];
}

interface Candidate {
  model: GoogleModel;
  inputTokens: number;
  inputCapacityTokens: number;
  availableAt: number;
  index: number;
}

interface Continuation {
  inputStartIndex: number;
  upstreamTarget: string;
  expiresAt: number;
  interactionId?: string;
  generateContents?: Content[];
}

type JsonObject = Record<string, unknown>;

function quota(rpm: number, tpm: number, rpd: number): QuotaPolicy {
  return {
    maxConcurrent: null,
    dailyWindow: GOOGLE_DAY,
    limits: {
      requestsPerMinute: rpm,
      inputTokensPerMinute: tpm,
      requestsPerDay: rpd,
      minimumSpacingMs: 0,
    },
  };
}

// Google AI Studio free-tier limits for this relay project, captured 2026-10-01.
// Only models that accept text and produce text are exposed here, and only when
// the dashboard reports non-zero RPM, TPM, and RPD quotas.
const ALL_THINKING: readonly GoogleThinkingLevel[] = ["minimal", "low", "medium", "high"];
const NO_MINIMAL_THINKING: readonly GoogleThinkingLevel[] = ["low", "medium", "high"];

export const GOOGLE_MODELS: readonly GoogleModel[] = [
  { id: "google/gemini-3.8-flash", upstreamModel: "gemini-3.8-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), preference: 1_300, transport: "interactions", thinkingLevels: NO_MINIMAL_THINKING },
  { id: "google/antigravity-preview-09-2026", upstreamModel: "gemini-3.8-flash", upstreamAgent: ANTIGRAVITY_AGENT, contextWindowTokens: 1_048_576, quota: quota(60, 100_000, 100), preference: 1_200, transport: "interactions" },
  { id: "google/gemini-3.7-flash", upstreamModel: "gemini-3.7-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), preference: 1_100, transport: "interactions", thinkingLevels: NO_MINIMAL_THINKING },
  { id: "google/gemini-3.6-flash", upstreamModel: "gemini-3.6-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), preference: 1_000, transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-3.5-flash", upstreamModel: "gemini-3.5-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), preference: 900, transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-3.5-flash-lite", upstreamModel: "gemini-3.5-flash-lite", contextWindowTokens: 1_048_576, quota: quota(15, 250_000, 500), preference: 800, transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-3.1-flash-lite", upstreamModel: "gemini-3.1-flash-lite", contextWindowTokens: 1_048_576, quota: quota(15, 250_000, 500), preference: 700, transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-3-flash-preview", upstreamModel: "gemini-3-flash-preview", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), preference: 600, transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-robotics-er-2-preview", upstreamModel: "gemini-robotics-er-2-preview", contextWindowTokens: 131_072, quota: quota(5, 250_000, 20), preference: 500, transport: "interactions", thinkingLevels: ALL_THINKING },
  // Gemini 2.5 Flash and Flash-Lite both returned Developer API 404s ("no longer available to new users") on 2026-10-01.
  { id: "google/gemma-4-31b-it", upstreamModel: "gemma-4-31b-it", contextWindowTokens: 262_144, quota: quota(30, 16_000, 14_400), preference: 200, transport: "interactions" },
  { id: "google/gemma-4-26b-a4b-it", upstreamModel: "gemma-4-26b-a4b-it", contextWindowTokens: 262_144, quota: quota(30, 16_000, 14_400), preference: 100, transport: "interactions" },
];

function object(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function compareCapacity(left: Candidate, right: Candidate): number {
  return left.inputCapacityTokens - right.inputCapacityTokens
    || left.availableAt - right.availableAt
    || right.model.preference - left.model.preference
    || left.index - right.index;
}

function compareAvailability(left: Candidate, right: Candidate): number {
  return left.availableAt - right.availableAt
    || left.inputCapacityTokens - right.inputCapacityTokens
    || right.model.preference - left.model.preference
    || left.index - right.index;
}

function messageArray(body: ChatCompletionRequest): unknown[] | null {
  return Array.isArray(body.messages) ? body.messages : null;
}

function prefixKey(messages: readonly unknown[]): string {
  return JSON.stringify(messages);
}

function transportFor(model: GoogleModel): GoogleTransport {
  return model.transport ?? "interactions";
}

function interactionTarget(model: GoogleModel): string {
  return model.upstreamAgent
    ? `agent:${model.upstreamAgent}`
    : `interactions:${model.upstreamModel}`;
}

function generateContentTarget(model: GoogleModel): string {
  return `generate-content:${model.upstreamModel}`;
}

function canReplayExternalToolHistory(model: GoogleModel): boolean {
  return !model.upstreamAgent && model.upstreamModel.startsWith("gemini-");
}

function modelSupportsRequest(model: GoogleModel, body: ChatCompletionRequest): boolean {
  if (model.upstreamAgent) {
    if (
      body.temperature !== undefined
      || body.top_p !== undefined
      || body.stop !== undefined
      || body.seed !== undefined
      || body.reasoning_effort !== undefined
    ) return false;
    if (body.tool_choice !== undefined && body.tool_choice !== "auto") return false;
  }
  const effort = body.reasoning_effort;
  if (effort !== undefined) {
    if (effort !== "minimal" && effort !== "low" && effort !== "medium" && effort !== "high") return true;
    if (!model.thinkingLevels?.includes(effort)) return false;
  }
  if (transportFor(model) === "interactions" && (body.temperature !== undefined || body.top_p !== undefined)) {
    return false;
  }
  return true;
}

interface GoogleAgentInteractionRequest {
  agent: string;
  agent_config: {
    type: "antigravity";
    model: string;
    max_total_tokens?: string;
  };
  input: GoogleInteractionRequest["input"];
  store: true;
  stream: boolean;
  previous_interaction_id?: string;
  system_instruction?: string;
  tools: Array<
    | { type: "google_search" }
    | NonNullable<GoogleInteractionRequest["tools"]>[number]
  >;
  response_format?: GoogleInteractionRequest["response_format"];
}

function antigravityRequest(
  request: GoogleInteractionRequest,
  model: GoogleModel,
  inputTokens: number,
): GoogleAgentInteractionRequest {
  if (!model.upstreamAgent) throw new Error(`Google model ${model.id} is not an agent route.`);
  const unsupported = Object.entries(request.generation_config ?? {})
    .filter(([key, value]) =>
      key !== "max_output_tokens"
      && !(key === "tool_choice" && value === "auto")
    )
    .map(([key]) => key);
  if (unsupported.length > 0) {
    throw new Error(`Antigravity does not support Chat Completions options: ${unsupported.join(", ")}.`);
  }

  const maxOutputTokens = request.generation_config?.max_output_tokens;
  const maxTotalTokens = maxOutputTokens === undefined
    ? undefined
    : BigInt(inputTokens) + BigInt(maxOutputTokens);
  return {
    agent: model.upstreamAgent,
    agent_config: {
      type: "antigravity",
      model: model.upstreamModel,
      ...(maxTotalTokens !== undefined ? { max_total_tokens: String(maxTotalTokens) } : {}),
    },
    input: request.input,
    store: true,
    stream: request.stream,
    ...(request.previous_interaction_id ? { previous_interaction_id: request.previous_interaction_id } : {}),
    system_instruction: request.system_instruction
      ? `${ANTIGRAVITY_SYSTEM_INSTRUCTION}\n\n${request.system_instruction}`
      : ANTIGRAVITY_SYSTEM_INSTRUCTION,
    tools: [
      { type: "google_search" },
      ...(request.tools ?? []),
    ],
    ...(request.response_format ? { response_format: request.response_format } : {}),
  };
}

function assistantMessageFromInteraction(interaction: unknown): JsonObject {
  const value = object(interaction);
  const steps = Array.isArray(value?.steps) ? value.steps : [];
  let content = "";
  const toolCalls: JsonObject[] = [];

  for (const rawStep of steps) {
    const step = object(rawStep);
    if (!step) continue;
    if (step.type === "model_output" && Array.isArray(step.content)) {
      for (const rawContent of step.content) {
        const block = object(rawContent);
        if (block?.type === "text" && typeof block.text === "string") content += block.text;
      }
    }
    if (step.type === "function_call" && typeof step.id === "string" && typeof step.name === "string") {
      toolCalls.push({
        id: step.id,
        type: "function",
        function: {
          name: step.name,
          arguments: JSON.stringify(object(step.arguments) ?? {}),
        },
      });
    }
  }

  const message: JsonObject = { role: "assistant", content: content || (toolCalls.length > 0 ? null : "") };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return message;
}

interface GenerateAssistant {
  assistant: JsonObject;
  modelContent?: Content;
  hasToolCalls: boolean;
}

function generateAssistant(response: unknown): GenerateAssistant {
  const value = object(response) ?? {};
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const firstCandidate = object(candidates[0]);
  const rawContent = object(firstCandidate?.content);
  const rawParts = Array.isArray(rawContent?.parts) ? rawContent.parts : [];
  const parts: JsonObject[] = [];
  let text = "";
  const toolCalls: JsonObject[] = [];

  for (const [index, rawPart] of rawParts.entries()) {
    const part = object(rawPart);
    if (!part) continue;
    const cloned: JsonObject = { ...part };
    if (typeof part.text === "string" && part.thought !== true) text += part.text;

    const functionCall = object(part.functionCall);
    if (functionCall && typeof functionCall.name === "string") {
      const id = typeof functionCall.id === "string" && functionCall.id
        ? functionCall.id
        : `call_google_${index}`;
      const normalizedCall = { ...functionCall, id };
      cloned.functionCall = normalizedCall;
      toolCalls.push({
        id,
        type: "function",
        function: {
          name: functionCall.name,
          arguments: JSON.stringify(object(functionCall.args) ?? {}),
        },
      });
    }
    parts.push(cloned);
  }

  const assistant: JsonObject = {
    role: "assistant",
    content: text || (toolCalls.length > 0 ? null : ""),
  };
  if (toolCalls.length > 0) assistant.tool_calls = toolCalls;

  return {
    assistant,
    modelContent: rawContent
      ? { role: typeof rawContent.role === "string" ? rawContent.role : "model", parts: parts as any[] }
      : undefined,
    hasToolCalls: toolCalls.length > 0,
  };
}

function generateUsagePayload(rawUsage: unknown, fallbackInputTokens: number): JsonObject {
  const usage = object(rawUsage);
  const promptTokens = typeof usage?.promptTokenCount === "number" ? usage.promptTokenCount : fallbackInputTokens;
  const completionTokens = typeof usage?.candidatesTokenCount === "number" ? usage.candidatesTokenCount : 0;
  const totalTokens = typeof usage?.totalTokenCount === "number" ? usage.totalTokenCount : promptTokens + completionTokens;
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
  };
}

function generateFinishReason(response: unknown, hasToolCalls: boolean): string {
  if (hasToolCalls) return "tool_calls";
  const value = object(response);
  const candidates = Array.isArray(value?.candidates) ? value.candidates : [];
  const reason = object(candidates[0])?.finishReason;
  if (reason === "MAX_TOKENS") return "length";
  if (
    reason === "SAFETY"
    || reason === "BLOCKLIST"
    || reason === "PROHIBITED_CONTENT"
    || reason === "SPII"
    || reason === "IMAGE_SAFETY"
    || reason === "IMAGE_PROHIBITED_CONTENT"
  ) return "content_filter";
  return "stop";
}

export function googleGenerateContentToOpenAI(
  response: unknown,
  relayModelId: string,
  inputTokens: number,
): JsonObject {
  const value = object(response) ?? {};
  const generated = generateAssistant(value);
  const createdMs = typeof value.createTime === "string" ? Date.parse(value.createTime) : Number.NaN;
  const responseId = typeof value.responseId === "string" && value.responseId ? value.responseId : "google";
  return {
    id: `chatcmpl-${responseId}`,
    object: "chat.completion",
    created: Number.isFinite(createdMs) ? Math.floor(createdMs / 1000) : Math.floor(Date.now() / 1000),
    model: relayModelId,
    choices: [{
      index: 0,
      message: generated.assistant,
      finish_reason: generateFinishReason(value, generated.hasToolCalls),
      logprobs: null,
    }],
    usage: generateUsagePayload(value.usageMetadata, inputTokens),
  };
}

function usagePayload(rawUsage: unknown, fallbackInputTokens: number): JsonObject {
  const usage = object(rawUsage);
  const promptTokens = typeof usage?.total_input_tokens === "number" ? usage.total_input_tokens : fallbackInputTokens;
  const completionTokens = typeof usage?.total_output_tokens === "number" ? usage.total_output_tokens : 0;
  const totalTokens = typeof usage?.total_tokens === "number" ? usage.total_tokens : promptTokens + completionTokens;
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
  };
}

function finishReason(status: unknown, hasToolCalls: boolean): string {
  if (hasToolCalls || status === "requires_action") return "tool_calls";
  if (status === "incomplete" || status === "budget_exceeded") return "length";
  return "stop";
}

export function googleInteractionToOpenAI(
  interaction: unknown,
  relayModelId: string,
  inputTokens: number,
): JsonObject {
  const value = object(interaction) ?? {};
  const assistant = assistantMessageFromInteraction(value);
  const toolCalls = Array.isArray(assistant.tool_calls) ? assistant.tool_calls : [];
  const createdMs = typeof value.created === "string" ? Date.parse(value.created) : Number.NaN;
  return {
    id: `chatcmpl-${typeof value.id === "string" ? value.id : "google"}`,
    object: "chat.completion",
    created: Number.isFinite(createdMs) ? Math.floor(createdMs / 1000) : Math.floor(Date.now() / 1000),
    model: relayModelId,
    choices: [{
      index: 0,
      message: assistant,
      finish_reason: finishReason(value.status, toolCalls.length > 0),
      logprobs: null,
    }],
    usage: usagePayload(value.usage, inputTokens),
  };
}

function sse(value: unknown): Uint8Array {
  return new TextEncoder().encode(`data: ${typeof value === "string" ? value : JSON.stringify(value)}\n\n`);
}

function streamIncludesUsage(body: ChatCompletionRequest): boolean {
  const options = object(body.stream_options);
  return options?.include_usage === true;
}

function statusCode(error: unknown): number | undefined {
  const value = object(error);
  const code = value?.statusCode ?? value?.status;
  return typeof code === "number" && Number.isInteger(code) ? code : undefined;
}

function errorHeaders(error: unknown): Headers | undefined {
  const value = object(error);
  return value?.headers instanceof Headers ? value.headers : undefined;
}

function errorBody(error: unknown): string {
  const value = object(error);
  if (typeof value?.body === "string" && value.body) return value.body;
  if (value?.error !== undefined) {
    try { return JSON.stringify(value.error); } catch { /* ignore */ }
  }
  return error instanceof Error ? error.message : String(error);
}

type GoogleDeveloperTokenCounter = (
  apiKey: string,
  model: string,
  input: GoogleCountInput,
) => Promise<number>;

async function countDeveloperApiTokens(
  apiKey: string,
  model: string,
  input: GoogleCountInput,
): Promise<number> {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:countTokens`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-goog-api-key": apiKey,
      },
      body: JSON.stringify(toGoogleDeveloperCountTokensRequest(input, model)),
      signal: AbortSignal.timeout(10_000),
    },
  );

  const bodyText = await response.text();
  if (!response.ok) {
    const error = Object.assign(
      new Error(`Google countTokens returned HTTP ${response.status}: ${bodyText || response.statusText}`),
      {
        statusCode: response.status,
        headers: response.headers,
        body: bodyText,
      },
    );
    throw error;
  }

  let payload: unknown;
  try {
    payload = JSON.parse(bodyText) as unknown;
  } catch {
    throw new Error("Google countTokens returned invalid JSON.");
  }
  const value = object(payload);
  const totalTokens = value?.totalTokens;
  if (typeof totalTokens !== "number" || !Number.isSafeInteger(totalTokens) || totalTokens < 0) {
    throw new Error(`Google did not return a valid token count for ${model}.`);
  }
  return totalTokens;
}

function confirmsDailyQuota(error: unknown): boolean {
  const value = object(error);
  const code = typeof value?.code === "string" ? value.code : "";
  const message = error instanceof Error ? error.message : "";
  const text = `${code} ${message} ${errorBody(error)}`.toLowerCase();

  return text.includes("quota_exceeded")
    || text.includes("generaterequestsperday")
    || text.includes("requestsperday")
    || text.includes("requests per day")
    || text.includes("daily quota");
}

interface GoogleRequestPlan {
  body: ChatCompletionRequest;
  transport: GoogleTransport;
  target: string;
  continuation?: Continuation;
  replayExternalToolHistory: boolean;
  bootstrapAntigravity: boolean;
}

export class GoogleProvider implements Provider {
  readonly id = "google";
  readonly priority = 10;
  private readonly models: readonly GoogleModel[];
  private readonly modelStates = new Map<string, QuotaRuntimeState>();
  private readonly providerState = emptyQuotaState();
  private readonly tokenCountCache = new WeakMap<ChatCompletionRequest, Map<string, Promise<number>>>();
  private readonly modelFailureCounts = new Map<string, number>();
  private readonly modelOverflowBlockedUntil = new Map<string, number>();
  private readonly continuations = new Map<string, Continuation>();
  private providerFailureCount = 0;
  private client?: GoogleGenAI;
  private clientApiKey?: string;
  private readonly clientFactory: (apiKey: string) => GoogleGenAI;
  private readonly developerTokenCounter: GoogleDeveloperTokenCounter;

  constructor(
    clientFactory: (apiKey: string) => GoogleGenAI = (apiKey) => new GoogleGenAI({ apiKey }),
    models: readonly GoogleModel[] = GOOGLE_MODELS,
    developerTokenCounter: GoogleDeveloperTokenCounter = countDeveloperApiTokens,
  ) {
    this.clientFactory = clientFactory;
    this.models = models;
    this.developerTokenCounter = developerTokenCounter;
    for (const model of this.models) this.modelStates.set(model.id, emptyQuotaState());
  }

  isConfigured(): boolean {
    return Boolean(process.env.GEMINI_API_KEY?.trim());
  }

  listModels(): readonly ProviderModelInfo[] {
    return this.models.map((model) => ({
      id: model.id,
      providerId: this.id,
      inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
    }));
  }

  private googleApiKey(): string {
    const apiKey = process.env.GEMINI_API_KEY?.trim();
    if (!apiKey) throw new Error("GEMINI_API_KEY is not configured.");
    return apiKey;
  }

  private googleClient(): GoogleGenAI {
    const apiKey = this.googleApiKey();
    if (!this.client || this.clientApiKey !== apiKey) {
      this.client = this.clientFactory(apiKey);
      this.clientApiKey = apiKey;
    }
    return this.client;
  }

  private modelState(modelId: string): QuotaRuntimeState {
    const state = this.modelStates.get(modelId);
    if (!state) throw new Error(`Unknown Google model state: ${modelId}`);
    return state;
  }

  private modelById(modelId: string): GoogleModel {
    const model = this.models.find((candidate) => candidate.id === modelId);
    if (!model) throw new Error(`Google does not own model ${modelId}`);
    return model;
  }

  private safeContinuation(body: ChatCompletionRequest, target: string): Continuation | undefined {
    const continuation = this.continuationFor(body, target);
    if (!continuation) return undefined;
    return hasAssistantToolCallsFrom(body, continuation.inputStartIndex)
      ? undefined
      : continuation;
  }

  private requestPlan(body: ChatCompletionRequest, model: GoogleModel): GoogleRequestPlan {
    if (model.upstreamAgent) {
      const target = interactionTarget(model);
      const continuation = this.safeContinuation(body, target);
      if (continuation) {
        return {
          body,
          transport: "interactions",
          target,
          continuation,
          replayExternalToolHistory: false,
          bootstrapAntigravity: false,
        };
      }
      const bootstrapAntigravity = hasGoogleToolHistory(body);
      return {
        body: bootstrapAntigravity ? toAntigravityBootstrapBody(body) : body,
        transport: "interactions",
        target,
        replayExternalToolHistory: false,
        bootstrapAntigravity,
      };
    }

    const interactionContinuation = this.safeContinuation(body, interactionTarget(model));
    const generateContinuation = this.safeContinuation(body, generateContentTarget(model));

    if (
      generateContinuation
      && (!interactionContinuation || generateContinuation.inputStartIndex >= interactionContinuation.inputStartIndex)
    ) {
      return {
        body,
        transport: "generate-content",
        target: generateContentTarget(model),
        continuation: generateContinuation,
        replayExternalToolHistory: false,
        bootstrapAntigravity: false,
      };
    }

    if (interactionContinuation && transportFor(model) !== "generate-content") {
      return {
        body,
        transport: "interactions",
        target: interactionTarget(model),
        continuation: interactionContinuation,
        replayExternalToolHistory: false,
        bootstrapAntigravity: false,
      };
    }

    const replayExternalToolHistory = hasGoogleToolHistory(body);
    if (
      transportFor(model) === "generate-content"
      || (replayExternalToolHistory && canReplayExternalToolHistory(model))
    ) {
      return {
        body,
        transport: "generate-content",
        target: generateContentTarget(model),
        ...(generateContinuation ? { continuation: generateContinuation } : {}),
        replayExternalToolHistory: replayExternalToolHistory && !generateContinuation,
        bootstrapAntigravity: false,
      };
    }

    return {
      body,
      transport: "interactions",
      target: interactionTarget(model),
      replayExternalToolHistory: false,
      bootstrapAntigravity: false,
    };
  }

  private async countInputTokens(body: ChatCompletionRequest, modelId: string): Promise<number> {
    const model = this.modelById(modelId);
    const plan = this.requestPlan(body, model);
    const cacheKey = [
      modelId,
      plan.transport,
      plan.target,
      plan.continuation?.inputStartIndex ?? -1,
      plan.replayExternalToolHistory ? "replay" : "native",
      plan.bootstrapAntigravity ? "bootstrap" : "direct",
    ].join(":");
    let perModel = this.tokenCountCache.get(body);
    if (!perModel) {
      perModel = new Map<string, Promise<number>>();
      this.tokenCountCache.set(body, perModel);
    }
    const cached = perModel.get(cacheKey);
    if (cached) return cached;

    const pending = Promise.resolve().then(async () => {
      const input = toGoogleCountInput(plan.body);
      if (model.upstreamAgent) {
        const existingSystem = object(input.config?.systemInstruction);
        const existingParts = Array.isArray(existingSystem?.parts) ? existingSystem.parts : [];
        input.config = {
          ...input.config,
          systemInstruction: {
            parts: [{ text: ANTIGRAVITY_SYSTEM_INSTRUCTION }, ...existingParts as any[]],
          },
          tools: [
            ...(input.config?.tools ?? []),
            { googleSearch: {} },
          ],
        };
      }
      let contents = input.contents;
      if (plan.transport === "generate-content") {
        const request = toGoogleGenerateContentRequest(
          plan.body,
          model.upstreamModel,
          plan.continuation?.generateContents ? {
            contents: plan.continuation.generateContents,
            inputStartIndex: plan.continuation.inputStartIndex,
          } : undefined,
          plan.replayExternalToolHistory,
        );
        contents = request.contents;
      } else if (
        hasGoogleToolHistory(plan.body)
        && (model.upstreamAgent || canReplayExternalToolHistory(model))
      ) {
        // Google-native signatures are not representable in OpenAI tool_calls.
        // Count an equivalent replay-safe Gemini request instead.
        contents = toGoogleGenerateContentRequest(
          plan.body,
          model.upstreamModel,
          undefined,
          true,
        ).contents;
      }
      if (input.config !== undefined) {
        return this.developerTokenCounter(this.googleApiKey(), model.upstreamModel, {
          contents,
          config: input.config,
        });
      }

      const response = await this.googleClient().models.countTokens({
        model: model.upstreamModel,
        contents,
        config: { abortSignal: AbortSignal.timeout(10_000) },
      });
      if (!Number.isSafeInteger(response.totalTokens) || (response.totalTokens ?? -1) < 0) {
        throw new Error(`Google did not return a valid token count for ${model.upstreamModel}.`);
      }
      return response.totalTokens!;
    }).catch((error) => {
      perModel!.delete(cacheKey);
      throw error;
    });

    perModel.set(cacheKey, pending);
    return pending;
  }

  private pruneOverflowBlocks(now = Date.now()): void {
    for (const [modelId, blockedUntil] of this.modelOverflowBlockedUntil) {
      if (blockedUntil > now) continue;
      this.modelOverflowBlockedUntil.delete(modelId);
      log("info", "overflow_boundary_expired", {
        provider: this.id,
        relay_model: modelId,
        reset_at: new Date(blockedUntil).toISOString(),
      });
    }
  }

  private async candidate(model: GoogleModel, index: number, request: OfferRequest, now: number): Promise<Candidate | null> {
    if (request.excludedModelIds.has(model.id)) return null;
    if (request.requestedModel !== "auto" && request.requestedModel !== model.id) return null;
    if (!modelSupportsRequest(model, request.body)) return null;

    const inputTokens = await this.countInputTokens(request.body, model.id);
    if (!quotaCanEverHandle(model.quota, inputTokens, model.contextWindowTokens)) return null;

    const evaluatedAt = Math.max(now, Date.now());
    this.pruneOverflowBlocks(evaluatedAt);
    const state = this.modelState(model.id);
    const modelDelayMs = quotaDelayMs(model.quota, state, inputTokens, evaluatedAt);
    const providerBlockedMs = Math.max(0, this.providerState.blockedUntil - evaluatedAt);

    if (request.offerKind === "overflow") {
      if (providerBlockedMs > 0) return null;
      if ((this.modelOverflowBlockedUntil.get(model.id) ?? 0) > evaluatedAt) return null;
      if (!quotaCanOverflow(
        model.quota,
        state,
        inputTokens,
        OVERFLOW_LIMITS,
        model.contextWindowTokens,
        evaluatedAt,
      )) return null;
      return {
        model,
        inputTokens,
        inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
        availableAt: now,
        index,
      };
    }

    const delayMs = Math.max(modelDelayMs, providerBlockedMs);
    return {
      model,
      inputTokens,
      inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
      availableAt: Number.isFinite(delayMs)
        ? (delayMs <= 0 ? now : evaluatedAt + delayMs)
        : Number.POSITIVE_INFINITY,
      index,
    };
  }

  async getBestOffer(request: OfferRequest, now = Date.now()): Promise<ProviderOfferResult> {
    if (!this.isConfigured()) {
      return { status: "no_offer", providerId: this.id, reason: "provider_not_configured" };
    }

    const eligibleModels = this.models
      .map((model, index) => ({ model, index }))
      .filter(({ model }) => !request.excludedModelIds.has(model.id))
      .filter(({ model }) => request.requestedModel === "auto" || request.requestedModel === model.id)
      .filter(({ model }) => modelSupportsRequest(model, request.body));

    if (eligibleModels.length === 0) {
      return { status: "no_offer", providerId: this.id, reason: "no_eligible_model" };
    }

    const results = await Promise.all(eligibleModels.map(async ({ model, index }) => {
      try { return { candidate: await this.candidate(model, index, request, now), error: undefined }; }
      catch (error) { return { candidate: null, error }; }
    }));
    const candidates = results.map((result) => result.candidate).filter((candidate): candidate is Candidate => candidate !== null);
    if (candidates.length === 0) {
      const countingError = results.find((result) => result.error !== undefined)?.error;
      if (countingError !== undefined) {
        return {
          status: "no_offer",
          providerId: this.id,
          reason: "token_count_failed",
          detail: countingError instanceof Error ? countingError.message : String(countingError),
        };
      }
      return {
        status: "no_offer",
        providerId: this.id,
        reason: request.offerKind === "standard" ? "request_exceeds_capacity" : "no_eligible_model",
      };
    }

    const chosen = request.requestedModel !== "auto"
      ? candidates[0]!
      : (candidates.filter((candidate) => candidate.availableAt <= now + request.maxOptimizationWaitMs).sort(compareCapacity)[0]
        ?? [...candidates].sort(compareAvailability)[0]!);

    return {
      status: "offer",
      offer: {
        kind: request.offerKind,
        providerId: this.id,
        providerPriority: this.priority,
        modelId: chosen.model.id,
        inputTokens: chosen.inputTokens,
        inputCapacityTokens: chosen.inputCapacityTokens,
        availableAt: chosen.availableAt,
      },
    };
  }

  private release(model: GoogleModel): void {
    releaseQuota(this.modelState(model.id));
  }

  private blockModel(model: GoogleModel, requestedDelayMs: number, now: number, reason: string): number {
    const failures = (this.modelFailureCounts.get(model.id) ?? 0) + 1;
    this.modelFailureCounts.set(model.id, failures);
    const multiplier = 2 ** Math.min(failures - 1, 4);
    const delayMs = Math.min(60_000, Math.max(DEFAULT_RETRY_MS, requestedDelayMs) * multiplier);
    const retryAt = now + delayMs;
    const state = this.modelState(model.id);
    state.blockedUntil = Math.max(state.blockedUntil, retryAt);
    log("warn", "provider_model_cooldown", {
      provider: this.id,
      relay_model: model.id,
      reason,
      consecutive_failures: failures,
      blocked_until: new Date(state.blockedUntil).toISOString(),
    });
    return retryAt;
  }

  private blockProvider(requestedDelayMs: number, now: number, reason: string): number {
    this.providerFailureCount += 1;
    const multiplier = 2 ** Math.min(this.providerFailureCount - 1, 4);
    const delayMs = Math.min(120_000, Math.max(PROVIDER_FAILURE_COOLDOWN_MS, requestedDelayMs) * multiplier);
    const retryAt = now + delayMs;
    this.providerState.blockedUntil = Math.max(this.providerState.blockedUntil, retryAt);
    log("warn", "provider_cooldown", {
      provider: this.id,
      reason,
      consecutive_failures: this.providerFailureCount,
      blocked_until: new Date(this.providerState.blockedUntil).toISOString(),
    });
    return retryAt;
  }

  private pruneContinuations(now = Date.now()): void {
    for (const [key, continuation] of this.continuations) {
      if (continuation.expiresAt <= now) this.continuations.delete(key);
    }
    while (this.continuations.size > MAX_CONTINUATIONS) {
      const oldest = this.continuations.keys().next().value;
      if (typeof oldest !== "string") break;
      this.continuations.delete(oldest);
    }
  }

  private continuationFor(body: ChatCompletionRequest, upstreamTarget: string): Continuation | undefined {
    const messages = messageArray(body);
    if (!messages) return undefined;
    this.pruneContinuations();
    for (let length = messages.length; length > 0; length -= 1) {
      const continuation = this.continuations.get(prefixKey(messages.slice(0, length)));
      if (continuation?.upstreamTarget === upstreamTarget) return continuation;
    }
    return undefined;
  }

  private rememberContinuation(body: ChatCompletionRequest, interaction: unknown, upstreamTarget: string): void {
    const messages = messageArray(body);
    const value = object(interaction);
    if (!messages || typeof value?.id !== "string") return;
    const assistant = assistantMessageFromInteraction(value);
    const continued = [...messages, assistant];
    this.continuations.delete(prefixKey(continued));
    this.continuations.set(prefixKey(continued), {
      interactionId: value.id,
      inputStartIndex: continued.length,
      upstreamTarget,
      expiresAt: Date.now() + CONTINUATION_TTL_MS,
    });
    this.pruneContinuations();
  }

  private rememberGenerateContinuation(
    body: ChatCompletionRequest,
    response: unknown,
    requestContents: Content[],
    upstreamTarget: string,
  ): void {
    const messages = messageArray(body);
    if (!messages) return;
    const generated = generateAssistant(response);
    if (!generated.modelContent) return;
    const continued = [...messages, generated.assistant];
    this.continuations.delete(prefixKey(continued));
    this.continuations.set(prefixKey(continued), {
      generateContents: [...requestContents, generated.modelContent],
      inputStartIndex: continued.length,
      upstreamTarget,
      expiresAt: Date.now() + CONTINUATION_TTL_MS,
    });
    this.pruneContinuations();
  }

  private rememberGenerateStreamContinuation(
    body: ChatCompletionRequest,
    assistant: JsonObject,
    modelContent: Content,
    requestContents: Content[],
    upstreamTarget: string,
  ): void {
    const messages = messageArray(body);
    if (!messages) return;
    const continued = [...messages, assistant];
    this.continuations.delete(prefixKey(continued));
    this.continuations.set(prefixKey(continued), {
      generateContents: [...requestContents, modelContent],
      inputStartIndex: continued.length,
      upstreamTarget,
      expiresAt: Date.now() + CONTINUATION_TTL_MS,
    });
    this.pruneContinuations();
  }

  private openAIStream(
    stream: AsyncIterable<unknown>,
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    upstreamTarget: string,
  ): Response {
    const encoder = new TextEncoder();
    const provider = this;
    const output = new ReadableStream<Uint8Array>({
      async start(controller) {
        let interactionId = "google";
        let created = Math.floor(Date.now() / 1000);
        let sentRole = false;
        let sawToolCall = false;
        let finalStatus: unknown = "completed";
        let finalUsage: unknown;
        let fullText = "";
        const toolIndexByStep = new Map<number, number>();
        const toolCalls: JsonObject[] = [];

        const emit = (delta: JsonObject, finishReason: string | null = null, usage?: JsonObject) => {
          if (!sentRole) {
            delta = { role: "assistant", ...delta };
            sentRole = true;
          }
          const chunk: JsonObject = {
            id: `chatcmpl-${interactionId}`,
            object: "chat.completion.chunk",
            created,
            model: offer.modelId,
            choices: [{ index: 0, delta, finish_reason: finishReason, logprobs: null }],
          };
          if (usage) chunk.usage = usage;
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
        };

        try {
          for await (const rawEvent of stream) {
            const event = object(rawEvent);
            if (!event || typeof event.event_type !== "string") continue;
            if (event.event_type === "interaction.created") {
              const interaction = object(event.interaction);
              if (typeof interaction?.id === "string") interactionId = interaction.id;
              if (typeof interaction?.created === "string") {
                const parsed = Date.parse(interaction.created);
                if (Number.isFinite(parsed)) created = Math.floor(parsed / 1000);
              }
              emit({});
              continue;
            }
            if (event.event_type === "step.start") {
              const step = object(event.step);
              if (step?.type === "function_call" && typeof step.id === "string" && typeof step.name === "string") {
                sawToolCall = true;
                const toolIndex = toolCalls.length;
                toolIndexByStep.set(typeof event.index === "number" ? event.index : toolIndex, toolIndex);
                const call = {
                  id: step.id,
                  type: "function",
                  function: { name: step.name, arguments: "" },
                };
                toolCalls.push(call);
                emit({ tool_calls: [{ index: toolIndex, ...call }] });
              }
              continue;
            }
            if (event.event_type === "step.delta") {
              const delta = object(event.delta);
              if (delta?.type === "text" && typeof delta.text === "string") {
                fullText += delta.text;
                emit({ content: delta.text });
              } else if (delta?.type === "arguments_delta" && typeof delta.arguments === "string") {
                const stepIndex = typeof event.index === "number" ? event.index : -1;
                const toolIndex = toolIndexByStep.get(stepIndex);
                if (toolIndex !== undefined) {
                  const call = object(toolCalls[toolIndex]);
                  const fn = call ? object(call.function) : null;
                  if (fn) fn.arguments = `${typeof fn.arguments === "string" ? fn.arguments : ""}${delta.arguments}`;
                  emit({ tool_calls: [{ index: toolIndex, function: { arguments: delta.arguments } }] });
                }
              }
              continue;
            }
            if (event.event_type === "interaction.status_update") {
              finalStatus = event.status;
              continue;
            }
            if (event.event_type === "interaction.completed") {
              const interaction = object(event.interaction);
              if (typeof interaction?.id === "string") interactionId = interaction.id;
              finalStatus = interaction?.status ?? finalStatus;
              finalUsage = interaction?.usage;
              if (interaction) {
                const synthetic = {
                  ...interaction,
                  steps: [
                    ...(fullText ? [{ type: "model_output", content: [{ type: "text", text: fullText }] }] : []),
                    ...toolCalls.map((call) => ({
                      type: "function_call",
                      id: call.id,
                      name: object(call.function)?.name,
                      arguments: (() => {
                        const args = object(call.function)?.arguments;
                        if (typeof args !== "string" || !args) return {};
                        try { return object(JSON.parse(args)) ?? {}; } catch { return {}; }
                      })(),
                    })),
                  ],
                };
                provider.rememberContinuation(body, synthetic, upstreamTarget);
              }
              continue;
            }
            if (event.event_type === "error") {
              controller.enqueue(sse({ error: event.error ?? { message: "Google interaction stream failed." } }));
              controller.enqueue(sse("[DONE]"));
              controller.close();
              return;
            }
          }

          emit({}, finishReason(finalStatus, sawToolCall));
          if (streamIncludesUsage(body)) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({
              id: `chatcmpl-${interactionId}`,
              object: "chat.completion.chunk",
              created,
              model: offer.modelId,
              choices: [],
              usage: usagePayload(finalUsage, offer.inputTokens),
            })}\n\n`));
          }
          controller.enqueue(sse("[DONE]"));
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });

    return new Response(output, {
      status: 200,
      headers: {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache",
      },
    });
  }

  private openAIGenerateStream(
    stream: AsyncIterable<unknown>,
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    requestContents: Content[],
    upstreamTarget: string,
  ): Response {
    const encoder = new TextEncoder();
    const provider = this;
    const output = new ReadableStream<Uint8Array>({
      async start(controller) {
        let responseId = "google";
        let created = Math.floor(Date.now() / 1000);
        let sentRole = false;
        let fullText = "";
        let finalFinishReason: unknown;
        let finalUsage: unknown;
        const toolCalls: JsonObject[] = [];
        const toolIndexById = new Map<string, number>();
        const modelParts: JsonObject[] = [];

        const emit = (delta: JsonObject, finishReasonValue: string | null = null) => {
          if (!sentRole) {
            delta = { role: "assistant", ...delta };
            sentRole = true;
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            id: `chatcmpl-${responseId}`,
            object: "chat.completion.chunk",
            created,
            model: offer.modelId,
            choices: [{ index: 0, delta, finish_reason: finishReasonValue, logprobs: null }],
          })}\n\n`));
        };

        try {
          for await (const rawChunk of stream) {
            const chunk = object(rawChunk);
            if (!chunk) continue;
            if (typeof chunk.responseId === "string" && chunk.responseId) responseId = chunk.responseId;
            if (typeof chunk.createTime === "string") {
              const parsed = Date.parse(chunk.createTime);
              if (Number.isFinite(parsed)) created = Math.floor(parsed / 1000);
            }
            if (chunk.usageMetadata !== undefined) finalUsage = chunk.usageMetadata;

            const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
            const candidate = object(candidates[0]);
            if (candidate?.finishReason !== undefined) finalFinishReason = candidate.finishReason;
            const content = object(candidate?.content);
            const parts = Array.isArray(content?.parts) ? content.parts : [];
            for (const [partIndex, rawPart] of parts.entries()) {
              const part = object(rawPart);
              if (!part) continue;
              const cloned: JsonObject = { ...part };
              if (typeof part.text === "string" && part.thought !== true) {
                fullText += part.text;
                emit({ content: part.text });
              }

              const functionCall = object(part.functionCall);
              if (functionCall && typeof functionCall.name === "string") {
                const id = typeof functionCall.id === "string" && functionCall.id
                  ? functionCall.id
                  : `call_google_${toolCalls.length + partIndex}`;
                cloned.functionCall = { ...functionCall, id };
                if (!toolIndexById.has(id)) {
                  const index = toolCalls.length;
                  toolIndexById.set(id, index);
                  const call: JsonObject = {
                    id,
                    type: "function",
                    function: {
                      name: functionCall.name,
                      arguments: JSON.stringify(object(functionCall.args) ?? {}),
                    },
                  };
                  toolCalls.push(call);
                  emit({ tool_calls: [{ index, ...call }] });
                }
              }
              modelParts.push(cloned);
            }
          }

          const assistant: JsonObject = {
            role: "assistant",
            content: fullText || (toolCalls.length > 0 ? null : ""),
          };
          if (toolCalls.length > 0) assistant.tool_calls = toolCalls;
          if (modelParts.length > 0) {
            provider.rememberGenerateStreamContinuation(
              body,
              assistant,
              { role: "model", parts: modelParts as any[] },
              requestContents,
              upstreamTarget,
            );
          }

          const finish = toolCalls.length > 0
            ? "tool_calls"
            : finalFinishReason === "MAX_TOKENS"
              ? "length"
              : (
                  finalFinishReason === "SAFETY"
                  || finalFinishReason === "BLOCKLIST"
                  || finalFinishReason === "PROHIBITED_CONTENT"
                  || finalFinishReason === "SPII"
                  || finalFinishReason === "IMAGE_SAFETY"
                  || finalFinishReason === "IMAGE_PROHIBITED_CONTENT"
                )
                ? "content_filter"
                : "stop";
          emit({}, finish);

          if (streamIncludesUsage(body)) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({
              id: `chatcmpl-${responseId}`,
              object: "chat.completion.chunk",
              created,
              model: offer.modelId,
              choices: [],
              usage: generateUsagePayload(finalUsage, offer.inputTokens),
            })}\n\n`));
          }
          controller.enqueue(sse("[DONE]"));
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });

    return new Response(output, {
      status: 200,
      headers: {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache",
      },
    });
  }

  private classifyFailure(offer: ProviderOffer, model: GoogleModel, error: unknown): ProviderExecutionResult {
    const code = statusCode(error);
    const headers = errorHeaders(error);
    const bodyText = errorBody(error);
    const retryAfterMs = parseRetryAfterMs(headers?.get("retry-after") ?? null, DEFAULT_RETRY_MS);
    const failedAt = Date.now();

    if (code === 429) {
      const dailyOverflowConfirmed = offer.kind === "overflow" && confirmsDailyQuota(error);
      if (dailyOverflowConfirmed) {
        const resetAt = calendarDayBounds(failedAt, GOOGLE_DAY.timeZone).end;
        this.modelOverflowBlockedUntil.set(model.id, resetAt);
        log("warn", "overflow_boundary_confirmed", {
          provider: this.id,
          relay_model: model.id,
          limit: "requests_per_day",
          reset_at: new Date(resetAt).toISOString(),
        });
      }
      const reason = dailyOverflowConfirmed ? "overflow_limit_confirmed" : "rate_limit";
      return {
        status: "retryable",
        scope: "model",
        reason,
        retryAt: this.blockModel(model, retryAfterMs, failedAt, reason),
      };
    }
    if (code === 408 || (code !== undefined && code >= 500)) {
      const reason = `upstream_${code}`;
      return {
        status: "retryable",
        scope: "provider",
        reason,
        retryAt: this.blockProvider(retryAfterMs, failedAt, reason),
      };
    }
    if (code === 401 || code === 403) {
      this.providerState.blockedUntil = Math.max(this.providerState.blockedUntil, failedAt + 60_000);
      log("warn", "provider_cooldown", {
        provider: this.id,
        reason: `upstream_${code}`,
        blocked_until: new Date(this.providerState.blockedUntil).toISOString(),
      });
      return { status: "rejected", scope: "provider", httpStatus: code, bodyText };
    }
    if (code !== undefined) {
      return { status: "rejected", scope: "model", httpStatus: code, bodyText };
    }

    const reason = error instanceof Error ? `network:${error.message}` : "network_error";
    return {
      status: "retryable",
      scope: "provider",
      reason,
      retryAt: this.blockProvider(DEFAULT_RETRY_MS, failedAt, reason),
    };
  }

  async execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    stream: boolean,
    signal: AbortSignal,
  ): Promise<ProviderExecutionResult> {
    if (!this.isConfigured()) {
      return { status: "retryable", scope: "provider", reason: "provider_not_configured", retryAt: Number.POSITIVE_INFINITY };
    }

    const model = this.modelById(offer.modelId);
    reserveQuota(model.quota, this.modelState(model.id), offer.inputTokens, Date.now());

    try {
      const plan = this.requestPlan(body, model);
      if (plan.bootstrapAntigravity || plan.replayExternalToolHistory) {
        log("info", "google_history_recovery", {
          provider: this.id,
          relay_model: model.id,
          mode: plan.bootstrapAntigravity ? "antigravity_transcript" : "generate_content_replay",
        });
      }
      const target = plan.target;
      const continuation = plan.continuation;
      const client = this.googleClient();
      let response: Response;

      if (plan.transport === "generate-content") {
        const request: GoogleGenerateContentRequest = toGoogleGenerateContentRequest(
          plan.body,
          model.upstreamModel,
          continuation?.generateContents ? {
            contents: continuation.generateContents,
            inputStartIndex: continuation.inputStartIndex,
          } : undefined,
          plan.replayExternalToolHistory,
        );
        request.config = { ...request.config, abortSignal: signal };

        if (stream) {
          const result = await client.models.generateContentStream(request as any);
          response = this.openAIGenerateStream(
            result as unknown as AsyncIterable<unknown>,
            offer,
            body,
            request.contents,
            target,
          );
        } else {
          const result = await client.models.generateContent(request as any);
          this.rememberGenerateContinuation(body, result, request.contents, target);
          response = new Response(JSON.stringify(googleGenerateContentToOpenAI(result, offer.modelId, offer.inputTokens)), {
            status: 200,
            headers: { "content-type": "application/json; charset=utf-8" },
          });
        }
      } else {
        const request: GoogleInteractionRequest = toGoogleInteractionRequest(
          plan.body,
          model.upstreamModel,
          stream,
          continuation?.interactionId ? {
            previousInteractionId: continuation.interactionId,
            inputStartIndex: continuation.inputStartIndex,
          } : undefined,
        );
        const interactionRequest = model.upstreamAgent
          ? antigravityRequest(request, model, offer.inputTokens)
          : request;
        const result = await client.interactions.create(interactionRequest as any, { fetchOptions: { signal } } as any);

        if (stream) {
          response = this.openAIStream(result as unknown as AsyncIterable<unknown>, offer, body, target);
        } else {
          const interaction = result as unknown;
          const value = object(interaction);
          if (value?.status === "failed" || value?.status === "cancelled") {
            this.release(model);
            return {
              status: "rejected",
              scope: "model",
              httpStatus: 502,
              bodyText: JSON.stringify({ error: value.errors ?? { message: `Google interaction ${value.status}.` } }),
            };
          }
          this.rememberContinuation(body, interaction, target);
          response = new Response(JSON.stringify(googleInteractionToOpenAI(interaction, offer.modelId, offer.inputTokens)), {
            status: 200,
            headers: { "content-type": "application/json; charset=utf-8" },
          });
        }
      }

      const providerFailures = this.providerFailureCount;
      const modelFailures = this.modelFailureCounts.get(model.id) ?? 0;
      this.providerFailureCount = 0;
      this.modelFailureCounts.set(model.id, 0);
      if (providerFailures > 0 || modelFailures > 0) {
        log("info", "provider_recovered", {
          provider: this.id,
          relay_model: model.id,
          provider_failures: providerFailures,
          model_failures: modelFailures,
        });
      }
      let released = false;
      return {
        status: "success",
        response,
        release: () => {
          if (released) return;
          released = true;
          this.release(model);
        },
      };
    } catch (error) {
      this.release(model);
      return this.classifyFailure(offer, model, error);
    }
  }

  status(now = Date.now()): ProviderStatus {
    this.pruneOverflowBlocks(Date.now());
    return {
      id: this.id,
      configured: this.isConfigured(),
      blockedUntil: this.providerState.blockedUntil > now ? this.providerState.blockedUntil : null,
      models: this.models.map((model) => {
        const state = this.modelState(model.id);
        return {
          id: model.id,
          active: state.active,
          blockedUntil: state.blockedUntil > now ? state.blockedUntil : null,
          overflowBlockedUntil: (this.modelOverflowBlockedUntil.get(model.id) ?? 0) > now
            ? this.modelOverflowBlockedUntil.get(model.id)!
            : null,
        };
      }),
    };
  }
}

export function createGoogleProvider(): GoogleProvider {
  return new GoogleProvider();
}