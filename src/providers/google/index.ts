import { GoogleGenAI, type Content } from "@google/genai";
import type { ChatCompletionRequest } from "../../types.ts";
import {
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
  ProviderStatus,
} from "../shared/types.ts";
import {
  toGoogleCountInput,
  toGoogleGenerateContentRequest,
  toGoogleInteractionRequest,
  type GoogleGenerateContentRequest,
  type GoogleInteractionRequest,
} from "./token-count.ts";

const GOOGLE_DAY = { type: "calendar-day", timeZone: "America/Los_Angeles" } as const;
const DEFAULT_RETRY_MS = 5_000;
const PROVIDER_FAILURE_COOLDOWN_MS = 15_000;
const OVERFLOW_HARD_CAP_TTL_MS = 24 * 60 * 60 * 1000;
const CONTINUATION_TTL_MS = 60 * 60 * 1000;
const MAX_CONTINUATIONS = 1_000;
const OVERFLOW_LIMITS = new Set<QuotaLimitName>(["requestsPerDay"]);

export type GoogleTransport = "interactions" | "generate-content";
export type GoogleThinkingLevel = "minimal" | "low" | "medium" | "high";

export interface GoogleModel {
  id: string;
  upstreamModel: string;
  contextWindowTokens: number;
  quota: QuotaPolicy;
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
  upstreamModel: string;
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
  { id: "google/gemma-4-26b-a4b-it", upstreamModel: "gemma-4-26b-a4b-it", contextWindowTokens: 262_144, quota: quota(30, 16_000, 14_400), transport: "interactions" },
  { id: "google/gemma-4-31b-it", upstreamModel: "gemma-4-31b-it", contextWindowTokens: 262_144, quota: quota(30, 16_000, 14_400), transport: "interactions" },
  { id: "google/gemini-robotics-er-2-preview", upstreamModel: "gemini-robotics-er-2-preview", contextWindowTokens: 131_072, quota: quota(5, 250_000, 20), transport: "generate-content", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-3.5-flash-lite", upstreamModel: "gemini-3.5-flash-lite", contextWindowTokens: 1_048_576, quota: quota(15, 250_000, 500), transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-3.1-flash-lite", upstreamModel: "gemini-3.1-flash-lite", contextWindowTokens: 1_048_576, quota: quota(15, 250_000, 500), transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-2.5-flash-lite", upstreamModel: "gemini-2.5-flash-lite", contextWindowTokens: 1_048_576, quota: quota(10, 250_000, 20), transport: "interactions", thinkingLevels: NO_MINIMAL_THINKING },
  { id: "google/gemini-3.8-flash", upstreamModel: "gemini-3.8-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), transport: "interactions", thinkingLevels: NO_MINIMAL_THINKING },
  { id: "google/gemini-3.7-flash", upstreamModel: "gemini-3.7-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), transport: "interactions", thinkingLevels: NO_MINIMAL_THINKING },
  { id: "google/gemini-3.6-flash", upstreamModel: "gemini-3.6-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-3.5-flash", upstreamModel: "gemini-3.5-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-3-flash-preview", upstreamModel: "gemini-3-flash-preview", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), transport: "interactions", thinkingLevels: ALL_THINKING },
  { id: "google/gemini-2.5-flash", upstreamModel: "gemini-2.5-flash", contextWindowTokens: 1_048_576, quota: quota(5, 250_000, 20), transport: "interactions", thinkingLevels: NO_MINIMAL_THINKING },
];

function object(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function compareCapacity(left: Candidate, right: Candidate): number {
  return left.inputCapacityTokens - right.inputCapacityTokens
    || left.availableAt - right.availableAt
    || left.index - right.index;
}

function compareAvailability(left: Candidate, right: Candidate): number {
  return left.availableAt - right.availableAt
    || left.inputCapacityTokens - right.inputCapacityTokens
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

function modelSupportsRequest(model: GoogleModel, body: ChatCompletionRequest): boolean {
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

  constructor(
    clientFactory: (apiKey: string) => GoogleGenAI = (apiKey) => new GoogleGenAI({ apiKey }),
    models: readonly GoogleModel[] = GOOGLE_MODELS,
  ) {
    this.clientFactory = clientFactory;
    this.models = models;
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

  private googleClient(): GoogleGenAI {
    const apiKey = process.env.GEMINI_API_KEY?.trim();
    if (!apiKey) throw new Error("GEMINI_API_KEY is not configured.");
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

  async countInputTokens(body: ChatCompletionRequest, modelId: string): Promise<number> {
    const model = this.modelById(modelId);
    let perModel = this.tokenCountCache.get(body);
    if (!perModel) {
      perModel = new Map<string, Promise<number>>();
      this.tokenCountCache.set(body, perModel);
    }
    const cached = perModel.get(modelId);
    if (cached) return cached;

    const pending = Promise.resolve().then(async () => {
      const input = toGoogleCountInput(body);
      let contents = input.contents;
      if (transportFor(model) === "generate-content") {
        const continuation = this.continuationFor(body, model.upstreamModel);
        const request = toGoogleGenerateContentRequest(
          body,
          model.upstreamModel,
          continuation?.generateContents ? {
            contents: continuation.generateContents,
            inputStartIndex: continuation.inputStartIndex,
          } : undefined,
        );
        contents = request.contents;
      }
      const response = await this.googleClient().models.countTokens({
        model: model.upstreamModel,
        contents,
        config: {
          ...input.config,
          abortSignal: AbortSignal.timeout(10_000),
        },
      });
      if (!Number.isSafeInteger(response.totalTokens) || (response.totalTokens ?? -1) < 0) {
        throw new Error(`Google did not return a valid token count for ${model.upstreamModel}.`);
      }
      return response.totalTokens!;
    }).catch((error) => {
      perModel!.delete(modelId);
      throw error;
    });

    perModel.set(modelId, pending);
    return pending;
  }

  private async candidate(model: GoogleModel, index: number, request: OfferRequest, now: number): Promise<Candidate | null> {
    if (request.excludedModelIds.has(model.id)) return null;
    if (request.requestedModel !== "auto" && request.requestedModel !== model.id) return null;
    if (!modelSupportsRequest(model, request.body)) return null;

    const inputTokens = await this.countInputTokens(request.body, model.id);
    if (!quotaCanEverHandle(model.quota, inputTokens, model.contextWindowTokens)) return null;

    const evaluatedAt = Math.max(now, Date.now());
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
        availableAt: evaluatedAt,
        index,
      };
    }

    const delayMs = Math.max(modelDelayMs, providerBlockedMs);
    return {
      model,
      inputTokens,
      inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
      availableAt: Number.isFinite(delayMs) ? evaluatedAt + Math.max(0, delayMs) : Number.POSITIVE_INFINITY,
      index,
    };
  }

  async getBestOffer(request: OfferRequest, now = Date.now()): Promise<ProviderOffer | null> {
    if (!this.isConfigured()) return null;
    const results = await Promise.all(this.models.map(async (model, index) => {
      try { return { candidate: await this.candidate(model, index, request, now), error: undefined }; }
      catch (error) { return { candidate: null, error }; }
    }));
    const candidates = results.map((result) => result.candidate).filter((candidate): candidate is Candidate => candidate !== null);
    if (candidates.length === 0) {
      const countingError = results.find((result) => result.error !== undefined)?.error;
      if (countingError !== undefined) throw countingError;
      return null;
    }

    const chosen = request.requestedModel !== "auto"
      ? candidates[0]!
      : (candidates.filter((candidate) => candidate.availableAt <= now + request.maxOptimizationWaitMs).sort(compareCapacity)[0]
        ?? [...candidates].sort(compareAvailability)[0]!);

    return {
      kind: request.offerKind,
      providerId: this.id,
      providerPriority: this.priority,
      modelId: chosen.model.id,
      inputTokens: chosen.inputTokens,
      inputCapacityTokens: chosen.inputCapacityTokens,
      availableAt: chosen.availableAt,
    };
  }

  private release(model: GoogleModel): void {
    releaseQuota(this.modelState(model.id));
  }

  private blockModel(model: GoogleModel, requestedDelayMs: number, now: number): number {
    const failures = (this.modelFailureCounts.get(model.id) ?? 0) + 1;
    this.modelFailureCounts.set(model.id, failures);
    const multiplier = 2 ** Math.min(failures - 1, 4);
    const delayMs = Math.min(60_000, Math.max(DEFAULT_RETRY_MS, requestedDelayMs) * multiplier);
    const retryAt = now + delayMs;
    const state = this.modelState(model.id);
    state.blockedUntil = Math.max(state.blockedUntil, retryAt);
    return retryAt;
  }

  private blockProvider(requestedDelayMs: number, now: number): number {
    this.providerFailureCount += 1;
    const multiplier = 2 ** Math.min(this.providerFailureCount - 1, 4);
    const delayMs = Math.min(120_000, Math.max(PROVIDER_FAILURE_COOLDOWN_MS, requestedDelayMs) * multiplier);
    const retryAt = now + delayMs;
    this.providerState.blockedUntil = Math.max(this.providerState.blockedUntil, retryAt);
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

  private continuationFor(body: ChatCompletionRequest, upstreamModel: string): Continuation | undefined {
    const messages = messageArray(body);
    if (!messages) return undefined;
    this.pruneContinuations();
    for (let length = messages.length; length > 0; length -= 1) {
      const continuation = this.continuations.get(prefixKey(messages.slice(0, length)));
      if (continuation?.upstreamModel === upstreamModel) return continuation;
    }
    return undefined;
  }

  private rememberContinuation(body: ChatCompletionRequest, interaction: unknown, upstreamModel: string): void {
    const messages = messageArray(body);
    const value = object(interaction);
    if (!messages || typeof value?.id !== "string") return;
    const assistant = assistantMessageFromInteraction(value);
    const continued = [...messages, assistant];
    this.continuations.delete(prefixKey(continued));
    this.continuations.set(prefixKey(continued), {
      interactionId: value.id,
      inputStartIndex: continued.length,
      upstreamModel,
      expiresAt: Date.now() + CONTINUATION_TTL_MS,
    });
    this.pruneContinuations();
  }

  private rememberGenerateContinuation(
    body: ChatCompletionRequest,
    response: unknown,
    requestContents: Content[],
    upstreamModel: string,
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
      upstreamModel,
      expiresAt: Date.now() + CONTINUATION_TTL_MS,
    });
    this.pruneContinuations();
  }

  private rememberGenerateStreamContinuation(
    body: ChatCompletionRequest,
    assistant: JsonObject,
    modelContent: Content,
    requestContents: Content[],
    upstreamModel: string,
  ): void {
    const messages = messageArray(body);
    if (!messages) return;
    const continued = [...messages, assistant];
    this.continuations.delete(prefixKey(continued));
    this.continuations.set(prefixKey(continued), {
      generateContents: [...requestContents, modelContent],
      inputStartIndex: continued.length,
      upstreamModel,
      expiresAt: Date.now() + CONTINUATION_TTL_MS,
    });
    this.pruneContinuations();
  }

  private openAIStream(
    stream: AsyncIterable<unknown>,
    offer: ProviderOffer,
    body: ChatCompletionRequest,
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
                provider.rememberContinuation(body, synthetic, provider.modelById(offer.modelId).upstreamModel);
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
    upstreamModel: string,
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
              upstreamModel,
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
      if (offer.kind === "overflow") {
        this.modelOverflowBlockedUntil.set(
          model.id,
          Math.max(this.modelOverflowBlockedUntil.get(model.id) ?? 0, failedAt + OVERFLOW_HARD_CAP_TTL_MS),
        );
      }
      return {
        status: "retryable",
        scope: "model",
        reason: offer.kind === "overflow" ? "overflow_limit_confirmed" : "rate_limit",
        retryAt: this.blockModel(model, retryAfterMs, failedAt),
      };
    }
    if (code === 408 || (code !== undefined && code >= 500)) {
      return {
        status: "retryable",
        scope: "provider",
        reason: `upstream_${code}`,
        retryAt: this.blockProvider(retryAfterMs, failedAt),
      };
    }
    if (code === 401 || code === 403) {
      this.providerState.blockedUntil = Math.max(this.providerState.blockedUntil, failedAt + 60_000);
      return { status: "rejected", scope: "provider", httpStatus: code, bodyText };
    }
    if (code !== undefined) {
      return { status: "rejected", scope: "model", httpStatus: code, bodyText };
    }

    return {
      status: "retryable",
      scope: "provider",
      reason: error instanceof Error ? `network:${error.message}` : "network_error",
      retryAt: this.blockProvider(DEFAULT_RETRY_MS, failedAt),
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
      const continuation = this.continuationFor(body, model.upstreamModel);
      const request: GoogleInteractionRequest = toGoogleInteractionRequest(
        body,
        model.upstreamModel,
        stream,
        continuation ? {
          previousInteractionId: continuation.interactionId,
          inputStartIndex: continuation.inputStartIndex,
        } : undefined,
      );
      const client = this.googleClient();
      const result = await client.interactions.create(request as any, { fetchOptions: { signal } } as any);

      this.providerFailureCount = 0;
      this.modelFailureCounts.set(model.id, 0);
      let response: Response;
      if (stream) {
        response = this.openAIStream(result as unknown as AsyncIterable<unknown>, offer, body);
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
        this.rememberContinuation(body, interaction, model.upstreamModel);
        response = new Response(JSON.stringify(googleInteractionToOpenAI(interaction, offer.modelId, offer.inputTokens)), {
          status: 200,
          headers: { "content-type": "application/json; charset=utf-8" },
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