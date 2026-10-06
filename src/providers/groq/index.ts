import { log } from "../../log.ts";
import type { ChatCompletionRequest } from "../../types.ts";
import {
  effectiveInputCapacity,
  emptyQuotaState,
  parseRetryAfterMs,
  pruneQuotaEvents,
  quotaDelayMs,
  releaseQuota,
  reserveQuota,
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
  ProviderUsage,
} from "../shared/types.ts";
import { observeSseResponse, type ObservedSseResponse } from "../shared/sse.ts";
import { countGroqInputTokens, GROQ_QWEN_IMAGE_TOKENS, type GroqTokenizerSpec } from "./token-count.ts";

const GROQ_BASE_URL = "https://api.groq.com/openai/v1";
const DAY_MS = 86_400_000;
const DEFAULT_RETRY_MS = 5_000;
const PROVIDER_FAILURE_COOLDOWN_MS = 15_000;

export interface GroqModel {
  id: string;
  upstreamModel: string;
  contextWindowTokens: number;
  maxOutputTokens: number;
  preference: number;
  vision: boolean;
  parallelToolCalls: boolean;
  reasoningEfforts: readonly string[];
  quota: QuotaPolicy;
  tokensPerDay: number;
  tokenizer: GroqTokenizerSpec;
}

function freeQuota(): QuotaPolicy {
  return {
    maxConcurrent: null,
    dailyWindow: { type: "rolling" },
    limits: {
      requestsPerMinute: 30,
      inputTokensPerMinute: 8_000,
      requestsPerDay: 1_000,
      minimumSpacingMs: 0,
    },
  };
}

// Current Groq Free-plan general Chat Completions catalog, verified 2026-10-03.
export const GROQ_MODELS: readonly GroqModel[] = [
  {
    id: "groq/openai/gpt-oss-120b",
    upstreamModel: "openai/gpt-oss-120b",
    contextWindowTokens: 131_072,
    maxOutputTokens: 65_536,
    preference: 300,
    vision: false,
    parallelToolCalls: false,
    reasoningEfforts: ["low", "medium", "high"],
    quota: freeQuota(),
    tokensPerDay: 200_000,
    tokenizer: { kind: "gpt-oss" },
  },
  {
    id: "groq/qwen/qwen3.8-27b",
    upstreamModel: "qwen/qwen3.8-27b",
    contextWindowTokens: 131_072,
    maxOutputTokens: 16_384,
    preference: 200,
    vision: true,
    parallelToolCalls: false,
    reasoningEfforts: ["none", "default", "low", "medium", "high"],
    quota: freeQuota(),
    tokensPerDay: 200_000,
    tokenizer: {
      kind: "huggingface",
      repository: "Qwen/Qwen3.8-27B",
      visionImageTokens: GROQ_QWEN_IMAGE_TOKENS,
    },
  },
  {
    id: "groq/openai/gpt-oss-20b",
    upstreamModel: "openai/gpt-oss-20b",
    contextWindowTokens: 131_072,
    maxOutputTokens: 65_536,
    preference: 100,
    vision: false,
    parallelToolCalls: false,
    reasoningEfforts: ["low", "medium", "high"],
    quota: freeQuota(),
    tokensPerDay: 200_000,
    tokenizer: { kind: "gpt-oss" },
  },
];

interface GroqState extends QuotaRuntimeState {
  failures: number;
}

interface Candidate {
  model: GroqModel;
  inputTokens: number;
  inputCapacityTokens: number;
  availableAt: number;
  index: number;
}

type JsonObject = Record<string, unknown>;

function object(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function imageCount(body: ChatCompletionRequest): number {
  if (!Array.isArray(body.messages)) return 0;
  let count = 0;
  for (const rawMessage of body.messages) {
    const message = object(rawMessage);
    if (!message || !Array.isArray(message.content)) continue;
    for (const rawPart of message.content) {
      const part = object(rawPart);
      if (part?.type === "image_url" || part?.type === "image") count += 1;
    }
  }
  return count;
}

function knownUnsupported(body: ChatCompletionRequest): boolean {
  if (!Array.isArray(body.messages)) return false;
  if (body.logprobs != null || body.logit_bias != null || body.top_logprobs != null || body.metadata != null) return true;
  if (body.frequency_penalty != null && body.frequency_penalty !== 0) return true;
  if (body.presence_penalty != null && body.presence_penalty !== 0) return true;
  if (body.n != null && body.n !== 1) return true;
  if (body.store != null) return true;

  // These Groq Chat Completions extensions are either tied to Compound/search
  // or require request rendering that this provider cannot yet count exactly.
  if (
    body.documents !== undefined
    || body.chat_template_kwargs !== undefined
    || body.compound_custom !== undefined
    || body.search_settings !== undefined
    || body.include_domains !== undefined
    || body.exclude_domains !== undefined
    || body.functions !== undefined
    || body.function_call !== undefined
  ) return true;

  if (body.include_reasoning !== undefined && body.reasoning_format !== undefined) return true;
  if (body.service_tier != null && body.service_tier !== "auto" && body.service_tier !== "on_demand") return true;
  if (body.messages.some((message) => object(message)?.name !== undefined)) return true;
  if (body.tools !== undefined && (!Array.isArray(body.tools) || body.tools.some((tool) => object(tool)?.type !== "function"))) return true;
  return false;
}

function requestedOutputTokens(body: ChatCompletionRequest): number | null {
  const raw = body.max_completion_tokens ?? body.max_tokens;
  if (raw == null) return null;
  return typeof raw === "number" && Number.isSafeInteger(raw) && raw > 0 ? raw : Number.NaN;
}

function supports(model: GroqModel, body: ChatCompletionRequest): boolean {
  if (knownUnsupported(body)) return false;
  const images = imageCount(body);
  if (images > 0 && (!model.vision || images > 3)) return false;

  const outputTokens = requestedOutputTokens(body);
  if (outputTokens !== null && (!Number.isFinite(outputTokens) || outputTokens > model.maxOutputTokens)) return false;

  const effort = body.reasoning_effort;
  if (effort != null && (typeof effort !== "string" || !model.reasoningEfforts.includes(effort))) return false;

  if (
    body.parallel_tool_calls === true
    && Array.isArray(body.tools)
    && body.tools.length > 0
    && !model.parallelToolCalls
  ) return false;
  return true;
}

function dayTokenDelay(model: GroqModel, state: GroqState, inputTokens: number, now: number): number {
  state.events = pruneQuotaEvents(state.events, now, statePolicy(model).dailyWindow);
  if (inputTokens > model.tokensPerDay) return Number.POSITIVE_INFINITY;
  const recent = state.events.filter((event) => event.at > now - DAY_MS).sort((a, b) => a.at - b.at);
  let total = recent.reduce((sum, event) => sum + event.inputTokens, 0);
  if (total + inputTokens <= model.tokensPerDay) return 0;
  for (const event of recent) {
    total -= event.inputTokens;
    if (total + inputTokens <= model.tokensPerDay) return Math.max(1, event.at + DAY_MS - now);
  }
  return Number.POSITIVE_INFINITY;
}

function statePolicy(model: GroqModel): QuotaPolicy {
  return model.quota;
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

function durationMs(value: string | null): number | undefined {
  if (!value) return undefined;
  const text = value.trim().toLowerCase();
  const numeric = Number(text);
  if (Number.isFinite(numeric) && numeric >= 0) return Math.ceil(numeric * 1_000);
  const regex = /(\d+(?:\.\d+)?)(ms|h|m|s)/g;
  let total = 0;
  let consumed = "";
  for (const match of text.matchAll(regex)) {
    const amount = Number(match[1]);
    const unit = match[2];
    if (!unit || !Number.isFinite(amount)) return undefined;
    consumed += match[0];
    total += unit === "h" ? amount * 3_600_000 : unit === "m" ? amount * 60_000 : unit === "s" ? amount * 1_000 : amount;
  }
  return consumed === text ? Math.max(1, Math.ceil(total)) : undefined;
}

function errorCode(bodyText: string): string | undefined {
  try {
    const error = object(object(JSON.parse(bodyText) as unknown)?.error);
    return typeof error?.code === "string" ? error.code : undefined;
  } catch {
    return undefined;
  }
}

interface Usage {
  prompt?: number;
  completion?: number;
  total?: number;
  cached?: number;
}

function usage(payload: unknown): Usage | undefined {
  const value = object(object(payload)?.usage);
  if (!value) return undefined;
  const details = object(value.prompt_tokens_details);
  return {
    ...(typeof value.prompt_tokens === "number" ? { prompt: value.prompt_tokens } : {}),
    ...(typeof value.completion_tokens === "number" ? { completion: value.completion_tokens } : {}),
    ...(typeof value.total_tokens === "number" ? { total: value.total_tokens } : {}),
    ...(typeof details?.cached_tokens === "number" ? { cached: details.cached_tokens } : {}),
  };
}

function providerUsage(parsed: Usage | undefined): ProviderUsage | undefined {
  if (!parsed) return undefined;
  return {
    ...(parsed.prompt !== undefined ? { inputTokens: parsed.prompt } : {}),
    ...(parsed.completion !== undefined ? { outputTokens: parsed.completion } : {}),
    ...(parsed.total !== undefined ? { totalTokens: parsed.total } : {}),
  };
}

export class GroqProvider implements Provider {
  readonly id = "groq";
  readonly priority = 30;
  private readonly models: readonly GroqModel[];
  private readonly states = new Map<string, GroqState>();
  private readonly tokenCountCache = new WeakMap<ChatCompletionRequest, Map<string, Promise<number>>>();
  private providerBlockedUntil = 0;
  private providerFailures = 0;

  constructor(models: readonly GroqModel[] = GROQ_MODELS) {
    this.models = models;
    for (const model of models) this.states.set(model.id, { ...emptyQuotaState(), failures: 0 });
  }

  isConfigured(): boolean {
    return Boolean(process.env.GROQ_API_KEY?.trim());
  }

  listModels(): readonly ProviderModelInfo[] {
    return this.models.map((model) => ({
      id: model.id,
      providerId: this.id,
      inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
    }));
  }

  private state(modelId: string): GroqState {
    const state = this.states.get(modelId);
    if (!state) throw new Error(`Unknown Groq model state: ${modelId}`);
    return state;
  }

  private model(modelId: string): GroqModel {
    const model = this.models.find((candidate) => candidate.id === modelId);
    if (!model) throw new Error(`Groq does not own model ${modelId}`);
    return model;
  }

  private async count(body: ChatCompletionRequest, model: GroqModel): Promise<number> {
    let perModel = this.tokenCountCache.get(body);
    if (!perModel) {
      perModel = new Map();
      this.tokenCountCache.set(body, perModel);
    }
    const cached = perModel.get(model.id);
    if (cached) return cached;
    const pending = countGroqInputTokens(body, model.tokenizer).catch((error) => {
      perModel!.delete(model.id);
      throw error;
    });
    perModel.set(model.id, pending);
    return pending;
  }

  async getBestOffer(request: OfferRequest, now = Date.now()): Promise<ProviderOfferResult> {
    if (!this.isConfigured()) return { status: "no_offer", providerId: this.id, reason: "provider_not_configured" };
    if (request.offerKind === "overflow") return { status: "no_offer", providerId: this.id, reason: "no_eligible_model" };

    const eligible = this.models
      .map((model, index) => ({ model, index }))
      .filter(({ model }) => !request.excludedModelIds.has(model.id))
      .filter(({ model }) => request.requestedModel === "auto" || request.requestedModel === model.id)
      .filter(({ model }) => supports(model, request.body));
    if (eligible.length === 0) return { status: "no_offer", providerId: this.id, reason: "no_eligible_model" };

    const results = await Promise.all(eligible.map(async ({ model, index }) => {
      try {
        const inputTokens = await this.count(request.body, model);
        const capacity = effectiveInputCapacity(model.quota, model.contextWindowTokens);
        if (inputTokens > capacity || inputTokens > model.tokensPerDay) return { candidate: null, error: undefined };
        const evaluatedAt = Math.max(now, Date.now());
        const state = this.state(model.id);
        const delay = Math.max(
          quotaDelayMs(model.quota, state, inputTokens, evaluatedAt),
          dayTokenDelay(model, state, inputTokens, evaluatedAt),
          Math.max(0, this.providerBlockedUntil - evaluatedAt),
        );
        return {
          candidate: {
            model,
            inputTokens,
            inputCapacityTokens: capacity,
            availableAt: Number.isFinite(delay) ? (delay <= 0 ? now : evaluatedAt + delay) : Number.POSITIVE_INFINITY,
            index,
          } satisfies Candidate,
          error: undefined,
        };
      } catch (error) {
        return { candidate: null, error };
      }
    }));

    const candidates = results.map((result) => result.candidate).filter((value): value is Candidate => value !== null);
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
      return { status: "no_offer", providerId: this.id, reason: "request_exceeds_capacity" };
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

  private blockModel(model: GroqModel, requestedMs: number, now: number, reason: string): number {
    const state = this.state(model.id);
    state.failures += 1;
    const delay = Math.min(60_000, Math.max(DEFAULT_RETRY_MS, requestedMs) * (2 ** Math.min(state.failures - 1, 4)));
    state.blockedUntil = Math.max(state.blockedUntil, now + delay);
    log("warn", "provider_model_cooldown", { provider: this.id, relay_model: model.id, reason, blocked_until: new Date(state.blockedUntil).toISOString() });
    return state.blockedUntil;
  }

  private blockProvider(requestedMs: number, now: number, reason: string): number {
    this.providerFailures += 1;
    const delay = Math.min(120_000, Math.max(PROVIDER_FAILURE_COOLDOWN_MS, requestedMs) * (2 ** Math.min(this.providerFailures - 1, 4)));
    this.providerBlockedUntil = Math.max(this.providerBlockedUntil, now + delay);
    log("warn", "provider_cooldown", { provider: this.id, reason, blocked_until: new Date(this.providerBlockedUntil).toISOString() });
    return this.providerBlockedUntil;
  }

  private applyHeaders(model: GroqModel, headers: Headers, now: number): void {
    const state = this.state(model.id);
    for (const [remainingName, resetName] of [
      ["x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"],
      ["x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"],
    ] as const) {
      const remaining = headers.get(remainingName);
      if (remaining === null || Number(remaining) !== 0) continue;
      const wait = durationMs(headers.get(resetName));
      if (wait !== undefined) state.blockedUntil = Math.max(state.blockedUntil, now + wait);
    }
  }

  private reconcile(state: GroqState, event: { at: number; inputTokens: number }, offer: ProviderOffer, parsed: Usage | undefined): ProviderUsage | undefined {
    if (!parsed) return undefined;
    if (parsed.prompt !== undefined && parsed.prompt !== offer.inputTokens) {
      log("warn", "groq_token_count_mismatch", { provider: this.id, relay_model: offer.modelId, local_input_tokens: offer.inputTokens, upstream_input_tokens: parsed.prompt });
    }
    const charged = Math.max(0, (parsed.prompt ?? offer.inputTokens) - (parsed.cached ?? 0)) + (parsed.completion ?? 0);
    if (Number.isSafeInteger(charged) && charged >= 0) event.inputTokens = charged;
    return providerUsage(parsed);
  }

  private observeJson(response: Response, state: GroqState, event: { at: number; inputTokens: number }, offer: ProviderOffer): Promise<ProviderUsage | undefined> {
    return response.json().then((payload: unknown) => this.reconcile(state, event, offer, usage(payload))).catch(() => undefined);
  }

  private observeStream(
    response: Response,
    state: GroqState,
    event: { at: number; inputTokens: number },
    offer: ProviderOffer,
    includeUsage: boolean,
  ): ObservedSseResponse<ProviderUsage> {
    return observeSseResponse(response, (data) => {
      if (data === "[DONE]") return;
      try {
        const payload = JSON.parse(data) as unknown;
        const parsed = usage(payload);
        if (!parsed) return;

        const value = this.reconcile(state, event, offer, parsed);
        if (!value) return;
        const choices = object(payload)?.choices;
        const usageOnly = Array.isArray(choices) && choices.length === 0;
        return {
          value,
          forward: includeUsage || !usageOnly,
        };
      } catch {
        // Preserve non-JSON SSE events unchanged.
        return;
      }
    });
  }

  async execute(offer: ProviderOffer, body: ChatCompletionRequest, stream: boolean, signal: AbortSignal): Promise<ProviderExecutionResult> {
    const apiKey = process.env.GROQ_API_KEY?.trim();
    if (!apiKey) return { status: "retryable", scope: "provider", reason: "provider_not_configured", retryAt: Number.POSITIVE_INFINITY };

    const model = this.model(offer.modelId);
    const state = this.state(model.id);
    reserveQuota(model.quota, state, offer.inputTokens, Date.now());
    const event = state.events[state.events.length - 1]!;
    const streamOptions = object(body.stream_options) ?? {};

    let response: Response;
    try {
      response = await fetch(`${GROQ_BASE_URL}/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${apiKey}`,
          "content-type": "application/json",
          accept: stream ? "text/event-stream, application/json" : "application/json",
          "user-agent": "ai-relay/1.0",
        },
        body: JSON.stringify({
          ...body,
          model: model.upstreamModel,
          stream,
          ...(Array.isArray(body.tools) && body.tools.length > 0 && !model.parallelToolCalls
            ? { parallel_tool_calls: false }
            : {}),
          ...(stream ? { stream_options: { ...streamOptions, include_usage: true } } : {}),
        }),
        signal,
      });
    } catch (error) {
      releaseQuota(state);
      const reason = error instanceof Error ? `network:${error.message}` : "network_error";
      return { status: "retryable", scope: "provider", reason, retryAt: this.blockProvider(DEFAULT_RETRY_MS, Date.now(), reason) };
    }

    const respondedAt = Date.now();
    this.applyHeaders(model, response.headers, respondedAt);
    if (response.ok) {
      const previousProviderFailures = this.providerFailures;
      const previousModelFailures = state.failures;
      this.providerFailures = 0;
      state.failures = 0;
      if (previousProviderFailures > 0 || previousModelFailures > 0) {
        log("info", "provider_recovered", { provider: this.id, relay_model: model.id, provider_failures: previousProviderFailures, model_failures: previousModelFailures });
      }
      const clientWantsUsage = streamOptions.include_usage === true;
      const streaming = stream
        ? this.observeStream(response, state, event, offer, clientWantsUsage)
        : undefined;
      const usagePromise = (streaming?.observed
        ?? this.observeJson(response.clone(), state, event, offer))
        .catch(() => undefined);
      let released = false;
      return {
        status: "success",
        response: streaming?.response ?? response,
        usage: usagePromise,
        release: () => {
          if (released) return;
          released = true;
          releaseQuota(state);
        },
      };
    }

    const bodyText = await response.text().catch(() => "");
    releaseQuota(state);
    const retryMs = parseRetryAfterMs(response.headers.get("retry-after"), DEFAULT_RETRY_MS);
    const code = errorCode(bodyText);
    if (response.status === 429) {
      return { status: "retryable", scope: "model", reason: "rate_limit", retryAt: this.blockModel(model, retryMs, respondedAt, "rate_limit") };
    }
    if (response.status === 498 || code === "capacity_exceeded") {
      return { status: "retryable", scope: "model", reason: "capacity_exceeded", retryAt: this.blockModel(model, retryMs, respondedAt, "capacity_exceeded") };
    }
    if (response.status === 408 || response.status >= 500) {
      const reason = `upstream_${response.status}`;
      return { status: "retryable", scope: "provider", reason, retryAt: this.blockProvider(retryMs, respondedAt, reason) };
    }
    if (response.status === 401 || response.status === 403 || code === "blocked_api_access") {
      this.providerBlockedUntil = Math.max(this.providerBlockedUntil, respondedAt + 60_000);
      return { status: "rejected", scope: "provider", httpStatus: response.status, bodyText };
    }
    return { status: "rejected", scope: "model", httpStatus: response.status, bodyText };
  }

  status(now = Date.now()): ProviderStatus {
    return {
      id: this.id,
      configured: this.isConfigured(),
      blockedUntil: this.providerBlockedUntil > now ? this.providerBlockedUntil : null,
      models: this.models.map((model) => {
        const state = this.state(model.id);
        state.events = pruneQuotaEvents(state.events, now, model.quota.dailyWindow);
        return {
          id: model.id,
          active: state.active,
          blockedUntil: state.blockedUntil > now ? state.blockedUntil : null,
          overflowBlockedUntil: null,
        };
      }),
    };
  }
}

export function createGroqProvider(): Provider {
  return new GroqProvider();
}
