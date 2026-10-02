import { log } from "../../log.ts";
import type { ChatCompletionRequest } from "../../types.ts";
import type {
  OfferRequest,
  Provider,
  ProviderExecutionResult,
  ProviderModelInfo,
  ProviderOffer,
  ProviderOfferResult,
  ProviderStatus,
} from "../shared/types.ts";
import { countNvidiaInputTokens, type NvidiaTokenizerSpec } from "./token-count.ts";

const NVIDIA_BASE_URL = "https://integrate.api.nvidia.com";
const DEFAULT_RETRY_MS = 5_000;
const PROVIDER_FAILURE_COOLDOWN_MS = 15_000;
const MAX_MODEL_RATE_LIMIT_BACKOFF_MS = 60_000;
const MAX_PROVIDER_FAILURE_BACKOFF_MS = 120_000;

export interface NvidiaModel {
  id: string;
  upstreamModel: string;
  contextWindowTokens: number | null;
  preference: number;
  enabled: boolean;
  tokenizer: NvidiaTokenizerSpec;
}

// General-purpose chat/text candidates from the NVIDIA free-endpoint catalog,
// ordered by the coding preference agreed for this relay. Only models whose
// transport, context limit and authoritative token counting have been verified
// should be enabled. Disabled upstream IDs are catalog slugs and must be
// verified against NVIDIA's API before those models are enabled.
export const NVIDIA_MODELS: readonly NvidiaModel[] = [
  {
    id: "nvidia/deepseek-ai/deepseek-v4.1-flash",
    upstreamModel: "deepseek-ai/deepseek-v4.1-flash",
    contextWindowTokens: null,
    preference: 1_500,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "deepseek-ai/DeepSeek-V4.1-Flash" },
  },
  {
    id: "nvidia/z-ai/glm-5-3",
    upstreamModel: "z-ai/glm-5-3",
    contextWindowTokens: null,
    preference: 1_400,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "zai-org/GLM-5.3" },
  },
  {
    id: "nvidia/moonshotai/kimi-k3",
    upstreamModel: "moonshotai/kimi-k3",
    contextWindowTokens: null,
    preference: 1_300,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "Xenova/Kimi-K3-tokenizer" },
  },
  {
    id: "nvidia/z-ai/glm-5-3-flash",
    upstreamModel: "z-ai/glm-5-3-flash",
    contextWindowTokens: null,
    preference: 1_200,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "zai-org/GLM-5.3-Flash" },
  },
  {
    id: "nvidia/nvidia/nemotron-3-ultra-550b-a55b",
    upstreamModel: "nvidia/nemotron-3-ultra-550b-a55b",
    contextWindowTokens: null,
    preference: 1_100,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4" },
  },
  {
    id: "nvidia/meta/muse-glimmer-30b",
    upstreamModel: "meta/muse-glimmer-30b",
    contextWindowTokens: null,
    preference: 1_000,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "meta-models/Muse-Glimmer-30B" },
  },
  {
    id: "nvidia/poolside/laguna-xs-2.1",
    upstreamModel: "poolside/laguna-xs-2.1",
    contextWindowTokens: null,
    preference: 900,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "poolside/Laguna-XS-2.1" },
  },
  {
    id: "nvidia/google/gemma-4-31b-it",
    upstreamModel: "google/gemma-4-31b-it",
    contextWindowTokens: null,
    preference: 800,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "nvidia/Gemma-4-31B-IT-NVFP4" },
  },
  {
    id: "nvidia/nvidia/nemotron-3-super-120b-a12b",
    upstreamModel: "nvidia/nemotron-3-super-120b-a12b",
    contextWindowTokens: null,
    preference: 700,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8" },
  },
  {
    id: "nvidia/google/diffusiongemma-26b-a4b-it",
    upstreamModel: "google/diffusiongemma-26b-a4b-it",
    contextWindowTokens: null,
    preference: 600,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "nvidia/diffusiongemma-26B-A4B-it-NVFP4" },
  },
  {
    id: "nvidia/nvidia/nemotron-3.5-lightning-30b-a3b",
    upstreamModel: "nvidia/nemotron-3.5-lightning-30b-a3b",
    contextWindowTokens: null,
    preference: 500,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" },
  },
  {
    id: "nvidia/openai/gpt-oss-20b",
    upstreamModel: "openai/gpt-oss-20b",
    contextWindowTokens: 131_072,
    preference: 400,
    enabled: true,
    tokenizer: { kind: "gpt-oss-20b" },
  },
  {
    id: "nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    upstreamModel: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    contextWindowTokens: null,
    preference: 300,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" },
  },
  {
    id: "nvidia/meta/llama-3.2-90b-vision-instruct",
    upstreamModel: "meta/llama-3.2-90b-vision-instruct",
    contextWindowTokens: null,
    preference: 200,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "alpindale/Llama-3.2-90B-Vision-Instruct" },
  },
  {
    id: "nvidia/meta/llama-3.2-11b-vision-instruct",
    upstreamModel: "meta/llama-3.2-11b-vision-instruct",
    contextWindowTokens: null,
    preference: 100,
    enabled: false,
    tokenizer: { kind: "huggingface", repository: "alpindale/Llama-3.2-11B-Vision-Instruct" },
  },
];

interface NvidiaModelState {
  active: number;
  blockedUntil: number;
  consecutiveRateLimits: number;
}

interface Candidate {
  model: NvidiaModel & { contextWindowTokens: number };
  inputTokens: number;
  availableAt: number;
  index: number;
}

function enabledModel(model: NvidiaModel): model is NvidiaModel & { contextWindowTokens: number } {
  return model.enabled && typeof model.contextWindowTokens === "number" && model.contextWindowTokens > 0;
}

function parseRetryAfterMs(value: string | null, fallbackMs: number): number {
  if (!value) return fallbackMs;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return Math.max(1_000, Math.ceil(seconds * 1000));
  const date = Date.parse(value);
  if (Number.isFinite(date)) return Math.max(1_000, date - Date.now());
  return fallbackMs;
}

function compareCapacity(left: Candidate, right: Candidate): number {
  return left.model.contextWindowTokens - right.model.contextWindowTokens
    || left.availableAt - right.availableAt
    || right.model.preference - left.model.preference
    || left.index - right.index;
}

function compareAvailability(left: Candidate, right: Candidate): number {
  return left.availableAt - right.availableAt
    || left.model.contextWindowTokens - right.model.contextWindowTokens
    || right.model.preference - left.model.preference
    || left.index - right.index;
}

class NvidiaProvider implements Provider {
  readonly id = "nvidia";
  readonly priority = 20;
  private readonly models = NVIDIA_MODELS.filter(enabledModel);
  private readonly modelStates = new Map<string, NvidiaModelState>();
  private readonly tokenCountCache = new WeakMap<ChatCompletionRequest, Map<string, Promise<number>>>();
  private providerBlockedUntil = 0;
  private providerFailureCount = 0;

  constructor() {
    for (const model of this.models) {
      this.modelStates.set(model.id, { active: 0, blockedUntil: 0, consecutiveRateLimits: 0 });
    }
  }

  isConfigured(): boolean {
    return Boolean(process.env.NVIDIA_API_KEY?.trim());
  }

  listModels(): readonly ProviderModelInfo[] {
    return this.models.map((model) => ({
      id: model.id,
      providerId: this.id,
      inputCapacityTokens: model.contextWindowTokens,
    }));
  }

  private modelById(modelId: string): NvidiaModel & { contextWindowTokens: number } {
    const model = this.models.find((candidate) => candidate.id === modelId);
    if (!model) throw new Error(`NVIDIA does not own an enabled model ${modelId}`);
    return model;
  }

  private modelState(modelId: string): NvidiaModelState {
    const state = this.modelStates.get(modelId);
    if (!state) throw new Error(`Unknown NVIDIA model state: ${modelId}`);
    return state;
  }

  private async countInputTokens(body: ChatCompletionRequest, model: NvidiaModel): Promise<number> {
    let perModel = this.tokenCountCache.get(body);
    if (!perModel) {
      perModel = new Map<string, Promise<number>>();
      this.tokenCountCache.set(body, perModel);
    }
    const cached = perModel.get(model.id);
    if (cached) return cached;

    const pending = countNvidiaInputTokens(body, model.tokenizer)
      .then((count) => {
        if (!Number.isSafeInteger(count) || count < 0) {
          throw new Error(`NVIDIA returned an invalid token count for ${model.id}: ${count}`);
        }
        return count;
      })
      .catch((error) => {
        perModel!.delete(model.id);
        throw error;
      });
    perModel.set(model.id, pending);
    return pending;
  }

  async getBestOffer(request: OfferRequest, now = Date.now()): Promise<ProviderOfferResult> {
    if (!this.isConfigured()) {
      return { status: "no_offer", providerId: this.id, reason: "provider_not_configured" };
    }
    if (request.offerKind === "overflow") {
      return {
        status: "no_offer",
        providerId: this.id,
        reason: "no_eligible_model",
        detail: "NVIDIA overflow behavior is not implemented because no overflow limit has been observed and verified.",
      };
    }

    const eligibleModels = this.models
      .map((model, index) => ({ model, index }))
      .filter(({ model }) => !request.excludedModelIds.has(model.id))
      .filter(({ model }) => request.requestedModel === "auto" || request.requestedModel === model.id);

    if (eligibleModels.length === 0) {
      return { status: "no_offer", providerId: this.id, reason: "no_eligible_model" };
    }

    const results = await Promise.all(eligibleModels.map(async ({ model, index }) => {
      try {
        const inputTokens = await this.countInputTokens(request.body, model);
        if (inputTokens > model.contextWindowTokens) return { candidate: null, error: undefined };
        const evaluatedAt = Math.max(now, Date.now());
        const state = this.modelState(model.id);
        return {
          candidate: {
            model,
            inputTokens,
            availableAt: Math.max(evaluatedAt, state.blockedUntil, this.providerBlockedUntil),
            index,
          } satisfies Candidate,
          error: undefined,
        };
      } catch (error) {
        return { candidate: null, error };
      }
    }));

    const candidates = results
      .map((result) => result.candidate)
      .filter((candidate): candidate is Candidate => candidate !== null);
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

    let chosen: Candidate;
    if (request.requestedModel !== "auto") {
      chosen = candidates[0]!;
    } else {
      const withinCutoff = candidates.filter((candidate) => candidate.availableAt <= now + request.maxOptimizationWaitMs);
      chosen = withinCutoff.length > 0
        ? withinCutoff.sort(compareCapacity)[0]!
        : candidates.sort(compareAvailability)[0]!;
    }

    return {
      status: "offer",
      offer: {
        kind: request.offerKind,
        providerId: this.id,
        providerPriority: this.priority,
        modelId: chosen.model.id,
        inputTokens: chosen.inputTokens,
        inputCapacityTokens: chosen.model.contextWindowTokens,
        availableAt: chosen.availableAt,
      },
    };
  }

  private release(model: NvidiaModel): void {
    const state = this.modelState(model.id);
    state.active = Math.max(0, state.active - 1);
  }

  private blockModel(model: NvidiaModel, requestedDelayMs: number, now: number): number {
    const state = this.modelState(model.id);
    state.consecutiveRateLimits += 1;
    const multiplier = 2 ** Math.min(state.consecutiveRateLimits - 1, 4);
    const delayMs = Math.min(
      MAX_MODEL_RATE_LIMIT_BACKOFF_MS,
      Math.max(DEFAULT_RETRY_MS, requestedDelayMs) * multiplier,
    );
    state.blockedUntil = Math.max(state.blockedUntil, now + delayMs);
    log("warn", "nvidia_model_rate_limit", {
      provider: this.id,
      relay_model: model.id,
      consecutive_rate_limits: state.consecutiveRateLimits,
      blocked_until: new Date(state.blockedUntil).toISOString(),
    });
    return state.blockedUntil;
  }

  private blockProvider(requestedDelayMs: number, now: number, reason: string): number {
    this.providerFailureCount += 1;
    const multiplier = 2 ** Math.min(this.providerFailureCount - 1, 4);
    const delayMs = Math.min(
      MAX_PROVIDER_FAILURE_BACKOFF_MS,
      Math.max(PROVIDER_FAILURE_COOLDOWN_MS, requestedDelayMs) * multiplier,
    );
    this.providerBlockedUntil = Math.max(this.providerBlockedUntil, now + delayMs);
    log("warn", "provider_cooldown", {
      provider: this.id,
      reason,
      consecutive_failures: this.providerFailureCount,
      blocked_until: new Date(this.providerBlockedUntil).toISOString(),
    });
    return this.providerBlockedUntil;
  }

  async execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    stream: boolean,
    signal: AbortSignal,
  ): Promise<ProviderExecutionResult> {
    const apiKey = process.env.NVIDIA_API_KEY?.trim();
    if (!apiKey) {
      return {
        status: "retryable",
        scope: "provider",
        reason: "provider_not_configured",
        retryAt: Number.POSITIVE_INFINITY,
      };
    }

    const model = this.modelById(offer.modelId);
    const state = this.modelState(model.id);
    state.active += 1;

    let response: Response;
    try {
      response = await fetch(`${NVIDIA_BASE_URL}/v1/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${apiKey}`,
          "content-type": "application/json",
          accept: stream ? "text/event-stream, application/json" : "application/json",
          "user-agent": "ai-relay/1.0",
        },
        body: JSON.stringify({ ...body, model: model.upstreamModel, stream }),
        signal,
      });
    } catch (error) {
      this.release(model);
      const reason = error instanceof Error ? `network:${error.message}` : "network_error";
      return {
        status: "retryable",
        scope: "provider",
        reason,
        retryAt: this.blockProvider(DEFAULT_RETRY_MS, Date.now(), reason),
      };
    }

    if (response.ok) {
      const previousProviderFailures = this.providerFailureCount;
      const previousRateLimits = state.consecutiveRateLimits;
      this.providerFailureCount = 0;
      state.consecutiveRateLimits = 0;
      if (previousProviderFailures > 0 || previousRateLimits > 0) {
        log("info", "provider_recovered", {
          provider: this.id,
          relay_model: model.id,
          provider_failures: previousProviderFailures,
          model_rate_limits: previousRateLimits,
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
    }

    const bodyText = await response.text().catch(() => "");
    this.release(model);
    const retryAfterMs = parseRetryAfterMs(response.headers.get("retry-after"), DEFAULT_RETRY_MS);
    const failedAt = Date.now();

    if (response.status === 429) {
      log("warn", "nvidia_rate_limit_response", {
        provider: this.id,
        relay_model: model.id,
        retry_after: response.headers.get("retry-after"),
        detail: bodyText.replace(/\s+/g, " ").trim().slice(0, 1_000),
      });
      return {
        status: "retryable",
        scope: "model",
        reason: "rate_limit",
        retryAt: this.blockModel(model, retryAfterMs, failedAt),
      };
    }

    if (response.status === 408 || response.status >= 500) {
      const reason = `upstream_${response.status}`;
      return {
        status: "retryable",
        scope: "provider",
        reason,
        retryAt: this.blockProvider(retryAfterMs, failedAt, reason),
      };
    }

    if (response.status === 401 || response.status === 403) {
      this.providerBlockedUntil = Math.max(this.providerBlockedUntil, failedAt + 60_000);
      log("warn", "provider_cooldown", {
        provider: this.id,
        reason: `upstream_${response.status}`,
        blocked_until: new Date(this.providerBlockedUntil).toISOString(),
      });
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
        const state = this.modelState(model.id);
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

export function createNvidiaProvider(): Provider {
  return new NvidiaProvider();
}
