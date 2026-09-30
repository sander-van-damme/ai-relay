import {
  effectiveInputCapacity,
  emptyQuotaState,
  parseRetryAfterMs,
  quotaCanEverHandle,
  quotaDelayMs,
  releaseQuota,
  reserveQuota,
  type QuotaPolicy,
  type QuotaRuntimeState,
} from "./quota.ts";
import type { ChatCompletionRequest } from "../../types.ts";
import type {
  OfferRequest,
  Provider,
  ProviderExecutionResult,
  ProviderModelInfo,
  ProviderOffer,
  ProviderStatus,
} from "./types.ts";

export interface ManagedModel {
  id: string;
  upstreamModel: string;
  contextWindowTokens: number;
  quota: QuotaPolicy;
}

export interface OpenAICompatibleProviderOptions {
  id: string;
  priority: number;
  credentialEnv: string;
  baseUrl: string;
  defaultRetryMs: number;
  providerFailureCooldownMs: number;
  providerQuota?: QuotaPolicy;
  models: readonly ManagedModel[];
}

interface Candidate {
  model: ManagedModel;
  delayMs: number;
  inputCapacityTokens: number;
  index: number;
}

function compareCapacity(left: Candidate, right: Candidate): number {
  return left.inputCapacityTokens - right.inputCapacityTokens
    || left.delayMs - right.delayMs
    || left.index - right.index;
}

function compareAvailability(left: Candidate, right: Candidate): number {
  return left.delayMs - right.delayMs
    || left.inputCapacityTokens - right.inputCapacityTokens
    || left.index - right.index;
}

export class OpenAICompatibleProvider implements Provider {
  readonly id: string;
  readonly priority: number;
  private readonly credentialEnv: string;
  private readonly baseUrl: string;
  private readonly defaultRetryMs: number;
  private readonly providerFailureCooldownMs: number;
  private readonly providerQuota?: QuotaPolicy;
  private readonly models: readonly ManagedModel[];
  private readonly modelStates = new Map<string, QuotaRuntimeState>();
  private readonly providerState = emptyQuotaState();
  private providerFailureCount = 0;
  private readonly modelFailureCounts = new Map<string, number>();

  constructor(options: OpenAICompatibleProviderOptions) {
    this.id = options.id;
    this.priority = options.priority;
    this.credentialEnv = options.credentialEnv;
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.defaultRetryMs = options.defaultRetryMs;
    this.providerFailureCooldownMs = options.providerFailureCooldownMs;
    this.providerQuota = options.providerQuota;
    this.models = options.models;
    for (const model of this.models) this.modelStates.set(model.id, emptyQuotaState());
  }

  isConfigured(): boolean {
    return Boolean(process.env[this.credentialEnv]?.trim());
  }

  listModels(): readonly ProviderModelInfo[] {
    return this.models.map((model) => ({
      id: model.id,
      providerId: this.id,
      inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
    }));
  }

  private modelState(modelId: string): QuotaRuntimeState {
    const state = this.modelStates.get(modelId);
    if (!state) throw new Error(`Unknown model state: ${modelId}`);
    return state;
  }

  private candidate(model: ManagedModel, index: number, request: OfferRequest, now: number): Candidate | null {
    if (request.excludedModelIds.has(model.id)) return null;
    if (request.requestedModel !== "auto" && request.requestedModel !== model.id) return null;
    if (!quotaCanEverHandle(model.quota, request.estimatedInputTokens, model.contextWindowTokens)) return null;

    const state = this.modelState(model.id);
    const delays = [
      quotaDelayMs(model.quota, state, request.estimatedInputTokens, now),
      Math.max(0, this.providerState.blockedUntil - now),
    ];
    if (this.providerQuota) {
      if (!quotaCanEverHandle(this.providerQuota, request.estimatedInputTokens)) return null;
      delays.push(quotaDelayMs(this.providerQuota, this.providerState, request.estimatedInputTokens, now));
    }
    const delayMs = Math.max(...delays);
    return {
      model,
      delayMs,
      inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
      index,
    };
  }

  getBestOffer(request: OfferRequest, now = Date.now()): ProviderOffer | null {
    if (!this.isConfigured()) return null;

    const candidates = this.models
      .map((model, index) => this.candidate(model, index, request, now))
      .filter((candidate): candidate is Candidate => candidate !== null);
    if (candidates.length === 0) return null;

    let chosen: Candidate;
    if (request.requestedModel !== "auto") {
      chosen = candidates[0]!;
    } else {
      const withinCutoff = candidates.filter((candidate) => candidate.delayMs <= request.maxOptimizationWaitMs);
      chosen = withinCutoff.length > 0
        ? withinCutoff.sort(compareCapacity)[0]!
        : candidates.sort(compareAvailability)[0]!;
    }

    return {
      providerId: this.id,
      providerPriority: this.priority,
      modelId: chosen.model.id,
      inputCapacityTokens: chosen.inputCapacityTokens,
      availableAt: Number.isFinite(chosen.delayMs) ? now + Math.max(0, chosen.delayMs) : Number.POSITIVE_INFINITY,
    };
  }

  private modelById(modelId: string): ManagedModel {
    const model = this.models.find((candidate) => candidate.id === modelId);
    if (!model) throw new Error(`Provider ${this.id} does not own model ${modelId}`);
    return model;
  }

  private release(model: ManagedModel): void {
    releaseQuota(this.modelState(model.id));
    if (this.providerQuota) releaseQuota(this.providerState);
  }

  private blockModel(model: ManagedModel, requestedDelayMs: number, now: number): number {
    const failures = (this.modelFailureCounts.get(model.id) ?? 0) + 1;
    this.modelFailureCounts.set(model.id, failures);
    const multiplier = 2 ** Math.min(failures - 1, 4);
    const delayMs = Math.min(60_000, Math.max(this.defaultRetryMs, requestedDelayMs) * multiplier);
    const retryAt = now + delayMs;
    const state = this.modelState(model.id);
    state.blockedUntil = Math.max(state.blockedUntil, retryAt);
    return retryAt;
  }

  private blockProvider(requestedDelayMs: number, now: number): number {
    this.providerFailureCount += 1;
    const multiplier = 2 ** Math.min(this.providerFailureCount - 1, 4);
    const delayMs = Math.min(
      120_000,
      Math.max(this.providerFailureCooldownMs, requestedDelayMs) * multiplier,
    );
    const retryAt = now + delayMs;
    this.providerState.blockedUntil = Math.max(this.providerState.blockedUntil, retryAt);
    return retryAt;
  }

  async execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    stream: boolean,
    estimatedInputTokens: number,
    signal: AbortSignal,
  ): Promise<ProviderExecutionResult> {
    const apiKey = process.env[this.credentialEnv]?.trim();
    if (!apiKey) {
      return { status: "retryable", scope: "provider", reason: "provider_not_configured", retryAt: Number.POSITIVE_INFINITY };
    }

    const model = this.modelById(offer.modelId);
    const now = Date.now();
    reserveQuota(model.quota, this.modelState(model.id), estimatedInputTokens, now);
    if (this.providerQuota) {
      reserveQuota(this.providerQuota, this.providerState, estimatedInputTokens, now);
    }

    const upstreamBody = { ...body, model: model.upstreamModel, stream };

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${apiKey}`,
          "content-type": "application/json",
          accept: stream ? "text/event-stream, application/json" : "application/json",
          "user-agent": "ai-relay/1.0",
        },
        body: JSON.stringify(upstreamBody),
        signal,
      });
    } catch (error) {
      this.release(model);
      const retryAt = this.blockProvider(this.defaultRetryMs, Date.now());
      return {
        status: "retryable",
        scope: "provider",
        reason: error instanceof Error ? `network:${error.message}` : "network_error",
        retryAt,
      };
    }

    if (response.ok) {
      this.providerFailureCount = 0;
      this.modelFailureCounts.set(model.id, 0);
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
    const retryAfterMs = parseRetryAfterMs(response.headers.get("retry-after"), this.defaultRetryMs);

    if (response.status === 429) {
      return {
        status: "retryable",
        scope: "model",
        reason: "rate_limit",
        retryAt: this.blockModel(model, retryAfterMs, Date.now()),
      };
    }
    if (response.status === 408 || response.status >= 500) {
      return {
        status: "retryable",
        scope: "provider",
        reason: `upstream_${response.status}`,
        retryAt: this.blockProvider(retryAfterMs, Date.now()),
      };
    }

    if (response.status === 401 || response.status === 403) {
      this.providerState.blockedUntil = Math.max(this.providerState.blockedUntil, Date.now() + 60_000);
      return { status: "rejected", scope: "provider", httpStatus: response.status, bodyText };
    }
    return { status: "rejected", scope: "model", httpStatus: response.status, bodyText };
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
        };
      }),
    };
  }
}
