import { log } from "../../log.ts";
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
} from "./quota.ts";
import type { ChatCompletionRequest } from "../../types.ts";
import type {
  OfferRequest,
  Provider,
  ProviderExecutionResult,
  ProviderModelInfo,
  ProviderOffer,
  ProviderOfferResult,
  ProviderStatus,
} from "./types.ts";

export interface ManagedModel {
  id: string;
  upstreamModel: string;
  contextWindowTokens: number;
  quota: QuotaPolicy;
}

export type InputTokenCounter = (
  body: ChatCompletionRequest,
  model: ManagedModel,
) => number | Promise<number>;

export interface OpenAICompatibleProviderOptions {
  id: string;
  priority: number;
  credentialEnv: string;
  baseUrl: string;
  defaultRetryMs: number;
  providerFailureCooldownMs: number;
  countInputTokens: InputTokenCounter;
  overflowProbe?: {
    hardCapTtlMs: number;
    limits: readonly QuotaLimitName[];
  };
  providerQuota?: QuotaPolicy;
  models: readonly ManagedModel[];
}

interface Candidate {
  model: ManagedModel;
  inputTokens: number;
  inputCapacityTokens: number;
  availableAt: number;
  index: number;
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

export class OpenAICompatibleProvider implements Provider {
  readonly id: string;
  readonly priority: number;
  private readonly credentialEnv: string;
  private readonly baseUrl: string;
  private readonly defaultRetryMs: number;
  private readonly providerFailureCooldownMs: number;
  private readonly inputTokenCounter: InputTokenCounter;
  private readonly tokenCountCache = new WeakMap<ChatCompletionRequest, Map<string, Promise<number>>>();
  private readonly overflowProbe?: {
    hardCapTtlMs: number;
    limits: ReadonlySet<QuotaLimitName>;
  };
  private readonly providerQuota?: QuotaPolicy;
  private readonly models: readonly ManagedModel[];
  private readonly modelStates = new Map<string, QuotaRuntimeState>();
  private readonly providerState = emptyQuotaState();
  private providerFailureCount = 0;
  private readonly modelFailureCounts = new Map<string, number>();
  private readonly modelOverflowBlockedUntil = new Map<string, number>();

  constructor(options: OpenAICompatibleProviderOptions) {
    this.id = options.id;
    this.priority = options.priority;
    this.credentialEnv = options.credentialEnv;
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.defaultRetryMs = options.defaultRetryMs;
    this.providerFailureCooldownMs = options.providerFailureCooldownMs;
    this.inputTokenCounter = options.countInputTokens;
    this.overflowProbe = options.overflowProbe
      ? {
          hardCapTtlMs: options.overflowProbe.hardCapTtlMs,
          limits: new Set(options.overflowProbe.limits),
        }
      : undefined;
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


  private async countInputTokens(body: ChatCompletionRequest, modelId: string): Promise<number> {
    const model = this.modelById(modelId);
    let perModel = this.tokenCountCache.get(body);
    if (!perModel) {
      perModel = new Map<string, Promise<number>>();
      this.tokenCountCache.set(body, perModel);
    }

    const cached = perModel.get(modelId);
    if (cached) return cached;

    const pending = Promise.resolve()
      .then(() => this.inputTokenCounter(body, model))
      .then((count) => {
        if (!Number.isSafeInteger(count) || count < 0) {
          throw new Error(`Provider ${this.id} returned an invalid input token count for ${modelId}: ${count}`);
        }
        return count;
      })
      .catch((error) => {
        perModel!.delete(modelId);
        throw error;
      });

    perModel.set(modelId, pending);
    return pending;
  }

  private async candidate(model: ManagedModel, index: number, request: OfferRequest, now: number): Promise<Candidate | null> {
    if (request.excludedModelIds.has(model.id)) return null;
    if (request.requestedModel !== "auto" && request.requestedModel !== model.id) return null;

    const inputTokens = await this.countInputTokens(request.body, model.id);
    if (!quotaCanEverHandle(model.quota, inputTokens, model.contextWindowTokens)) return null;

    const evaluatedAt = Math.max(now, Date.now());
    const state = this.modelState(model.id);
    const modelDelayMs = quotaDelayMs(model.quota, state, inputTokens, evaluatedAt);
    const providerBlockedMs = Math.max(0, this.providerState.blockedUntil - evaluatedAt);

    if (request.offerKind === "overflow") {
      if (!this.overflowProbe || providerBlockedMs > 0) return null;
      if ((this.modelOverflowBlockedUntil.get(model.id) ?? 0) > evaluatedAt) return null;

      const modelCanOverflow = quotaCanOverflow(
        model.quota,
        state,
        inputTokens,
        this.overflowProbe.limits,
        model.contextWindowTokens,
        evaluatedAt,
      );

      let providerDelayMs = 0;
      let providerCanOverflow = false;
      if (this.providerQuota) {
        if (!quotaCanEverHandle(this.providerQuota, inputTokens)) return null;
        providerDelayMs = quotaDelayMs(this.providerQuota, this.providerState, inputTokens, evaluatedAt);
        providerCanOverflow = quotaCanOverflow(
          this.providerQuota,
          this.providerState,
          inputTokens,
          this.overflowProbe.limits,
          Number.POSITIVE_INFINITY,
          evaluatedAt,
        );
      }

      if (modelDelayMs > 0 && !modelCanOverflow) return null;
      if (providerDelayMs > 0 && !providerCanOverflow) return null;
      if (!modelCanOverflow && !providerCanOverflow) return null;

      return {
        model,
        inputTokens,
        inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
        availableAt: evaluatedAt,
        index,
      };
    }

    const delays = [modelDelayMs, providerBlockedMs];
    if (this.providerQuota) {
      if (!quotaCanEverHandle(this.providerQuota, inputTokens)) return null;
      delays.push(quotaDelayMs(this.providerQuota, this.providerState, inputTokens, evaluatedAt));
    }

    const delayMs = Math.max(...delays);
    return {
      model,
      inputTokens,
      inputCapacityTokens: effectiveInputCapacity(model.quota, model.contextWindowTokens),
      availableAt: Number.isFinite(delayMs) ? evaluatedAt + Math.max(0, delayMs) : Number.POSITIVE_INFINITY,
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
      .filter(({ model }) => request.requestedModel === "auto" || request.requestedModel === model.id);

    if (eligibleModels.length === 0) {
      return { status: "no_offer", providerId: this.id, reason: "no_eligible_model" };
    }

    const results = await Promise.all(eligibleModels.map(async ({ model, index }) => {
      try {
        return { candidate: await this.candidate(model, index, request, now), error: undefined };
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
      return {
        status: "no_offer",
        providerId: this.id,
        reason: request.offerKind === "standard" ? "request_exceeds_capacity" : "no_eligible_model",
      };
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
        inputCapacityTokens: chosen.inputCapacityTokens,
        availableAt: chosen.availableAt,
      },
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

  private blockModel(model: ManagedModel, requestedDelayMs: number, now: number, reason: string): number {
    const failures = (this.modelFailureCounts.get(model.id) ?? 0) + 1;
    this.modelFailureCounts.set(model.id, failures);
    const multiplier = 2 ** Math.min(failures - 1, 4);
    const delayMs = Math.min(60_000, Math.max(this.defaultRetryMs, requestedDelayMs) * multiplier);
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
    const delayMs = Math.min(
      120_000,
      Math.max(this.providerFailureCooldownMs, requestedDelayMs) * multiplier,
    );
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

  async execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    stream: boolean,
    signal: AbortSignal,
  ): Promise<ProviderExecutionResult> {
    const apiKey = process.env[this.credentialEnv]?.trim();
    if (!apiKey) {
      return { status: "retryable", scope: "provider", reason: "provider_not_configured", retryAt: Number.POSITIVE_INFINITY };
    }

    const model = this.modelById(offer.modelId);
    const now = Date.now();
    reserveQuota(model.quota, this.modelState(model.id), offer.inputTokens, now);
    if (this.providerQuota) {
      reserveQuota(this.providerQuota, this.providerState, offer.inputTokens, now);
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
      const reason = error instanceof Error ? `network:${error.message}` : "network_error";
      const retryAt = this.blockProvider(this.defaultRetryMs, Date.now(), reason);
      return {
        status: "retryable",
        scope: "provider",
        reason,
        retryAt,
      };
    }

    if (response.ok) {
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
    }

    const bodyText = await response.text().catch(() => "");
    this.release(model);
    const retryAfterMs = parseRetryAfterMs(response.headers.get("retry-after"), this.defaultRetryMs);

    if (response.status === 429) {
      const failedAt = Date.now();
      if (offer.kind === "overflow" && this.overflowProbe) {
        const blockedUntil = failedAt + this.overflowProbe.hardCapTtlMs;
        this.modelOverflowBlockedUntil.set(
          model.id,
          Math.max(this.modelOverflowBlockedUntil.get(model.id) ?? 0, blockedUntil),
        );
      }
      const reason = offer.kind === "overflow" ? "overflow_limit_confirmed" : "rate_limit";
      return {
        status: "retryable",
        scope: "model",
        reason,
        retryAt: this.blockModel(model, retryAfterMs, failedAt, reason),
      };
    }
    if (response.status === 408 || response.status >= 500) {
      const reason = `upstream_${response.status}`;
      return {
        status: "retryable",
        scope: "provider",
        reason,
        retryAt: this.blockProvider(retryAfterMs, Date.now(), reason),
      };
    }

    if (response.status === 401 || response.status === 403) {
      this.providerState.blockedUntil = Math.max(this.providerState.blockedUntil, Date.now() + 60_000);
      log("warn", "provider_cooldown", {
        provider: this.id,
        reason: `upstream_${response.status}`,
        blocked_until: new Date(this.providerState.blockedUntil).toISOString(),
      });
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
          overflowBlockedUntil: (this.modelOverflowBlockedUntil.get(model.id) ?? 0) > now
            ? this.modelOverflowBlockedUntil.get(model.id)!
            : null,
        };
      }),
    };
  }
}
