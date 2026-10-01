import { once } from "node:events";
import type { ServerResponse } from "node:http";
import { log } from "./log.ts";
import {
  createProviders,
  type Provider,
  type ProviderExecutionResult,
  type ProviderOffer,
  type ProviderOfferKind,
  type ProviderOfferResult,
} from "./providers/index.ts";
import type { RelayConfig, RelayJob } from "./types.ts";

const MAX_QUEUE_BYPASSES = 8;
export const INITIAL_OPTIMIZATION_WAIT_MS = 15_000;
export const MAX_RETRYABLE_FAILURES_PER_PATH = 3;

export function isAutoModel(model: string): boolean {
  return model === "auto";
}

export function optimizationWaitMs(failureCount: number): number {
  return INITIAL_OPTIMIZATION_WAIT_MS / (2 ** Math.max(0, failureCount));
}

function openAiError(code: string, message: string): Record<string, unknown> {
  return { error: { type: "relay_error", code, message } };
}

const UPSTREAM_REJECTION_DETAIL_MAX_CHARS = 1_000;

function redactUpstreamDetail(value: string): string {
  return value
    .replace(/AIza[0-9A-Za-z_-]{20,}/g, "[REDACTED_GOOGLE_API_KEY]")
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer [REDACTED]");
}

export function upstreamRejectionDetail(bodyText: string): string | null {
  const text = bodyText.trim();
  if (!text) return null;

  let detail = text;
  try {
    const parsed = JSON.parse(text) as unknown;
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      const value = parsed as Record<string, unknown>;
      const rawError = value.error;
      if (typeof rawError === "object" && rawError !== null && !Array.isArray(rawError)) {
        const error = rawError as Record<string, unknown>;
        const parts: string[] = [];
        if (typeof error.code === "number" || typeof error.code === "string") parts.push(`code=${String(error.code)}`);
        if (typeof error.status === "string" && error.status) parts.push(`status=${error.status}`);
        if (typeof error.message === "string" && error.message) parts.push(`message=${error.message}`);
        if (parts.length > 0) detail = parts.join(" ");
      }
    }
  } catch {
    // Some SDK errors prepend status text before an embedded JSON body.
  }

  const redacted = redactUpstreamDetail(detail);
  return redacted.length <= UPSTREAM_REJECTION_DETAIL_MAX_CHARS
    ? redacted
    : `${redacted.slice(0, UPSTREAM_REJECTION_DETAIL_MAX_CHARS - 1)}…`;
}

function finishJson(response: ServerResponse, value: unknown): void {
  if (response.writableEnded || response.destroyed) return;
  response.end(JSON.stringify(value));
}

function finishStreamError(response: ServerResponse, value: unknown): void {
  if (response.writableEnded || response.destroyed) return;
  response.write(`data: ${JSON.stringify(value)}\n\n`);
  response.end("data: [DONE]\n\n");
}

async function writeChunk(response: ServerResponse, chunk: Uint8Array | string): Promise<void> {
  if (response.writableEnded || response.destroyed) return;
  if (!response.write(chunk)) await once(response, "drain");
}

function compareCapacity(left: ProviderOffer, right: ProviderOffer): number {
  return left.inputCapacityTokens - right.inputCapacityTokens
    || left.providerPriority - right.providerPriority
    || left.availableAt - right.availableAt;
}

function compareAvailability(left: ProviderOffer, right: ProviderOffer): number {
  return left.availableAt - right.availableAt
    || left.inputCapacityTokens - right.inputCapacityTokens
    || left.providerPriority - right.providerPriority;
}

export function selectOffer(
  offers: readonly ProviderOffer[],
  now: number,
  maxOptimizationWaitMs: number,
): ProviderOffer | null {
  if (offers.length === 0) return null;
  const withinCutoff = offers.filter((offer) => offer.availableAt <= now + maxOptimizationWaitMs);
  if (withinCutoff.length > 0) return [...withinCutoff].sort(compareCapacity)[0] ?? null;
  return [...offers].sort(compareAvailability)[0] ?? null;
}

interface JobChoice {
  offer: ProviderOffer;
  provider: Provider;
}

export class RelayScheduler {
  readonly config: RelayConfig;
  readonly queue: RelayJob[] = [];
  readonly providers: readonly Provider[];
  private readonly providerById: Map<string, Provider>;
  private timer?: NodeJS.Timeout;
  private draining = false;

  constructor(config: RelayConfig, providers: readonly Provider[] = createProviders()) {
    this.config = config;
    this.providers = providers;
    this.providerById = new Map(providers.map((provider) => [provider.id, provider]));
  }

  listModels(): Array<{ id: string; providerId: string; inputCapacityTokens: number }> {
    return this.providers.flatMap((provider) => provider.listModels());
  }

  hasModel(modelId: string): boolean {
    return this.listModels().some((model) => model.id === modelId);
  }

  enqueue(job: RelayJob): void {
    this.queue.push(job);
    log("info", "queue_enqueued", {
      request_id: job.id,
      requested_model: job.requestedModel,
      stream: job.stream,
      queue_depth: this.queue.length,
    });
    this.triggerDrain();
  }

  status(): Record<string, unknown> {
    return {
      queue_depth: this.queue.filter((job) => !job.cancelled).length,
      providers: this.providers.map((provider) => {
        const status = provider.status();
        return {
          id: status.id,
          configured: status.configured,
          blocked_until: status.blockedUntil === null ? null : new Date(status.blockedUntil).toISOString(),
          models: status.models.map((model) => ({
            id: model.id,
            active: model.active,
            blocked_until: model.blockedUntil === null ? null : new Date(model.blockedUntil).toISOString(),
            overflow_blocked_until: model.overflowBlockedUntil === null
              ? null
              : new Date(model.overflowBlockedUntil).toISOString(),
          })),
        };
      }),
    };
  }

  private triggerDrain(delayMs = 0): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }
    if (delayMs > 0) {
      this.timer = setTimeout(() => {
        this.timer = undefined;
        void this.drain();
      }, delayMs);
      this.timer.unref();
      return;
    }
    queueMicrotask(() => void this.drain());
  }

  private async offerResults(
    job: RelayJob,
    now: number,
    maxWait: number,
    avoidLastFailure: boolean,
    offerKind: ProviderOfferKind,
  ): Promise<ProviderOfferResult[]> {
    const requestedModel = isAutoModel(job.requestedModel) ? "auto" : job.requestedModel;
    const excludedModelIds = new Set(job.excludedModelIds);
    const lastFailure = avoidLastFailure ? job.lastRetryFailure : undefined;
    if (lastFailure?.scope === "model") excludedModelIds.add(lastFailure.modelId);

    const providers = this.providers
      .filter((provider) => !job.excludedProviderIds.has(provider.id))
      .filter((provider) => !(lastFailure?.scope === "provider" && lastFailure.providerId === provider.id));

    return Promise.all(providers.map(async (provider): Promise<ProviderOfferResult> => {
      try {
        const result = await provider.getBestOffer({
          offerKind,
          body: job.body,
          requestedModel,
          maxOptimizationWaitMs: maxWait,
          excludedModelIds,
        }, now);
        if (result.status === "offer") {
          log("debug", "provider_offer", {
            request_id: job.id,
            provider: provider.id,
            relay_model: result.offer.modelId,
            offer_kind: offerKind,
            input_tokens: result.offer.inputTokens,
            input_capacity_tokens: result.offer.inputCapacityTokens,
            available_at: Number.isFinite(result.offer.availableAt)
              ? new Date(result.offer.availableAt).toISOString()
              : null,
            avoid_last_failure: avoidLastFailure,
          });
        } else {
          log(result.reason === "token_count_failed" ? "warn" : "debug", "provider_no_offer", {
            request_id: job.id,
            provider: provider.id,
            offer_kind: offerKind,
            reason: result.reason,
            detail: result.detail ?? null,
            avoid_last_failure: avoidLastFailure,
          });
        }
        return result;
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        log("warn", "provider_offer_error", {
          request_id: job.id,
          provider: provider.id,
          offer_kind: offerKind,
          error: detail,
        });
        return {
          status: "no_offer",
          providerId: provider.id,
          reason: "offer_evaluation_failed",
          detail,
        };
      }
    }));
  }

  private async offers(
    job: RelayJob,
    now: number,
    maxWait: number,
    avoidLastFailure: boolean,
    offerKind: ProviderOfferKind,
  ): Promise<ProviderOffer[]> {
    const results = await this.offerResults(job, now, maxWait, avoidLastFailure, offerKind);
    return results
      .filter((result): result is Extract<ProviderOfferResult, { status: "offer" }> => result.status === "offer")
      .map((result) => result.offer);
  }

  private async choice(job: RelayJob, now: number): Promise<JobChoice | null> {
    const maxWait = optimizationWaitMs(job.failureCount);
    const withinStandardWindow = (offer: ProviderOffer): boolean =>
      offer.availableAt <= Date.now() + maxWait;

    if (job.lastRetryFailure) {
      const standardAlternative = selectOffer(
        await this.offers(job, now, maxWait, true, "standard"),
        Date.now(),
        maxWait,
      );
      if (standardAlternative && withinStandardWindow(standardAlternative)) {
        const provider = this.providerById.get(standardAlternative.providerId);
        if (provider) return { provider, offer: standardAlternative };
      }
    }

    const standard = selectOffer(
      await this.offers(job, now, maxWait, false, "standard"),
      Date.now(),
      maxWait,
    );
    if (standard && withinStandardWindow(standard)) {
      const provider = this.providerById.get(standard.providerId);
      if (provider) return { provider, offer: standard };
    }

    if (job.lastRetryFailure) {
      const overflowAlternative = selectOffer(
        await this.offers(job, now, 0, true, "overflow"),
        Date.now(),
        0,
      );
      if (overflowAlternative && overflowAlternative.availableAt <= Date.now()) {
        const provider = this.providerById.get(overflowAlternative.providerId);
        if (provider) return { provider, offer: overflowAlternative };
      }
    }

    const overflow = selectOffer(
      await this.offers(job, now, 0, false, "overflow"),
      Date.now(),
      0,
    );
    if (overflow && overflow.availableAt <= Date.now()) {
      const provider = this.providerById.get(overflow.providerId);
      if (provider) return { provider, offer: overflow };
    }

    if (!standard) return null;
    const provider = this.providerById.get(standard.providerId);
    return provider ? { provider, offer: standard } : null;
  }

  private async failNoOffer(job: RelayJob): Promise<void> {
    const models = this.listModels();
    const eligible = (isAutoModel(job.requestedModel)
      ? models.filter((model) => !job.excludedModelIds.has(model.id))
      : models.filter((model) => model.id === job.requestedModel && !job.excludedModelIds.has(model.id)))
      .filter((model) => !job.excludedProviderIds.has(model.providerId));

    let code = "model_unavailable";
    let message = "No provider can currently handle this request.";

    if (eligible.length === 0 && (job.excludedModelIds.size > 0 || job.excludedProviderIds.size > 0)) {
      code = "upstream_unavailable";
      message = "Every eligible provider/model path is unavailable, rejected, or exhausted for this request.";
    } else {
      const relevantProviderIds = new Set(eligible.map((model) => model.providerId));
      const relevantProviders = this.providers.filter((provider) => relevantProviderIds.has(provider.id));
      const requestedModel = isAutoModel(job.requestedModel) ? "auto" : job.requestedModel;

      const diagnostics = await Promise.all(relevantProviders.map(async (provider): Promise<ProviderOfferResult> => {
        try {
          return await provider.getBestOffer({
            offerKind: "standard",
            body: job.body,
            requestedModel,
            maxOptimizationWaitMs: 0,
            excludedModelIds: new Set(job.excludedModelIds),
          }, Date.now());
        } catch (error) {
          return {
            status: "no_offer",
            providerId: provider.id,
            reason: "offer_evaluation_failed",
            detail: error instanceof Error ? error.message : String(error),
          };
        }
      }));

      const configuredDiagnostics = diagnostics.filter(
        (result) => !(result.status === "no_offer" && result.reason === "provider_not_configured"),
      );
      if (relevantProviders.length > 0 && configuredDiagnostics.length === 0) {
        code = "provider_not_configured";
        message = "No configured provider API key is available for an eligible model.";
      } else if (
        configuredDiagnostics.length > 0
        && configuredDiagnostics.every(
          (result) => result.status === "no_offer" && result.reason === "request_exceeds_capacity",
        )
      ) {
        code = "request_exceeds_provider_capacity";
        message = "Provider-specific input token counts exceed every configured eligible model's effective request capacity.";
      }
    }

    const error = openAiError(code, message);
    job.stream ? finishStreamError(job.response, error) : finishJson(job.response, error);
  }

  private removeCancelled(): void {
    for (let index = this.queue.length - 1; index >= 0; index -= 1) {
      if (this.queue[index]?.cancelled) this.queue.splice(index, 1);
    }
  }

  private dispatch(index: number, choice: JobChoice, now: number): void {
    const job = this.queue[index];
    if (!job) return;
    const skipped = this.queue.slice(0, index);
    this.queue.splice(index, 1);

    for (const older of skipped) {
      if (older.yieldOnce) {
        older.yieldOnce = false;
      } else {
        older.bypassCount += 1;
      }
    }
    job.yieldOnce = false;
    job.lastRetryFailure = undefined;

    log("info", "queue_dispatched", {
      request_id: job.id,
      relay_model: choice.offer.modelId,
      provider: choice.offer.providerId,
      offer_kind: choice.offer.kind,
      input_tokens: choice.offer.inputTokens,
      queue_ms: now - job.enqueuedAt,
      queue_bypasses: job.bypassCount,
      failure_count: job.failureCount,
      optimization_wait_ms: optimizationWaitMs(job.failureCount),
      queue_depth: this.queue.length,
    });

    void this.execute(job, choice.provider, choice.offer).finally(() => this.triggerDrain());
  }

  private requeueByAge(job: RelayJob): void {
    if (job.cancelled || job.response.writableEnded || job.response.destroyed) return;
    const index = this.queue.findIndex((queued) => queued.enqueuedAt > job.enqueuedAt);
    if (index < 0) this.queue.push(job);
    else this.queue.splice(index, 0, job);
  }

  private markFailureAndRequeue(job: RelayJob): void {
    job.failureCount += 1;
    job.yieldOnce = true;
    job.bypassCount = 0;
    this.requeueByAge(job);
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;

    try {
      for (;;) {
        this.removeCancelled();
        if (this.queue.length === 0) return;

        const now = Date.now();
        let nextWakeMs = Number.POSITIVE_INFINITY;
        let firstYieldingReady: { index: number; choice: JobChoice } | undefined;
        let changed = false;

        for (let index = 0; index < this.queue.length; index += 1) {
          const job = this.queue[index];
          if (!job) continue;
          const choice = await this.choice(job, now);
          const currentIndex = this.queue.indexOf(job);
          if (currentIndex < 0) {
            changed = true;
            break;
          }
          if (currentIndex !== index) {
            changed = true;
            break;
          }
          if (job.cancelled) {
            this.queue.splice(index, 1);
            changed = true;
            break;
          }
          if (!choice) {
            this.queue.splice(index, 1);
            await this.failNoOffer(job);
            changed = true;
            break;
          }

          const decisionNow = Date.now();
          const delayMs = choice.offer.availableAt - decisionNow;
          if (delayMs <= 0) {
            if (job.yieldOnce) {
              firstYieldingReady ??= { index, choice };
              continue;
            }
            this.dispatch(index, choice, decisionNow);
            changed = true;
            break;
          }

          if (Number.isFinite(delayMs)) nextWakeMs = Math.min(nextWakeMs, Math.max(1, delayMs));
          if (job.bypassCount >= MAX_QUEUE_BYPASSES && !job.yieldOnce) break;
        }

        if (changed) continue;
        if (firstYieldingReady) {
          this.dispatch(firstYieldingReady.index, firstYieldingReady.choice, Date.now());
          continue;
        }
        if (Number.isFinite(nextWakeMs)) this.triggerDrain(nextWakeMs);
        return;
      }
    } finally {
      this.draining = false;
    }
  }

  private terminalForJob(job: RelayJob, status: number, bodyText: string): void {
    let error: unknown;
    try {
      error = JSON.parse(bodyText) as unknown;
    } catch {
      error = openAiError("upstream_rejected", `Upstream returned HTTP ${status}.`);
    }
    job.stream ? finishStreamError(job.response, error) : finishJson(job.response, error);
  }

  private retryFailureKey(scope: "provider" | "model", providerId: string, modelId: string): string {
    return scope === "provider" ? `provider:${providerId}` : `model:${modelId}`;
  }

  private recordRetryableFailure(
    job: RelayJob,
    provider: Provider,
    offer: ProviderOffer,
    result: Extract<ProviderExecutionResult, { status: "retryable" }>,
  ): number {
    const key = this.retryFailureKey(result.scope, provider.id, offer.modelId);
    const count = (job.retryableFailureCounts.get(key) ?? 0) + 1;
    job.retryableFailureCounts.set(key, count);

    if (count >= MAX_RETRYABLE_FAILURES_PER_PATH) {
      if (result.scope === "provider") job.excludedProviderIds.add(provider.id);
      else job.excludedModelIds.add(offer.modelId);
      job.lastRetryFailure = undefined;
    } else {
      job.lastRetryFailure = { scope: result.scope, providerId: provider.id, modelId: offer.modelId };
    }
    return count;
  }

  private async execute(job: RelayJob, provider: Provider, offer: ProviderOffer): Promise<void> {
    if (job.cancelled) return;

    const controller = new AbortController();
    job.upstreamAbort = controller;
    const timeout = setTimeout(
      () => controller.abort(new Error("upstream timeout")),
      this.config.server.upstreamTimeoutSeconds * 1000,
    );
    timeout.unref();

    const startedAt = Date.now();
    const result = await provider.execute(
      offer,
      job.body,
      job.stream,
      controller.signal,
    );
    clearTimeout(timeout);
    job.upstreamAbort = undefined;
    if (job.cancelled) {
      if (result.status === "success") result.release();
      return;
    }

    if (result.status === "retryable") {
      const retryCount = this.recordRetryableFailure(job, provider, offer, result);
      log("warn", "provider_retryable_failure", {
        request_id: job.id,
        relay_model: offer.modelId,
        provider: provider.id,
        offer_kind: offer.kind,
        scope: result.scope,
        reason: result.reason,
        retry_count: retryCount,
        retry_budget: MAX_RETRYABLE_FAILURES_PER_PATH,
        retry_at: Number.isFinite(result.retryAt) ? new Date(result.retryAt).toISOString() : null,
      });
      this.markFailureAndRequeue(job);
      return;
    }

    if (result.status === "rejected") {
      log("warn", "provider_rejected", {
        request_id: job.id,
        relay_model: offer.modelId,
        provider: provider.id,
        offer_kind: offer.kind,
        scope: result.scope,
        status: result.httpStatus,
        detail: upstreamRejectionDetail(result.bodyText),
      });
      if (isAutoModel(job.requestedModel)) {
        if (result.scope === "provider") job.excludedProviderIds.add(provider.id);
        else job.excludedModelIds.add(offer.modelId);
        job.lastRetryFailure = undefined;
        this.markFailureAndRequeue(job);
      } else {
        this.terminalForJob(job, result.httpStatus, result.bodyText);
      }
      return;
    }

    log("info", "upstream_response", {
      request_id: job.id,
      relay_model: offer.modelId,
      provider: provider.id,
      offer_kind: offer.kind,
      input_tokens: offer.inputTokens,
      status: result.response.status,
      connect_ms: Date.now() - startedAt,
    });

    try {
      if (job.stream) await this.forwardStream(job, offer.modelId, result.response);
      else {
        const text = await result.response.text();
        if (!job.cancelled && !job.response.writableEnded && !job.response.destroyed) job.response.end(text);
      }
      log("info", "request_complete", {
        request_id: job.id,
        relay_model: offer.modelId,
        provider: provider.id,
        offer_kind: offer.kind,
        failovers: job.failureCount,
        total_ms: Date.now() - job.enqueuedAt,
      });
    } catch (error) {
      if (!job.cancelled) {
        log("warn", "upstream_body_error", {
          request_id: job.id,
          relay_model: offer.modelId,
          provider: provider.id,
          error: error instanceof Error ? error.message : String(error),
        });
        if (!job.response.writableEnded && !job.response.destroyed) job.response.end();
      }
    } finally {
      result.release();
    }
  }

  private async forwardStream(job: RelayJob, modelId: string, response: Response): Promise<void> {
    if (!response.body) throw new Error("Upstream streaming response had no body");
    if (job.heartbeatTimer) {
      clearInterval(job.heartbeatTimer);
      job.heartbeatTimer = undefined;
    }
    await writeChunk(job.response, `: ai-relay selected ${modelId}\n\n`);
    const reader = response.body.getReader();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (job.cancelled) {
          await reader.cancel().catch(() => undefined);
          return;
        }
        if (value?.byteLength) await writeChunk(job.response, value);
      }
      if (!job.response.writableEnded && !job.response.destroyed) job.response.end();
    } finally {
      reader.releaseLock();
    }
  }
}
