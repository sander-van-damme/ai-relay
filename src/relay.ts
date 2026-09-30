import { once } from "node:events";
import type { ServerResponse } from "node:http";
import { log } from "./log.ts";
import { createProviders, type Provider, type ProviderOffer } from "./providers/index.ts";
import type { RelayConfig, RelayJob } from "./types.ts";

const AUTO_MODELS = new Set(["auto", "relay/auto", "free", "best-free", "best"]);
const MAX_QUEUE_BYPASSES = 8;
export const INITIAL_OPTIMIZATION_WAIT_MS = 15_000;

export function isAutoModel(model: string): boolean {
  return AUTO_MODELS.has(model);
}

export function optimizationWaitMs(failureCount: number): number {
  return INITIAL_OPTIMIZATION_WAIT_MS / (2 ** Math.max(0, failureCount));
}

function openAiError(code: string, message: string): Record<string, unknown> {
  return { error: { type: "relay_error", code, message } };
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

  constructor(config: RelayConfig, providers: readonly Provider[] = createProviders(config.server)) {
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
      estimated_input_tokens: job.estimatedInputTokens,
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

  private choice(job: RelayJob, now: number): JobChoice | null {
    const maxWait = optimizationWaitMs(job.failureCount);
    const requestedModel = isAutoModel(job.requestedModel) ? "auto" : job.requestedModel;
    const offers = this.providers
      .map((provider) => ({
        provider,
        offer: provider.getBestOffer({
          requestedModel,
          estimatedInputTokens: job.estimatedInputTokens,
          maxOptimizationWaitMs: maxWait,
          excludedModelIds: job.excludedModelIds,
        }, now),
      }))
      .filter((entry): entry is { provider: Provider; offer: ProviderOffer } => entry.offer !== null);

    const selected = selectOffer(offers.map((entry) => entry.offer), now, maxWait);
    if (!selected) return null;
    const provider = this.providerById.get(selected.providerId);
    return provider ? { provider, offer: selected } : null;
  }

  private failNoOffer(job: RelayJob): void {
    const models = this.listModels();
    const eligible = isAutoModel(job.requestedModel)
      ? models.filter((model) => !job.excludedModelIds.has(model.id))
      : models.filter((model) => model.id === job.requestedModel && !job.excludedModelIds.has(model.id));
    const anyConfigured = this.providers.some((provider) => provider.isConfigured());
    const capacityExists = eligible.some((model) => job.estimatedInputTokens <= model.inputCapacityTokens);

    let code = "model_unavailable";
    let message = "No provider can currently handle this request.";
    if (!anyConfigured) {
      code = "provider_not_configured";
      message = "No configured provider API key is available.";
    } else if (!capacityExists && eligible.length > 0) {
      code = "request_exceeds_provider_capacity";
      message = "Estimated input tokens exceed every eligible model's effective request capacity.";
    } else if (eligible.length === 0 && job.excludedModelIds.size > 0) {
      code = "upstream_rejected";
      message = "Every eligible model rejected this request.";
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

    log("info", "queue_dispatched", {
      request_id: job.id,
      relay_model: choice.offer.modelId,
      provider: choice.offer.providerId,
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
          const choice = this.choice(job, now);
          if (!choice) {
            this.queue.splice(index, 1);
            this.failNoOffer(job);
            changed = true;
            break;
          }

          const delayMs = choice.offer.availableAt - now;
          if (delayMs <= 0) {
            if (job.yieldOnce) {
              firstYieldingReady ??= { index, choice };
              continue;
            }
            this.dispatch(index, choice, now);
            changed = true;
            break;
          }

          if (Number.isFinite(delayMs)) nextWakeMs = Math.min(nextWakeMs, Math.max(1, delayMs));
          if (job.bypassCount >= MAX_QUEUE_BYPASSES && !job.yieldOnce) break;
        }

        if (changed) continue;
        if (firstYieldingReady) {
          this.dispatch(firstYieldingReady.index, firstYieldingReady.choice, now);
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
      job.estimatedInputTokens,
      controller.signal,
    );
    clearTimeout(timeout);
    job.upstreamAbort = undefined;
    if (job.cancelled) {
      if (result.status === "success") result.release();
      return;
    }

    if (result.status === "retryable") {
      log("warn", "provider_retryable_failure", {
        request_id: job.id,
        relay_model: offer.modelId,
        provider: provider.id,
        reason: result.reason,
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
        status: result.httpStatus,
      });
      if (isAutoModel(job.requestedModel)) {
        job.excludedModelIds.add(offer.modelId);
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
