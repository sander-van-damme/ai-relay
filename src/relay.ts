import { once } from "node:events";
import type { ServerResponse } from "node:http";
import { providerApiKeyEnv, resolveProviderApiKey } from "./config.ts";
import { log } from "./log.ts";
import {
  parseRetryAfterMs,
  quotaCanEverHandle,
  quotaDelayMs,
  reserveQuota,
} from "./quota.ts";
import type {
  ModelConfig,
  ModelRuntimeState,
  ProviderId,
  RelayConfig,
  RelayJob,
} from "./types.ts";

const AUTO_MODELS = new Set(["auto", "relay/auto", "free", "best-free", "best"]);
const MAX_QUEUE_BYPASSES = 8;

function isAuto(model: string): boolean {
  return AUTO_MODELS.has(model);
}

function emptyState(): ModelRuntimeState {
  return { active: 0, blockedUntil: 0, lastStartedAt: 0, events: [] };
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

interface CandidateAvailability {
  model: ModelConfig;
  delayMs: number;
}

export class RelayScheduler {
  readonly config: RelayConfig;
  readonly modelStates = new Map<string, ModelRuntimeState>();
  readonly providerStates = new Map<ProviderId, ModelRuntimeState>();
  readonly queue: RelayJob[] = [];
  private timer?: NodeJS.Timeout;
  private draining = false;

  constructor(config: RelayConfig) {
    this.config = config;
    for (const model of config.models) this.modelStates.set(model.id, emptyState());
    for (const provider of Object.keys(config.providers) as ProviderId[]) {
      this.providerStates.set(provider, emptyState());
    }
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
    const now = Date.now();
    return {
      queue_depth: this.queue.filter((job) => !job.cancelled).length,
      providers: (Object.entries(this.config.providers) as [ProviderId, RelayConfig["providers"][ProviderId]][])
        .map(([id, provider]) => {
          const state = this.providerStates.get(id) ?? emptyState();
          return {
            id,
            active: state.active,
            max_concurrent: provider.maxConcurrent,
            blocked_until: state.blockedUntil > now ? new Date(state.blockedUntil).toISOString() : null,
          };
        }),
      models: this.config.models.filter((model) => model.enabled).map((model) => {
        const state = this.modelStates.get(model.id) ?? emptyState();
        return {
          id: model.id,
          provider: model.provider,
          active: state.active,
          max_concurrent: model.maxConcurrent,
          blocked_until: state.blockedUntil > now ? new Date(state.blockedUntil).toISOString() : null,
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

  private candidates(job: RelayJob): ModelConfig[] {
    const enabled = this.config.models.filter((model) => model.enabled && !job.excludedModels.has(model.id));
    if (isAuto(job.requestedModel)) return enabled;
    return enabled.filter((model) => model.id === job.requestedModel);
  }

  private possibleCandidates(job: RelayJob): ModelConfig[] {
    return this.candidates(job).filter((model) => {
      const provider = this.config.providers[model.provider];
      return quotaCanEverHandle(model, job.estimatedInputTokens)
        && quotaCanEverHandle(provider, job.estimatedInputTokens);
    });
  }

  private availability(job: RelayJob, now: number): CandidateAvailability[] {
    return this.possibleCandidates(job).map((model) => {
      const provider = this.config.providers[model.provider];
      const modelState = this.modelStates.get(model.id) ?? emptyState();
      const providerState = this.providerStates.get(model.provider) ?? emptyState();
      return {
        model,
        delayMs: Math.max(
          quotaDelayMs(model, modelState, job.estimatedInputTokens, now),
          quotaDelayMs(provider, providerState, job.estimatedInputTokens, now),
        ),
      };
    });
  }

  private failImpossibleJob(job: RelayJob, configuredCandidates: ModelConfig[]): void {
    const tokenLimited = configuredCandidates.length > 0;
    const error = openAiError(
      tokenLimited ? "request_exceeds_configured_tpm" : "model_unavailable",
      tokenLimited
        ? "Estimated input tokens exceed the configured inputTokensPerMinute limit for every eligible model/provider path."
        : isAuto(job.requestedModel)
          ? "No configured model can handle this request."
          : `Configured model not found: ${job.requestedModel}`,
    );
    job.stream ? finishStreamError(job.response, error) : finishJson(job.response, error);
  }

  private providerNotConfigured(job: RelayJob, model: ModelConfig): void {
    const env = providerApiKeyEnv(model.provider);
    log("warn", "provider_not_configured", {
      request_id: job.id,
      provider: model.provider,
      credential_env: env,
    });

    if (isAuto(job.requestedModel)) {
      for (const candidate of this.config.models) {
        if (candidate.provider === model.provider) job.excludedModels.add(candidate.id);
      }
      if (this.candidates(job).length > 0) {
        this.requeueFirst(job);
        return;
      }
    }

    const error = openAiError(
      "provider_not_configured",
      `No API key found for provider ${model.provider}. Set ${env}.`,
    );
    job.stream ? finishStreamError(job.response, error) : finishJson(job.response, error);
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;

    try {
      for (;;) {
        for (let index = this.queue.length - 1; index >= 0; index -= 1) {
          if (this.queue[index]?.cancelled) this.queue.splice(index, 1);
        }
        if (this.queue.length === 0) return;

        const now = Date.now();
        let nextWakeMs = Number.POSITIVE_INFINITY;
        let changed = false;

        for (let index = 0; index < this.queue.length; index += 1) {
          const job = this.queue[index];
          if (!job) continue;

          const configuredCandidates = this.candidates(job);
          const candidates = this.availability(job, now);
          if (candidates.length === 0) {
            this.queue.splice(index, 1);
            this.failImpossibleJob(job, configuredCandidates);
            changed = true;
            break;
          }

          const ready = candidates.find((candidate) => candidate.delayMs <= 0);
          if (ready) {
            const credential = resolveProviderApiKey(ready.model.provider);
            const skipped = this.queue.slice(0, index);
            this.queue.splice(index, 1);
            if (!credential) {
              this.providerNotConfigured(job, ready.model);
              changed = true;
              break;
            }
            for (const older of skipped) older.bypassCount += 1;

            const modelState = this.modelStates.get(ready.model.id) ?? emptyState();
            const providerState = this.providerStates.get(ready.model.provider) ?? emptyState();
            this.modelStates.set(ready.model.id, modelState);
            this.providerStates.set(ready.model.provider, providerState);
            reserveQuota(modelState, job.estimatedInputTokens, now);
            reserveQuota(providerState, job.estimatedInputTokens, now);

            log("info", "queue_dispatched", {
              request_id: job.id,
              relay_model: ready.model.id,
              provider: ready.model.provider,
              queue_ms: now - job.enqueuedAt,
              queue_bypasses: job.bypassCount,
              queue_depth: this.queue.length,
            });

            void this.execute(job, ready.model, credential.value, modelState, providerState).finally(() => {
              modelState.active = Math.max(0, modelState.active - 1);
              providerState.active = Math.max(0, providerState.active - 1);
              this.triggerDrain();
            });
            changed = true;
            break;
          }

          const finiteDelays = candidates
            .map((candidate) => candidate.delayMs)
            .filter(Number.isFinite);
          if (finiteDelays.length > 0) {
            nextWakeMs = Math.min(nextWakeMs, ...finiteDelays);
          }

          // Keep the queue work-conserving, but stop younger work from bypassing
          // one blocked request forever under sustained small-request traffic.
          if (job.bypassCount >= MAX_QUEUE_BYPASSES) break;
        }

        if (changed) continue;
        if (Number.isFinite(nextWakeMs)) this.triggerDrain(Math.max(1, nextWakeMs));
        return;
      }
    } finally {
      this.draining = false;
    }
  }

  private requeueFirst(job: RelayJob): void {
    if (job.cancelled || job.response.writableEnded || job.response.destroyed) return;
    this.queue.unshift(job);
  }

  private terminalForModel(job: RelayJob, model: ModelConfig, status: number, bodyText: string): void {
    log("warn", "upstream_rejected", {
      request_id: job.id,
      relay_model: model.id,
      provider: model.provider,
      status,
    });

    if (isAuto(job.requestedModel)) {
      job.excludedModels.add(model.id);
      if (this.candidates(job).length > 0) {
        this.requeueFirst(job);
        return;
      }
    }

    let error: unknown;
    try {
      error = JSON.parse(bodyText) as unknown;
    } catch {
      error = openAiError("upstream_rejected", `Upstream returned HTTP ${status}.`);
    }
    job.stream ? finishStreamError(job.response, error) : finishJson(job.response, error);
  }

  private async execute(
    job: RelayJob,
    model: ModelConfig,
    apiKey: string,
    modelState: ModelRuntimeState,
    providerState: ModelRuntimeState,
  ): Promise<void> {
    if (job.cancelled) return;

    const provider = this.config.providers[model.provider];
    const controller = new AbortController();
    job.upstreamAbort = controller;
    const timeout = setTimeout(
      () => controller.abort(new Error("upstream timeout")),
      this.config.server.upstreamTimeoutSeconds * 1000,
    );
    timeout.unref();

    const startedAt = Date.now();
    let response: Response;
    try {
      response = await fetch(`${provider.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          authorization: `Bearer ${apiKey}`,
          "content-type": "application/json",
          accept: job.stream ? "text/event-stream, application/json" : "application/json",
          "user-agent": "ai-relay/0.1",
        },
        body: JSON.stringify({ ...job.body, model: model.upstreamModel, stream: job.stream }),
        signal: controller.signal,
      });
    } catch (error) {
      clearTimeout(timeout);
      if (job.cancelled) return;
      const retryMs = this.config.server.retrySeconds * 1000;
      modelState.blockedUntil = Math.max(modelState.blockedUntil, Date.now() + retryMs);
      log("warn", "upstream_network_retry", {
        request_id: job.id,
        relay_model: model.id,
        retry_ms: retryMs,
        error: error instanceof Error ? error.message : String(error),
      });
      this.requeueFirst(job);
      return;
    }

    log("info", "upstream_response", {
      request_id: job.id,
      relay_model: model.id,
      provider: model.provider,
      status: response.status,
      connect_ms: Date.now() - startedAt,
    });

    if (!response.ok) {
      clearTimeout(timeout);
      const bodyText = await response.text().catch(() => "");
      if (response.status === 429 || response.status === 408 || response.status >= 500) {
        const retryMs = parseRetryAfterMs(
          response.headers.get("retry-after"),
          this.config.server.retrySeconds * 1000,
        );
        modelState.blockedUntil = Math.max(modelState.blockedUntil, Date.now() + retryMs);
        if (response.status === 429) {
          providerState.blockedUntil = Math.max(providerState.blockedUntil, Date.now() + retryMs);
        }
        log("warn", "upstream_rate_or_transient_retry", {
          request_id: job.id,
          relay_model: model.id,
          status: response.status,
          retry_ms: retryMs,
        });
        this.requeueFirst(job);
        return;
      }

      this.terminalForModel(job, model, response.status, bodyText);
      return;
    }

    try {
      if (job.stream) {
        await this.forwardStream(job, model, response);
      } else {
        const text = await response.text();
        if (!job.cancelled && !job.response.writableEnded && !job.response.destroyed) {
          job.response.end(text);
        }
      }
      log("info", "request_complete", {
        request_id: job.id,
        relay_model: model.id,
        total_ms: Date.now() - job.enqueuedAt,
      });
    } catch (error) {
      if (job.cancelled) return;
      log("warn", "upstream_body_error", {
        request_id: job.id,
        relay_model: model.id,
        error: error instanceof Error ? error.message : String(error),
      });
      if (!job.response.writableEnded && !job.response.destroyed) job.response.end();
    } finally {
      clearTimeout(timeout);
      job.upstreamAbort = undefined;
    }
  }

  private async forwardStream(job: RelayJob, model: ModelConfig, response: Response): Promise<void> {
    if (!response.body) throw new Error("Upstream streaming response had no body");
    if (job.heartbeatTimer) {
      clearInterval(job.heartbeatTimer);
      job.heartbeatTimer = undefined;
    }
    await writeChunk(job.response, `: ai-relay selected ${model.id}\n\n`);

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
