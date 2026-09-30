import { once } from "node:events";
import type { ServerResponse } from "node:http";
import { resolveProviderApiKey } from "./config.ts";
import { log } from "./log.ts";
import {
  parseRetryAfterMs,
  quotaDelayMs,
  reserveQuota,
} from "./quota.ts";
import type {
  ChatCompletionRequest,
  ModelConfig,
  ModelRuntimeState,
  RelayConfig,
  RelayJob,
} from "./types.ts";

const AUTO_MODELS = new Set(["auto", "relay/auto", "free", "best-free", "best"]);

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
  readonly states = new Map<string, ModelRuntimeState>();
  readonly queue: RelayJob[] = [];
  private timer?: NodeJS.Timeout;
  private draining = false;

  constructor(config: RelayConfig) {
    this.config = config;
    for (const model of config.models) this.states.set(model.id, emptyState());
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
      models: this.config.models.filter((model) => model.enabled).map((model) => {
        const state = this.states.get(model.id) ?? emptyState();
        return {
          id: model.id,
          provider: model.provider,
          active: state.active,
          max_concurrent: model.maxConcurrent,
          blocked_until: state.blockedUntil > Date.now() ? new Date(state.blockedUntil).toISOString() : null,
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
    return this.candidates(job).filter((model) =>
      model.limits.inputTokensPerMinute === null
      || job.estimatedInputTokens <= model.limits.inputTokensPerMinute
    );
  }

  private availability(job: RelayJob, now: number): CandidateAvailability[] {
    return this.possibleCandidates(job).map((model) => ({
      model,
      delayMs: quotaDelayMs(
        model,
        this.states.get(model.id) ?? emptyState(),
        job.estimatedInputTokens,
        now,
      ),
    }));
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;

    try {
      for (;;) {
        while (this.queue[0]?.cancelled) this.queue.shift();
        const job = this.queue[0];
        if (!job) return;

        const now = Date.now();
        const configuredCandidates = this.candidates(job);
        const candidates = this.availability(job, now);
        if (candidates.length === 0) {
          this.queue.shift();
          const tokenLimited = configuredCandidates.length > 0;
          const error = openAiError(
            tokenLimited ? "request_exceeds_configured_tpm" : "model_unavailable",
            tokenLimited
              ? "Estimated input tokens exceed the configured inputTokensPerMinute limit for every eligible model."
              : isAuto(job.requestedModel)
                ? "No configured model can handle this request."
                : `Configured model not found: ${job.requestedModel}`,
          );
          job.stream ? finishStreamError(job.response, error) : finishJson(job.response, error);
          continue;
        }

        const ready = candidates.find((candidate) => candidate.delayMs <= 0);
        if (ready) {
          this.queue.shift();
          const state = this.states.get(ready.model.id) ?? emptyState();
          this.states.set(ready.model.id, state);
          reserveQuota(state, job.estimatedInputTokens, now);

          log("info", "queue_dispatched", {
            request_id: job.id,
            relay_model: ready.model.id,
            provider: ready.model.provider,
            queue_ms: now - job.enqueuedAt,
            queue_depth: this.queue.length,
          });

          void this.execute(job, ready.model, state).finally(() => {
            state.active = Math.max(0, state.active - 1);
            this.triggerDrain();
          });
          continue;
        }

        const finiteDelays = candidates
          .map((candidate) => candidate.delayMs)
          .filter(Number.isFinite);
        if (finiteDelays.length > 0) {
          this.triggerDrain(Math.max(1, Math.min(...finiteDelays)));
        }
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

  private async execute(job: RelayJob, model: ModelConfig, state: ModelRuntimeState): Promise<void> {
    if (job.cancelled) return;

    const provider = this.config.providers[model.provider];
    const credential = resolveProviderApiKey(provider);
    if (!credential) {
      this.terminalForModel(job, model, 503, JSON.stringify(openAiError(
        "provider_not_configured",
        `No API key found for provider ${model.provider}. Expected one of: ${provider.apiKeyEnv.join(", ")}`,
      )));
      return;
    }

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
          authorization: `Bearer ${credential.value}`,
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
      state.blockedUntil = Math.max(state.blockedUntil, Date.now() + retryMs);
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
        state.blockedUntil = Math.max(state.blockedUntil, Date.now() + retryMs);
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
