import type { Provider, ProviderUsage } from "./providers/index.ts";

interface Counters {
  requests: number;
  successes: number;
  terminalFailures: number;
  cancellations: number;
  attempts: number;
  failedAttempts: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

function counters(): Counters {
  return { requests: 0, successes: 0, terminalFailures: 0, cancellations: 0, attempts: 0, failedAttempts: 0, inputTokens: 0, outputTokens: 0, totalTokens: 0 };
}

export class Observability {
  readonly startedAt = Date.now();
  private peakQueueDepth = 0;
  private readonly totals = counters();
  private readonly providers = new Map<string, Counters>();
  private readonly models = new Map<string, Counters>();
  private readonly requestedModels = new Map<string, number>();
  private readonly registeredProviders: readonly Provider[];

  constructor(registeredProviders: readonly Provider[]) {
    this.registeredProviders = registeredProviders;
    this.requestedModels.set("auto", 0);
    for (const provider of registeredProviders) {
      this.providers.set(provider.id, counters());
      for (const model of provider.listModels()) {
        this.models.set(model.id, counters());
        this.requestedModels.set(model.id, 0);
      }
    }
  }

  request(requestedModel: string, queueDepth: number): void {
    this.totals.requests += 1;
    this.requestedModels.set(requestedModel, (this.requestedModels.get(requestedModel) ?? 0) + 1);
    this.peakQueueDepth = Math.max(this.peakQueueDepth, queueDepth);
  }

  attempt(providerId: string, modelId: string, inputTokens: number): void {
    for (const item of [this.totals, this.providers.get(providerId), this.models.get(modelId)]) {
      if (!item) continue;
      item.attempts += 1;
      item.inputTokens += inputTokens;
      item.totalTokens += inputTokens;
    }
  }

  failedAttempt(providerId: string, modelId: string): void {
    for (const item of [this.totals, this.providers.get(providerId), this.models.get(modelId)]) {
      if (item) item.failedAttempts += 1;
    }
  }

  success(providerId: string, modelId: string, inputTokens: number, usage?: ProviderUsage): void {
    for (const item of [this.totals, this.providers.get(providerId), this.models.get(modelId)]) {
      if (!item) continue;
      item.successes += 1;
      if (usage?.outputTokens !== undefined) item.outputTokens += usage.outputTokens;
      if (usage?.totalTokens !== undefined) item.totalTokens += usage.totalTokens - inputTokens;
      else if (usage?.outputTokens !== undefined) item.totalTokens += usage.outputTokens;
    }
  }

  terminalFailure(providerId?: string, modelId?: string): void {
    this.totals.terminalFailures += 1;
    if (providerId) this.providers.get(providerId)!.terminalFailures += 1;
    if (modelId) this.models.get(modelId)!.terminalFailures += 1;
  }
  cancellation(): void { this.totals.cancellations += 1; }

  snapshot(currentQueueDepth: number): Record<string, unknown> {
    const now = Date.now();
    const status = new Map(this.registeredProviders.map((provider) => [provider.id, provider.status(now)]));
    return {
      started_at: new Date(this.startedAt).toISOString(),
      uptime_seconds: Math.floor((now - this.startedAt) / 1000),
      queue: { current_depth: currentQueueDepth, peak_depth: this.peakQueueDepth },
      totals: { ...this.totals },
      providers: [...this.providers].map(([id, value]) => {
        const health = status.get(id);
        return { id, ...value, configured: health?.configured ?? false, blocked_until: health?.blockedUntil ? new Date(health.blockedUntil).toISOString() : null };
      }),
      models: [...this.models].map(([id, value]) => {
        const providerId = this.registeredProviders.find((provider) => provider.listModels().some((model) => model.id === id))?.id ?? "unknown";
        const health = status.get(providerId)?.models.find((model) => model.id === id);
        return { id, provider: providerId, ...value, active: health?.active ?? 0, blocked_until: health?.blockedUntil ? new Date(health.blockedUntil).toISOString() : null, overflow_blocked_until: health?.overflowBlockedUntil ? new Date(health.overflowBlockedUntil).toISOString() : null };
      }),
      requested_models: [...this.requestedModels].map(([model, requests]) => ({ model, kind: model === "auto" ? "auto" : "explicit", requests })),
    };
  }
}
