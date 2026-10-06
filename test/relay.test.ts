import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import {
  MAX_RETRYABLE_FAILURES_PER_PATH,
  optimizationWaitMs,
  RelayScheduler,
  selectOffer,
  upstreamRejectionDetail,
} from "../src/relay.ts";
import type {
  OfferRequest,
  Provider,
  ProviderExecutionResult,
  ProviderModelInfo,
  ProviderOffer,
  ProviderOfferResult,
  ProviderStatus,
} from "../src/providers/index.ts";
import type { ChatCompletionRequest, RelayConfig, RelayJob } from "../src/types.ts";

class FakeResponse extends EventEmitter {
  writableEnded = false;
  destroyed = false;
  body = "";
  write(chunk: unknown): boolean { this.body += String(chunk ?? ""); return true; }
  end(chunk?: unknown): void {
    if (chunk !== undefined) this.body += String(chunk);
    this.writableEnded = true;
    this.emit("finish");
  }
}

function config(): RelayConfig {
  return {
    server: {
      host: "127.0.0.1",
      port: 8787,
      heartbeatSeconds: 15,
      upstreamTimeoutSeconds: 300,
      bodyLimitBytes: 1024 * 1024,
    },
  };
}

function job(id: string, response: FakeResponse): RelayJob {
  return {
    id,
    body: { model: "auto", testId: id, messages: [], testInputTokens: 10 },
    response: response as never,
    enqueuedAt: Date.now(),
    requestedModel: "auto",
    stream: false,
    excludedModelIds: new Set<string>(),
    excludedProviderIds: new Set<string>(),
    retryableFailureCounts: new Map<string, number>(),
    cancelled: false,
    bypassCount: 0,
    failureCount: 0,
    yieldOnce: false,
  };
}

class FakeProvider implements Provider {
  blockedUntil = 0;
  modelBlockedUntil = 0;
  configured = true;
  private active = false;
  readonly executionOrder: string[] = [];
  failFirstFor = new Set<string>();
  failAlwaysFor = new Set<string>();
  blockOnFailure = true;
  supportsOverflow = false;
  standardAvailableAt = 0;
  readonly executionKinds: string[] = [];
  executionDelayMs = 0;

  readonly id: string;
  readonly priority: number;
  readonly capacity: number;

  constructor(id: string, priority: number, capacity: number) {
    this.id = id;
    this.priority = priority;
    this.capacity = capacity;
  }

  isConfigured(): boolean { return this.configured; }
  listModels(): readonly ProviderModelInfo[] {
    return [{ id: `${this.id}/model`, providerId: this.id, inputCapacityTokens: this.capacity }];
  }
  async getBestOffer(request: OfferRequest, now = Date.now()): Promise<ProviderOfferResult> {
    const modelId = `${this.id}/model`;
    if (request.offerKind === "overflow" && !this.supportsOverflow) {
      return { status: "no_offer", providerId: this.id, reason: "no_eligible_model" };
    }
    if (
      request.excludedModelIds.has(modelId)
      || (request.requestedModel !== "auto" && request.requestedModel !== modelId)
    ) {
      return { status: "no_offer", providerId: this.id, reason: "no_eligible_model" };
    }

    const inputTokens = Number(request.body.testInputTokens ?? 10);
    if (inputTokens > this.capacity) {
      return { status: "no_offer", providerId: this.id, reason: "request_exceeds_capacity" };
    }

    return {
      status: "offer",
      offer: {
        kind: request.offerKind,
        providerId: this.id,
        providerPriority: this.priority,
        modelId,
        inputTokens,
        inputCapacityTokens: this.capacity,
        availableAt: request.offerKind === "overflow"
          ? now
          : this.active
            ? Number.POSITIVE_INFINITY
            : Math.max(now, this.standardAvailableAt, this.blockedUntil),
      },
    };
  }
  async execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    _stream: boolean,
    _signal: AbortSignal,
  ): Promise<ProviderExecutionResult> {
    const id = String(body.testId);
    this.executionOrder.push(id);
    this.executionKinds.push(offer.kind);
    this.active = true;
    await new Promise((resolve) => setTimeout(resolve, this.executionDelayMs));
    this.active = false;
    if (this.failFirstFor.delete(id) || this.failAlwaysFor.has(id)) {
      this.blockedUntil = this.blockOnFailure ? Date.now() + 20_000 : 0;
      return { status: "retryable", scope: "provider", reason: "test_failure", retryAt: this.blockedUntil || Date.now() };
    }
    return { status: "success", response: new Response(JSON.stringify({ id, provider: this.id })), release: () => undefined };
  }
  status(): ProviderStatus {
    return {
      id: this.id,
      configured: this.configured,
      blockedUntil: this.blockedUntil || null,
      models: [{
        id: `${this.id}/model`,
        active: 0,
        blockedUntil: this.modelBlockedUntil || null,
        overflowBlockedUntil: null,
      }],
    };
  }
}


class StreamingProvider extends FakeProvider {
  releaseCount = 0;
  cancelCount = 0;
  lastSignal?: AbortSignal;
  stall = false;
  ignoreAbort = false;

  override async execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    _stream: boolean,
    signal: AbortSignal,
  ): Promise<ProviderExecutionResult> {
    const id = String(body.testId);
    this.executionOrder.push(id);
    this.executionKinds.push(offer.kind);
    this.lastSignal = signal;

    const encoder = new TextEncoder();
    const stalled = this.stall;
    const ignoreAbort = this.ignoreAbort;
    const provider = this;
    const response = new Response(new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"choices":[]}\\n\\n'));
        if (!stalled) {
          controller.enqueue(encoder.encode("data: [DONE]\\n\\n"));
          controller.close();
          return;
        }
        if (!ignoreAbort) {
          signal.addEventListener("abort", () => {
            controller.error(signal.reason ?? new Error("aborted"));
          }, { once: true });
        }
      },
      cancel() {
        provider.cancelCount += 1;
      },
    }));

    let released = false;
    return {
      status: "success",
      response,
      release: () => {
        if (released) return;
        released = true;
        this.releaseCount += 1;
      },
    };
  }
}

class ThrowingProvider extends FakeProvider {
  readonly mode: "sync" | "async";

  constructor(id: string, priority: number, capacity: number, mode: "sync" | "async") {
    super(id, priority, capacity);
    this.mode = mode;
  }

  override execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    _stream: boolean,
    _signal: AbortSignal,
  ): Promise<ProviderExecutionResult> {
    const id = String(body.testId);
    this.executionOrder.push(id);
    this.executionKinds.push(offer.kind);
    const error = new Error(`${this.mode} provider execute failure`);
    if (this.mode === "sync") throw error;
    return Promise.reject(error);
  }
}

test("upstream rejection details extract structured errors and redact secrets", () => {
  assert.equal(
    upstreamRejectionDetail(JSON.stringify({
      error: {
        code: 400,
        status: "INVALID_ARGUMENT",
        message: "Unsupported generation option.",
      },
    })),
    "code=400 status=INVALID_ARGUMENT message=Unsupported generation option.",
  );

  const apiKey = "AIza12345678901234567890123456789012345";
  const detail = upstreamRejectionDetail(`400 Bad Request using ${apiKey} and Bearer abc.def.ghi`);
  assert.ok(detail);
  assert.doesNotMatch(detail, /AIza/);
  assert.doesNotMatch(detail, /abc\.def\.ghi/);
  assert.match(detail, /REDACTED/);
});

test("upstream rejection details are bounded for journald", () => {
  const detail = upstreamRejectionDetail("x".repeat(5_000));
  assert.ok(detail);
  assert.equal(detail.length, 1_000);
  assert.ok(detail.endsWith("…"));
});

test("optimization wait halves after every failed execution", () => {
  assert.equal(optimizationWaitMs(0), 15_000);
  assert.equal(optimizationWaitMs(1), 7_500);
  assert.equal(optimizationWaitMs(2), 3_750);
});

test("offer selection waits up to cutoff for smaller capacity", () => {
  const now = 1_000_000;
  const small: ProviderOffer = { kind: "standard", providerId: "google", providerPriority: 10, modelId: "small", inputTokens: 100, inputCapacityTokens: 16_000, availableAt: now + 12_000 };
  const large: ProviderOffer = { kind: "standard", providerId: "nvidia", providerPriority: 20, modelId: "large", inputTokens: 100, inputCapacityTokens: 32_000, availableAt: now };
  assert.equal(selectOffer([small, large], now, 15_000)?.modelId, "small");
  assert.equal(selectOffer([{ ...small, availableAt: now + 30_000 }, large], now, 15_000)?.modelId, "large");
});

test("configured model catalog excludes providers without credentials", () => {
  const configured = new FakeProvider("configured", 10, 32_000);
  const missing = new FakeProvider("missing", 20, 1_048_576);
  missing.configured = false;

  const scheduler = new RelayScheduler(config(), [configured, missing]);

  assert.deepEqual(
    scheduler.listConfiguredModels().map((model) => model.id),
    ["configured/model"],
  );
  assert.equal(scheduler.hasModel("configured/model"), true);
  assert.equal(scheduler.hasModel("missing/model"), false);
});

test("auto input capacity follows the largest configured model outside failure cooldown", () => {
  const now = 1_000_000;
  const small = new FakeProvider("small", 10, 250_000);
  const large = new FakeProvider("large", 20, 1_048_576);
  const missing = new FakeProvider("missing", 30, 2_000_000);
  missing.configured = false;

  const scheduler = new RelayScheduler(config(), [small, large, missing]);

  assert.equal(scheduler.autoInputCapacityTokens(now), 1_048_576);

  large.modelBlockedUntil = now + 30_000;
  assert.equal(scheduler.autoInputCapacityTokens(now), 250_000);

  large.modelBlockedUntil = 0;
  large.blockedUntil = now + 30_000;
  assert.equal(scheduler.autoInputCapacityTokens(now), 250_000);

  small.blockedUntil = now + 30_000;
  assert.equal(scheduler.autoInputCapacityTokens(now), null);
});

test("a failed old request yields one dispatch turn to a younger runnable request", async () => {
  const p = new FakeProvider("only", 10, 32_000);
  p.failFirstFor.add("A");
  p.blockOnFailure = false;
  const scheduler = new RelayScheduler(config(), [p]);
  const aResponse = new FakeResponse();
  const bResponse = new FakeResponse();
  const a = job("A", aResponse);
  const b = job("B", bResponse);
  b.enqueuedAt = a.enqueuedAt + 1;

  scheduler.enqueue(a);
  scheduler.enqueue(b);
  await new Promise((resolve) => setTimeout(resolve, 30));

  assert.deepEqual(p.executionOrder.slice(0, 2), ["A", "B"]);
  assert.equal(bResponse.writableEnded, true);
  assert.equal(a.failureCount, 1);
  a.cancelled = true;
});



test("stream keeps upstream abort controller until normal completion", async () => {
  const provider = new StreamingProvider("stream", 10, 32_000);
  const scheduler = new RelayScheduler(config(), [provider]);
  const response = new FakeResponse();
  const request = job("stream-normal", response);
  request.stream = true;

  scheduler.enqueue(request);
  await new Promise((resolve) => setTimeout(resolve, 30));

  assert.equal(response.writableEnded, true);
  assert.equal(provider.releaseCount, 1);
  assert.equal(provider.lastSignal?.aborted, false);
  assert.equal(request.upstreamAbort, undefined);
});

test("client cancellation exits a stalled non-abort-aware stream read", async () => {
  const provider = new StreamingProvider("stream", 10, 32_000);
  provider.stall = true;
  provider.ignoreAbort = true;
  const scheduler = new RelayScheduler(config(), [provider]);
  const response = new FakeResponse();
  const request = job("stream-client-cancel", response);
  request.stream = true;

  scheduler.enqueue(request);
  await new Promise((resolve) => setTimeout(resolve, 10));

  assert.ok(request.upstreamAbort);
  request.cancelled = true;
  request.upstreamAbort.abort(new Error("client disconnected"));
  await new Promise((resolve) => setTimeout(resolve, 20));

  assert.equal(provider.lastSignal?.aborted, true);
  assert.equal(provider.cancelCount, 1);
  assert.equal(provider.releaseCount, 1);
  assert.equal(request.upstreamAbort, undefined);
});

test("upstream timeout interrupts a stalled non-abort-aware stream read", async () => {
  const provider = new StreamingProvider("stream", 10, 32_000);
  provider.stall = true;
  provider.ignoreAbort = true;
  const timeoutConfig = config();
  timeoutConfig.server.upstreamTimeoutSeconds = 0.02;
  const scheduler = new RelayScheduler(timeoutConfig, [provider]);
  const response = new FakeResponse();
  const request = job("stream-timeout", response);
  request.stream = true;

  scheduler.enqueue(request);
  await new Promise((resolve) => setTimeout(resolve, 60));

  assert.equal(provider.lastSignal?.aborted, true);
  assert.equal(provider.cancelCount, 1);
  assert.equal(response.writableEnded, true);
  assert.equal(provider.releaseCount, 1);
  assert.equal(request.upstreamAbort, undefined);
});

test("retry dispatch timing separates queue wait from request age", async () => {
  const google = new FakeProvider("google", 10, 16_000);
  const nvidia = new FakeProvider("nvidia", 20, 32_000);
  google.failFirstFor.add("timing-A");
  google.blockOnFailure = false;
  google.executionDelayMs = 25;

  const scheduler = new RelayScheduler(config(), [google, nvidia]);
  const response = new FakeResponse();
  const request = job("timing-A", response);
  const lines: Array<Record<string, unknown>> = [];
  const originalLog = console.log;
  console.log = (value?: unknown): void => {
    if (typeof value !== "string") return;
    try {
      const parsed = JSON.parse(value) as Record<string, unknown>;
      if (parsed.request_id === request.id) lines.push(parsed);
    } catch {
      // Ignore unrelated console output.
    }
  };

  try {
    scheduler.enqueue(request);
    await new Promise((resolve) => setTimeout(resolve, 80));
  } finally {
    console.log = originalLog;
  }

  const dispatches = lines.filter((line) => line.event === "queue_dispatched");
  assert.equal(dispatches.length, 2);

  const first = dispatches[0]!;
  const second = dispatches[1]!;
  assert.equal(typeof first.queue_wait_ms, "number");
  assert.equal(typeof first.request_age_ms, "number");
  assert.equal(typeof second.queue_wait_ms, "number");
  assert.equal(typeof second.request_age_ms, "number");
  assert.equal("queue_ms" in first, false);
  assert.equal("queue_ms" in second, false);

  const secondQueueWait = Number(second.queue_wait_ms);
  const secondRequestAge = Number(second.request_age_ms);
  assert.ok(secondRequestAge - secondQueueWait >= 15);
  assert.equal(response.writableEnded, true);
});

test("synchronous provider.execute exceptions fail over under the finite provider retry policy", async () => {
  const throwing = new ThrowingProvider("throwing", 10, 32_000, "sync");
  const backup = new FakeProvider("backup", 20, 32_000);
  const scheduler = new RelayScheduler(config(), [throwing, backup]);
  const response = new FakeResponse();
  const request = job("sync-throw", response);
  const lines: Array<Record<string, unknown>> = [];
  const originalLog = console.log;
  console.log = (value?: unknown): void => {
    if (typeof value !== "string") return;
    try {
      const parsed = JSON.parse(value) as Record<string, unknown>;
      if (parsed.request_id === request.id) lines.push(parsed);
    } catch {
      // Ignore unrelated output.
    }
  };

  try {
    scheduler.enqueue(request);
    await new Promise((resolve) => setTimeout(resolve, 50));
  } finally {
    console.log = originalLog;
  }

  assert.deepEqual(throwing.executionOrder, ["sync-throw"]);
  assert.deepEqual(backup.executionOrder, ["sync-throw"]);
  assert.equal(response.writableEnded, true);
  assert.match(response.body, /backup/);
  assert.equal(request.upstreamAbort, undefined);

  const snapshot = scheduler.observability.snapshot(0) as any;
  assert.equal(snapshot.totals.attempts, 2);
  assert.equal(snapshot.totals.failedAttempts, 1);
  assert.equal(snapshot.totals.successes, 1);
  assert.equal(snapshot.totals.terminalFailures, 0);

  const exception = lines.find((line) => line.event === "provider_execute_exception");
  assert.ok(exception);
  assert.equal(exception.request_id, request.id);
  assert.equal(exception.provider, "throwing");
  assert.equal(exception.relay_model, "throwing/model");
  assert.match(String(exception.error), /sync provider execute failure/);
  assert.equal(typeof exception.attempt_ms, "number");
});

test("asynchronous provider.execute exceptions on an explicit model exhaust the finite path budget", async () => {
  const throwing = new ThrowingProvider("throwing", 10, 32_000, "async");
  const scheduler = new RelayScheduler(config(), [throwing]);
  const response = new FakeResponse();
  const request = job("async-throw", response);
  request.requestedModel = "throwing/model";
  request.body.model = "throwing/model";

  scheduler.enqueue(request);
  await new Promise((resolve) => setTimeout(resolve, 60));

  assert.equal(throwing.executionOrder.length, MAX_RETRYABLE_FAILURES_PER_PATH);
  assert.equal(response.writableEnded, true);
  assert.match(response.body, /upstream_unavailable/);
  assert.equal(request.upstreamAbort, undefined);
  assert.equal(request.retryableFailureCounts.get("provider:throwing"), MAX_RETRYABLE_FAILURES_PER_PATH);
  assert.equal(request.excludedProviderIds.has("throwing"), true);

  const snapshot = scheduler.observability.snapshot(0) as any;
  assert.equal(snapshot.totals.attempts, MAX_RETRYABLE_FAILURES_PER_PATH);
  assert.equal(snapshot.totals.failedAttempts, MAX_RETRYABLE_FAILURES_PER_PATH);
  assert.equal(snapshot.totals.successes, 0);
  assert.equal(snapshot.totals.terminalFailures, 1);
});

test("retryable provider failure falls through to another provider instead of looping", async () => {
  const google = new FakeProvider("google", 10, 16_000);
  const nvidia = new FakeProvider("nvidia", 20, 32_000);
  google.failFirstFor.add("A");
  google.blockOnFailure = false;
  const scheduler = new RelayScheduler(config(), [google, nvidia]);
  const response = new FakeResponse();
  scheduler.enqueue(job("A", response));

  await new Promise((resolve) => setTimeout(resolve, 30));

  assert.deepEqual(google.executionOrder, ["A"]);
  assert.deepEqual(nvidia.executionOrder, ["A"]);
  assert.equal(response.writableEnded, true);
  assert.match(response.body, /nvidia/);
});

test("retryable failures exhaust a finite per-request path budget", async () => {
  const p = new FakeProvider("only", 10, 32_000);
  p.failAlwaysFor.add("A");
  p.blockOnFailure = false;
  const scheduler = new RelayScheduler(config(), [p]);
  const response = new FakeResponse();
  scheduler.enqueue(job("A", response));

  await new Promise((resolve) => setTimeout(resolve, 50));

  assert.equal(p.executionOrder.length, MAX_RETRYABLE_FAILURES_PER_PATH);
  assert.equal(response.writableEnded, true);
  assert.match(response.body, /upstream_unavailable/);
});

test("standard offer inside the optimization window beats immediate overflow", async () => {
  const standard = new FakeProvider("standard", 20, 32_000);
  standard.standardAvailableAt = Date.now() + 5;
  const overflow = new FakeProvider("overflow", 10, 16_000);
  overflow.supportsOverflow = true;
  overflow.standardAvailableAt = Date.now() + 60_000;

  const scheduler = new RelayScheduler(config(), [overflow, standard]);
  const response = new FakeResponse();
  scheduler.enqueue(job("A", response));

  await new Promise((resolve) => setTimeout(resolve, 30));

  assert.deepEqual(standard.executionOrder, ["A"]);
  assert.deepEqual(standard.executionKinds, ["standard"]);
  assert.deepEqual(overflow.executionOrder, []);
  assert.equal(response.writableEnded, true);
});

test("overflow is used as an immediate last resort before a long quota wait", async () => {
  const p = new FakeProvider("overflow", 10, 32_000);
  p.supportsOverflow = true;
  p.standardAvailableAt = Date.now() + 60_000;

  const scheduler = new RelayScheduler(config(), [p]);
  const response = new FakeResponse();
  scheduler.enqueue(job("A", response));

  await new Promise((resolve) => setTimeout(resolve, 30));

  assert.deepEqual(p.executionOrder, ["A"]);
  assert.deepEqual(p.executionKinds, ["overflow"]);
  assert.equal(response.writableEnded, true);
});


test("terminal capacity errors use provider offer assessments", async () => {
  const p = new FakeProvider("only", 10, 5);
  const scheduler = new RelayScheduler(config(), [p]);
  const response = new FakeResponse();
  scheduler.enqueue(job("too-large", response));

  await new Promise((resolve) => setTimeout(resolve, 20));

  assert.equal(response.writableEnded, true);
  assert.match(response.body, /request_exceeds_provider_capacity/);
  assert.deepEqual(p.executionOrder, []);
});
