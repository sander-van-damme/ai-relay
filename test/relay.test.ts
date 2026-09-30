import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import { optimizationWaitMs, RelayScheduler, selectOffer } from "../src/relay.ts";
import type {
  OfferRequest,
  Provider,
  ProviderExecutionResult,
  ProviderModelInfo,
  ProviderOffer,
  ProviderStatus,
} from "../src/providers/types.ts";
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
      retrySeconds: 5,
      upstreamTimeoutSeconds: 300,
      bodyLimitBytes: 1024 * 1024,
    },
  };
}

function job(id: string, response: FakeResponse): RelayJob {
  return {
    id,
    body: { model: "auto", testId: id, messages: [] },
    response: response as never,
    enqueuedAt: Date.now(),
    estimatedInputTokens: 10,
    requestedModel: "auto",
    stream: false,
    excludedModelIds: new Set<string>(),
    cancelled: false,
    bypassCount: 0,
    failureCount: 0,
    yieldOnce: false,
  };
}

class FakeProvider implements Provider {
  readonly credentialEnv = "FAKE_KEY";
  private blockedUntil = 0;
  private active = false;
  readonly executionOrder: string[] = [];
  failFirstFor = new Set<string>();
  blockOnFailure = true;

  readonly id: string;
  readonly priority: number;
  readonly capacity: number;

  constructor(id: string, priority: number, capacity: number) {
    this.id = id;
    this.priority = priority;
    this.capacity = capacity;
  }

  isConfigured(): boolean { return true; }
  listModels(): readonly ProviderModelInfo[] {
    return [{ id: `${this.id}/model`, providerId: this.id, inputCapacityTokens: this.capacity }];
  }
  getBestOffer(request: OfferRequest, now = Date.now()): ProviderOffer | null {
    if (request.requestedModel !== "auto" && request.requestedModel !== `${this.id}/model`) return null;
    if (request.excludedModelIds.has(`${this.id}/model`) || request.estimatedInputTokens > this.capacity) return null;
    return {
      providerId: this.id,
      providerPriority: this.priority,
      modelId: `${this.id}/model`,
      inputCapacityTokens: this.capacity,
      availableAt: this.active ? Number.POSITIVE_INFINITY : Math.max(now, this.blockedUntil),
    };
  }
  async execute(
    _offer: ProviderOffer,
    body: ChatCompletionRequest,
    _stream: boolean,
    _estimatedInputTokens: number,
    _signal: AbortSignal,
  ): Promise<ProviderExecutionResult> {
    const id = String(body.testId);
    this.executionOrder.push(id);
    this.active = true;
    await new Promise((resolve) => setTimeout(resolve, 0));
    this.active = false;
    if (this.failFirstFor.delete(id)) {
      this.blockedUntil = this.blockOnFailure ? Date.now() + 20_000 : 0;
      return { status: "retryable", reason: "test_failure", retryAt: this.blockedUntil || Date.now() };
    }
    return { status: "success", response: new Response(JSON.stringify({ id, provider: this.id })), release: () => undefined };
  }
  status(): ProviderStatus {
    return { id: this.id, configured: true, blockedUntil: this.blockedUntil || null, models: [{ id: `${this.id}/model`, active: 0, blockedUntil: null }] };
  }
}

test("optimization wait halves after every failed execution", () => {
  assert.equal(optimizationWaitMs(0), 15_000);
  assert.equal(optimizationWaitMs(1), 7_500);
  assert.equal(optimizationWaitMs(2), 3_750);
});

test("offer selection waits up to cutoff for smaller capacity", () => {
  const now = 1_000_000;
  const small: ProviderOffer = { providerId: "google", providerPriority: 10, modelId: "small", inputCapacityTokens: 16_000, availableAt: now + 12_000 };
  const large: ProviderOffer = { providerId: "nvidia", providerPriority: 20, modelId: "large", inputCapacityTokens: 32_000, availableAt: now };
  assert.equal(selectOffer([small, large], now, 15_000)?.modelId, "small");
  assert.equal(selectOffer([{ ...small, availableAt: now + 30_000 }, large], now, 15_000)?.modelId, "large");
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

test("retryable provider failure falls through to another provider instead of looping", async () => {
  const google = new FakeProvider("google", 10, 16_000);
  const nvidia = new FakeProvider("nvidia", 20, 32_000);
  google.failFirstFor.add("A");
  const scheduler = new RelayScheduler(config(), [google, nvidia]);
  const response = new FakeResponse();
  scheduler.enqueue(job("A", response));

  await new Promise((resolve) => setTimeout(resolve, 30));

  assert.deepEqual(google.executionOrder, ["A"]);
  assert.deepEqual(nvidia.executionOrder, ["A"]);
  assert.equal(response.writableEnded, true);
  assert.match(response.body, /nvidia/);
});
