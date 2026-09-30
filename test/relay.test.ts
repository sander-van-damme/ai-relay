import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";
import { RelayScheduler } from "../src/relay.ts";
import type { RelayConfig, RelayJob } from "../src/types.ts";

class FakeResponse extends EventEmitter {
  writableEnded = false;
  destroyed = false;
  body = "";

  write(chunk: unknown): boolean {
    this.body += String(chunk ?? "");
    return true;
  }

  end(chunk?: unknown): void {
    if (chunk !== undefined) this.body += String(chunk);
    this.writableEnded = true;
    this.emit("finish");
  }
}

function job(id: string, tokens: number, response: FakeResponse): RelayJob {
  return {
    id,
    body: { model: "auto", messages: [] },
    response: response as never,
    enqueuedAt: Date.now(),
    estimatedInputTokens: tokens,
    requestedModel: "auto",
    stream: false,
    excludedModels: new Set<string>(),
    cancelled: false,
    bypassCount: 0,
  };
}

const config: RelayConfig = {
  server: {
    host: "127.0.0.1",
    port: 8787,
    heartbeatSeconds: 15,
    retrySeconds: 5,
    upstreamTimeoutSeconds: 300,
    bodyLimitBytes: 1024 * 1024,
  },
  providers: {
    google: {
      baseUrl: "https://example.test/v1",
      maxConcurrent: null,
      limits: {
        requestsPerMinute: null,
        inputTokensPerMinute: 100,
        requestsPerDay: null,
        minimumSpacingMs: 0,
      },
    },
    nvidia: {
      baseUrl: "https://example.test/v1",
      maxConcurrent: null,
      limits: {
        requestsPerMinute: null,
        inputTokensPerMinute: null,
        requestsPerDay: null,
        minimumSpacingMs: 0,
      },
    },
  },
  models: [
    {
      id: "google/test",
      provider: "google",
      upstreamModel: "test",
      enabled: true,
      maxConcurrent: null,
      limits: {
        requestsPerMinute: null,
        inputTokensPerMinute: null,
        requestsPerDay: null,
        minimumSpacingMs: 0,
      },
    },
  ],
};

test("a smaller runnable request can bypass an older request waiting for provider TPM", async () => {
  const originalKey = process.env.GEMINI_API_KEY;
  const originalFetch = globalThis.fetch;
  process.env.GEMINI_API_KEY = "test-key";
  globalThis.fetch = async () => new Response('{"ok":true}', { status: 200 });

  try {
    const scheduler = new RelayScheduler(config);
    const providerState = scheduler.providerStates.get("google");
    if (!providerState) throw new Error("google provider state missing");
    providerState.events.push({ at: Date.now() - 1_000, inputTokens: 80 });

    const largeResponse = new FakeResponse();
    const smallResponse = new FakeResponse();
    const large = job("large", 50, largeResponse);
    const small = job("small", 20, smallResponse);

    scheduler.enqueue(large);
    scheduler.enqueue(small);
    await new Promise((resolve) => setTimeout(resolve, 25));

    assert.equal(smallResponse.writableEnded, true);
    assert.equal(largeResponse.writableEnded, false);
    assert.equal(large.bypassCount, 1);
    assert.equal(scheduler.queue[0]?.id, "large");
    large.cancelled = true;
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalKey;
  }
});
