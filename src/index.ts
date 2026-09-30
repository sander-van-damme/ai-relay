import { randomUUID } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { isAuthorized } from "./auth.ts";
import { loadConfig, relayApiKeys, resolveProviderApiKey } from "./config.ts";
import { log } from "./log.ts";
import { estimateInputTokens } from "./quota.ts";
import { RelayScheduler } from "./relay.ts";
import type { ChatCompletionRequest, RelayConfig, RelayJob } from "./types.ts";

function json(response: ServerResponse, status: number, value: unknown): void {
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  response.end(JSON.stringify(value));
}

function openAiHttpError(response: ServerResponse, status: number, code: string, message: string): void {
  json(response, status, { error: { type: "relay_error", code, message } });
}

async function readJson(request: IncomingMessage, limitBytes: number): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > limitBytes) throw new Error("request_too_large");
    chunks.push(buffer);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  return JSON.parse(text);
}

function startPersistentResponse(response: ServerResponse, stream: boolean, heartbeatSeconds: number): NodeJS.Timeout {
  if (stream) {
    response.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-store",
      connection: "keep-alive",
      "x-accel-buffering": "no",
      "x-ai-relay-queued": "true",
    });
    response.write(": ai-relay queued\n\n");
  } else {
    response.writeHead(200, {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "x-accel-buffering": "no",
      "x-ai-relay-queued": "true",
    });
    response.write("\n");
  }

  const timer = setInterval(() => {
    if (response.writableEnded || response.destroyed) return;
    response.write(stream ? ": ai-relay waiting\n\n" : "\n");
  }, heartbeatSeconds * 1000);
  timer.unref();
  response.once("close", () => clearInterval(timer));
  response.once("finish", () => clearInterval(timer));
  return timer;
}

function modelsPayload(config: RelayConfig): Record<string, unknown> {
  const created = Math.floor(Date.now() / 1000);
  const concrete = config.models.filter((model) => model.enabled).map((model) => ({
    id: model.id,
    object: "model",
    created,
    owned_by: model.provider,
  }));

  return {
    object: "list",
    data: [
      { id: "auto", object: "model", created, owned_by: "ai-relay" },
      ...concrete,
    ],
  };
}

async function main(): Promise<void> {
  const config = await loadConfig();
  const relayKeys = relayApiKeys();
  if (relayKeys.length === 0) throw new Error("At least one RELAY_API_KEY or RELAY_API_KEY_<number> is required");

  const scheduler = new RelayScheduler(config);
  for (const [providerId, provider] of Object.entries(config.providers)) {
    const credential = resolveProviderApiKey(provider);
    log(credential ? "info" : "warn", "provider_config", {
      provider: providerId,
      configured: Boolean(credential),
      credential_env: credential?.env ?? null,
    });
  }

  const server = createServer(async (request, response) => {
    const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);

    if (!isAuthorized(request, relayKeys)) {
      openAiHttpError(response, 401, "unauthorized", "Invalid or missing relay API key.");
      return;
    }

    if (request.method === "GET" && url.pathname === "/") {
      json(response, 200, {
        name: "ai-relay",
        mode: "self-hosted-static-config",
        endpoints: ["/health", "/v1/models", "/v1/chat/completions"],
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/health") {
      json(response, 200, { status: "ok", ...scheduler.status() });
      return;
    }

    if (request.method === "GET" && url.pathname === "/v1/models") {
      json(response, 200, modelsPayload(config));
      return;
    }

    if (request.method !== "POST" || url.pathname !== "/v1/chat/completions") {
      openAiHttpError(response, 404, "not_found", "Endpoint not found.");
      return;
    }

    let body: ChatCompletionRequest;
    try {
      body = (await readJson(request, config.server.bodyLimitBytes)) as ChatCompletionRequest;
    } catch (error) {
      if (error instanceof Error && error.message === "request_too_large") {
        openAiHttpError(response, 413, "request_too_large", "Request body exceeds configured limit.");
      } else {
        openAiHttpError(response, 400, "invalid_json", "Request body must be valid JSON.");
      }
      return;
    }

    const requestedModel = typeof body.model === "string" && body.model.trim() ? body.model.trim() : "auto";
    const stream = body.stream === true;
    const isKnown = requestedModel === "auto" || ["relay/auto", "free", "best-free", "best"].includes(requestedModel)
      || config.models.some((model) => model.enabled && model.id === requestedModel);
    if (!isKnown) {
      openAiHttpError(response, 404, "model_not_found", `Configured model not found: ${requestedModel}`);
      return;
    }

    const heartbeatTimer = startPersistentResponse(response, stream, config.server.heartbeatSeconds);

    const job: RelayJob = {
      id: randomUUID(),
      body: { ...body, model: requestedModel, stream },
      response,
      enqueuedAt: Date.now(),
      estimatedInputTokens: estimateInputTokens(body),
      requestedModel,
      stream,
      excludedModels: new Set<string>(),
      cancelled: false,
      heartbeatTimer,
    };

    response.once("close", () => {
      if (response.writableEnded) return;
      job.cancelled = true;
      job.upstreamAbort?.abort(new Error("client disconnected"));
      log("info", "client_disconnected", { request_id: job.id });
    });

    scheduler.enqueue(job);
  });

  server.keepAliveTimeout = Math.max(server.keepAliveTimeout, 75_000);
  server.requestTimeout = 0;
  server.headersTimeout = 65_000;

  server.listen(config.server.port, config.server.host, () => {
    log("info", "server_started", {
      host: config.server.host,
      port: config.server.port,
      models: config.models.filter((model) => model.enabled).map((model) => model.id),
    });
  });

  const shutdown = (signal: string) => {
    log("info", "server_stopping", { signal });
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(1), 10_000).unref();
  };
  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));
}

main().catch((error) => {
  log("error", "startup_failed", {
    error: error instanceof Error ? error.stack ?? error.message : String(error),
  });
  process.exitCode = 1;
});
