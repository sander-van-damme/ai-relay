import type { ServerResponse } from "node:http";

export interface ServerConfig {
  host: string;
  port: number;
  heartbeatSeconds: number;
  retrySeconds: number;
  upstreamTimeoutSeconds: number;
  bodyLimitBytes: number;
}

export interface RelayConfig {
  server: ServerConfig;
}

export interface ChatCompletionRequest extends Record<string, unknown> {
  model?: string;
  stream?: boolean;
  messages?: unknown;
}

export interface RelayJob {
  id: string;
  body: ChatCompletionRequest;
  response: ServerResponse;
  enqueuedAt: number;
  estimatedInputTokens: number;
  requestedModel: string;
  stream: boolean;
  excludedModelIds: Set<string>;
  cancelled: boolean;
  bypassCount: number;
  failureCount: number;
  yieldOnce: boolean;
  upstreamAbort?: AbortController;
  heartbeatTimer?: NodeJS.Timeout;
}
