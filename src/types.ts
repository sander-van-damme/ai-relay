import type { ServerResponse } from "node:http";

export type ProviderId = "google" | "nvidia";

export interface ServerConfig {
  host: string;
  port: number;
  heartbeatSeconds: number;
  retrySeconds: number;
  upstreamTimeoutSeconds: number;
  bodyLimitBytes: number;
}

export interface ProviderConfig {
  baseUrl: string;
  apiKeyEnv: string[];
}

export interface ModelLimits {
  requestsPerMinute: number | null;
  inputTokensPerMinute: number | null;
  requestsPerDay: number | null;
  minimumSpacingMs: number;
}

export interface ModelConfig {
  id: string;
  provider: ProviderId;
  upstreamModel: string;
  enabled: boolean;
  maxConcurrent: number;
  limits: ModelLimits;
}

export interface RelayConfig {
  server: ServerConfig;
  providers: Record<ProviderId, ProviderConfig>;
  models: ModelConfig[];
}

export interface ChatCompletionRequest extends Record<string, unknown> {
  model?: string;
  stream?: boolean;
  messages?: unknown;
}

export interface QuotaEvent {
  at: number;
  inputTokens: number;
}

export interface ModelRuntimeState {
  active: number;
  blockedUntil: number;
  lastStartedAt: number;
  events: QuotaEvent[];
}

export interface RelayJob {
  id: string;
  body: ChatCompletionRequest;
  response: ServerResponse;
  enqueuedAt: number;
  estimatedInputTokens: number;
  requestedModel: string;
  stream: boolean;
  excludedModels: Set<string>;
  cancelled: boolean;
  upstreamAbort?: AbortController;
  heartbeatTimer?: NodeJS.Timeout;
}
