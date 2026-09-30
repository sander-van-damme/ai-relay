import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type {
  ModelConfig,
  ModelLimits,
  ProviderConfig,
  ProviderId,
  RelayConfig,
  ServerConfig,
} from "./types.ts";

const PROVIDER_IDS = new Set<ProviderId>(["google", "nvidia"]);

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function stringValue(value: unknown, label: string, fallback?: string): string {
  if (value === undefined && fallback !== undefined) return fallback;
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value.trim();
}

function numberValue(
  value: unknown,
  label: string,
  fallback: number,
  min: number,
  max: number,
): number {
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${label} must be a number between ${min} and ${max}`);
  }
  return value;
}

function positiveLimit(value: unknown, label: string): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be null or a positive number`);
  }
  return value;
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  return value === undefined ? fallback : Boolean(value);
}

function parseServer(value: unknown): ServerConfig {
  const raw = record(value ?? {}, "server");
  return {
    host: stringValue(raw.host, "server.host", "127.0.0.1"),
    port: numberValue(raw.port, "server.port", 8787, 1, 65535),
    heartbeatSeconds: numberValue(raw.heartbeatSeconds, "server.heartbeatSeconds", 15, 1, 300),
    retrySeconds: numberValue(raw.retrySeconds, "server.retrySeconds", 5, 1, 300),
    upstreamTimeoutSeconds: numberValue(
      raw.upstreamTimeoutSeconds,
      "server.upstreamTimeoutSeconds",
      300,
      1,
      3600,
    ),
    bodyLimitBytes: numberValue(raw.bodyLimitBytes, "server.bodyLimitBytes", 10 * 1024 * 1024, 1024, 100 * 1024 * 1024),
  };
}

function parseProvider(value: unknown, id: ProviderId): ProviderConfig {
  const raw = record(value, `providers.${id}`);
  const env = raw.apiKeyEnv;
  if (!Array.isArray(env) || env.length === 0 || env.some((entry) => typeof entry !== "string" || entry.trim().length === 0)) {
    throw new Error(`providers.${id}.apiKeyEnv must be a non-empty string array`);
  }

  return {
    baseUrl: stringValue(raw.baseUrl, `providers.${id}.baseUrl`).replace(/\/$/, ""),
    apiKeyEnv: env.map((entry) => String(entry).trim()),
  };
}

function parseLimits(value: unknown, label: string): ModelLimits {
  const raw = record(value ?? {}, label);
  return {
    requestsPerMinute: positiveLimit(raw.requestsPerMinute, `${label}.requestsPerMinute`),
    inputTokensPerMinute: positiveLimit(raw.inputTokensPerMinute, `${label}.inputTokensPerMinute`),
    requestsPerDay: positiveLimit(raw.requestsPerDay, `${label}.requestsPerDay`),
    minimumSpacingMs: numberValue(raw.minimumSpacingMs, `${label}.minimumSpacingMs`, 0, 0, 86_400_000),
  };
}

function parseModel(value: unknown, index: number): ModelConfig {
  const label = `models[${index}]`;
  const raw = record(value, label);
  const provider = stringValue(raw.provider, `${label}.provider`) as ProviderId;
  if (!PROVIDER_IDS.has(provider)) {
    throw new Error(`${label}.provider must be google or nvidia`);
  }

  return {
    id: stringValue(raw.id, `${label}.id`),
    provider,
    upstreamModel: stringValue(raw.upstreamModel, `${label}.upstreamModel`),
    enabled: booleanValue(raw.enabled, true),
    maxConcurrent: numberValue(raw.maxConcurrent, `${label}.maxConcurrent`, 1, 1, 64),
    limits: parseLimits(raw.limits, `${label}.limits`),
  };
}

export function parseConfig(value: unknown): RelayConfig {
  const raw = record(value, "config");
  const providersRaw = record(raw.providers, "providers");
  const modelsRaw = raw.models;
  if (!Array.isArray(modelsRaw) || modelsRaw.length === 0) {
    throw new Error("models must be a non-empty array");
  }

  const models = modelsRaw.map(parseModel);
  const ids = new Set<string>();
  for (const model of models) {
    if (ids.has(model.id)) throw new Error(`duplicate model id: ${model.id}`);
    ids.add(model.id);
  }

  if (!models.some((model) => model.enabled)) {
    throw new Error("at least one model must be enabled");
  }

  return {
    server: parseServer(raw.server),
    providers: {
      google: parseProvider(providersRaw.google, "google"),
      nvidia: parseProvider(providersRaw.nvidia, "nvidia"),
    },
    models,
  };
}

export async function loadConfig(path = process.env.AI_RELAY_CONFIG ?? "./config/models.json"): Promise<RelayConfig> {
  const fullPath = resolve(path);
  const text = await readFile(fullPath, "utf8");
  return parseConfig(JSON.parse(text) as unknown);
}

export function resolveProviderApiKey(provider: ProviderConfig): { env: string; value: string } | undefined {
  for (const name of provider.apiKeyEnv) {
    const value = process.env[name];
    if (value?.trim()) return { env: name, value: value.trim() };
  }
  return undefined;
}

export function relayApiKeys(): string[] {
  const keys = new Set<string>();
  const direct = process.env.RELAY_API_KEY?.trim();
  if (direct) keys.add(direct);

  for (const [name, value] of Object.entries(process.env)) {
    if (/^RELAY_API_KEY_\d+$/.test(name) && value?.trim()) keys.add(value.trim());
  }
  return [...keys];
}
