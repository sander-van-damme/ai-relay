import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { RelayConfig, ServerConfig } from "./types.ts";

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

function numberValue(value: unknown, label: string, fallback: number, min: number, max: number): number {
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${label} must be a number between ${min} and ${max}`);
  }
  return value;
}

function parseServer(value: unknown): ServerConfig {
  const raw = record(value ?? {}, "server");
  return {
    host: stringValue(raw.host, "server.host", "0.0.0.0"),
    port: numberValue(raw.port, "server.port", 8787, 1, 65535),
    heartbeatSeconds: numberValue(raw.heartbeatSeconds, "server.heartbeatSeconds", 15, 1, 300),
    upstreamTimeoutSeconds: numberValue(raw.upstreamTimeoutSeconds, "server.upstreamTimeoutSeconds", 300, 1, 3600),
    bodyLimitBytes: numberValue(raw.bodyLimitBytes, "server.bodyLimitBytes", 10 * 1024 * 1024, 1024, 100 * 1024 * 1024),
  };
}

export function parseConfig(value: unknown): RelayConfig {
  const raw = record(value, "config");
  return { server: parseServer(raw.server) };
}

export async function loadConfig(path = process.env.AI_RELAY_CONFIG ?? "./config/relay.json"): Promise<RelayConfig> {
  const text = await readFile(resolve(path), "utf8");
  return parseConfig(JSON.parse(text) as unknown);
}
