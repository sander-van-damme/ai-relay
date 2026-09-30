import type { ModelRuntimeState, QuotaPolicy } from "./types.ts";

const MINUTE_MS = 60_000;
const DAY_MS = 86_400_000;

export function estimateInputTokens(body: Record<string, unknown>): number {
  const copy = { ...body };
  delete copy.model;
  delete copy.stream;
  const serialized = JSON.stringify(copy);
  return Math.max(1, Math.ceil(serialized.length / 4));
}

export function pruneQuotaEvents(events: ModelRuntimeState["events"], now: number): ModelRuntimeState["events"] {
  const cutoff = now - DAY_MS;
  return events.filter((event) => event.at > cutoff);
}

function waitForRequestWindow(
  events: ModelRuntimeState["events"],
  now: number,
  windowMs: number,
  limit: number | null,
): number {
  if (limit === null) return 0;
  const recent = events.filter((event) => event.at > now - windowMs);
  if (recent.length < limit) return 0;

  const index = Math.max(0, recent.length - Math.ceil(limit));
  const blocking = recent[index];
  return blocking ? Math.max(1, blocking.at + windowMs - now) : 0;
}

function waitForTokenWindow(
  events: ModelRuntimeState["events"],
  now: number,
  inputTokens: number,
  limit: number | null,
): number {
  if (limit === null) return 0;
  if (inputTokens > limit) return Number.POSITIVE_INFINITY;

  const recent = events
    .filter((event) => event.at > now - MINUTE_MS)
    .sort((a, b) => a.at - b.at);
  let total = recent.reduce((sum, event) => sum + event.inputTokens, 0);
  if (total + inputTokens <= limit) return 0;

  for (const event of recent) {
    total -= event.inputTokens;
    if (total + inputTokens <= limit) {
      return Math.max(1, event.at + MINUTE_MS - now);
    }
  }

  return Number.POSITIVE_INFINITY;
}

export function quotaCanEverHandle(policy: QuotaPolicy, inputTokens: number): boolean {
  return policy.limits.inputTokensPerMinute === null
    || inputTokens <= policy.limits.inputTokensPerMinute;
}

export function quotaDelayMs(
  policy: QuotaPolicy,
  state: ModelRuntimeState,
  inputTokens: number,
  now = Date.now(),
): number {
  if (policy.maxConcurrent !== null && state.active >= policy.maxConcurrent) {
    return Number.POSITIVE_INFINITY;
  }

  state.events = pruneQuotaEvents(state.events, now);
  const delays = [
    Math.max(0, state.blockedUntil - now),
    Math.max(0, state.lastStartedAt + policy.limits.minimumSpacingMs - now),
    waitForRequestWindow(state.events, now, MINUTE_MS, policy.limits.requestsPerMinute),
    waitForRequestWindow(state.events, now, DAY_MS, policy.limits.requestsPerDay),
    waitForTokenWindow(state.events, now, inputTokens, policy.limits.inputTokensPerMinute),
  ];

  return Math.max(...delays);
}

export function reserveQuota(
  state: ModelRuntimeState,
  inputTokens: number,
  now = Date.now(),
): void {
  state.events = pruneQuotaEvents(state.events, now);
  state.events.push({ at: now, inputTokens });
  state.lastStartedAt = now;
  state.active += 1;
}

export function parseRetryAfterMs(value: string | null, fallbackMs: number): number {
  if (!value) return fallbackMs;

  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.max(1_000, Math.ceil(seconds * 1000));
  }

  const date = Date.parse(value);
  if (Number.isFinite(date)) {
    return Math.max(1_000, date - Date.now());
  }

  return fallbackMs;
}
