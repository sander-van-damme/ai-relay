export type QuotaLimitName = "requestsPerMinute" | "inputTokensPerMinute" | "requestsPerDay";

export interface QuotaLimits {
  requestsPerMinute: number | null;
  inputTokensPerMinute: number | null;
  requestsPerDay: number | null;
  minimumSpacingMs: number;
}

export interface QuotaPolicy {
  maxConcurrent: number | null;
  limits: QuotaLimits;
  dailyWindow: DailyWindow;
}

export type DailyWindow =
  | { type: "rolling" }
  | { type: "calendar-day"; timeZone: string };

export interface QuotaEvent {
  at: number;
  inputTokens: number;
}

export interface QuotaRuntimeState {
  active: number;
  blockedUntil: number;
  lastStartedAt: number;
  events: QuotaEvent[];
}

const MINUTE_MS = 60_000;
const DAY_MS = 86_400_000;

export function emptyQuotaState(): QuotaRuntimeState {
  return { active: 0, blockedUntil: 0, lastStartedAt: 0, events: [] };
}

function zonedParts(epochMs: number, timeZone: string): { year: number; month: number; day: number; hour: number; minute: number; second: number } {
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  });
  const values = Object.fromEntries(
    formatter.formatToParts(new Date(epochMs)).filter((part) => part.type !== "literal").map((part) => [part.type, part.value]),
  );
  return {
    year: Number(values.year),
    month: Number(values.month),
    day: Number(values.day),
    hour: Number(values.hour),
    minute: Number(values.minute),
    second: Number(values.second),
  };
}

function localDateTimeToEpoch(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute: number,
  second: number,
  timeZone: string,
): number {
  const target = Date.UTC(year, month - 1, day, hour, minute, second);
  let guess = target;
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const actual = zonedParts(guess, timeZone);
    const actualAsUtc = Date.UTC(actual.year, actual.month - 1, actual.day, actual.hour, actual.minute, actual.second);
    const delta = target - actualAsUtc;
    guess += delta;
    if (delta === 0) break;
  }
  return guess;
}

export function calendarDayBounds(now: number, timeZone: string): { start: number; end: number } {
  const local = zonedParts(now, timeZone);
  const start = localDateTimeToEpoch(local.year, local.month, local.day, 0, 0, 0, timeZone);
  const nextDate = new Date(Date.UTC(local.year, local.month - 1, local.day + 1));
  const end = localDateTimeToEpoch(
    nextDate.getUTCFullYear(),
    nextDate.getUTCMonth() + 1,
    nextDate.getUTCDate(),
    0,
    0,
    0,
    timeZone,
  );
  return { start, end };
}

function retentionCutoff(now: number, dailyWindow: DailyWindow): number {
  if (dailyWindow.type === "rolling") return now - DAY_MS;
  return calendarDayBounds(now, dailyWindow.timeZone).start;
}

export function pruneQuotaEvents(events: QuotaEvent[], now: number, dailyWindow: DailyWindow): QuotaEvent[] {
  const cutoff = Math.min(now - MINUTE_MS, retentionCutoff(now, dailyWindow));
  return events.filter((event) => event.at > cutoff);
}

function waitForRequestWindow(events: QuotaEvent[], now: number, windowMs: number, limit: number | null): number {
  if (limit === null) return 0;
  const recent = events.filter((event) => event.at > now - windowMs);
  if (recent.length < limit) return 0;
  const index = Math.max(0, recent.length - Math.ceil(limit));
  const blocking = recent[index];
  return blocking ? Math.max(1, blocking.at + windowMs - now) : 0;
}

function waitForDailyWindow(events: QuotaEvent[], now: number, limit: number | null, dailyWindow: DailyWindow): number {
  if (limit === null) return 0;
  if (dailyWindow.type === "rolling") return waitForRequestWindow(events, now, DAY_MS, limit);
  const bounds = calendarDayBounds(now, dailyWindow.timeZone);
  const count = events.filter((event) => event.at >= bounds.start).length;
  return count < limit ? 0 : Math.max(1, bounds.end - now);
}

function waitForTokenWindow(events: QuotaEvent[], now: number, inputTokens: number, limit: number | null): number {
  if (limit === null) return 0;
  if (inputTokens > limit) return Number.POSITIVE_INFINITY;

  const recent = events
    .filter((event) => event.at > now - MINUTE_MS)
    .sort((a, b) => a.at - b.at);
  let total = recent.reduce((sum, event) => sum + event.inputTokens, 0);
  if (total + inputTokens <= limit) return 0;

  for (const event of recent) {
    total -= event.inputTokens;
    if (total + inputTokens <= limit) return Math.max(1, event.at + MINUTE_MS - now);
  }
  return Number.POSITIVE_INFINITY;
}

export function effectiveInputCapacity(policy: QuotaPolicy, contextWindowTokens: number): number {
  return Math.min(contextWindowTokens, policy.limits.inputTokensPerMinute ?? Number.POSITIVE_INFINITY);
}

export function quotaCanEverHandle(policy: QuotaPolicy, inputTokens: number, contextWindowTokens = Number.POSITIVE_INFINITY): boolean {
  return inputTokens <= effectiveInputCapacity(policy, contextWindowTokens);
}

export function quotaDelayMs(policy: QuotaPolicy, state: QuotaRuntimeState, inputTokens: number, now = Date.now()): number {
  if (!quotaCanEverHandle(policy, inputTokens)) return Number.POSITIVE_INFINITY;
  if (policy.maxConcurrent !== null && state.active >= policy.maxConcurrent) return Number.POSITIVE_INFINITY;

  state.events = pruneQuotaEvents(state.events, now, policy.dailyWindow);
  return Math.max(
    Math.max(0, state.blockedUntil - now),
    Math.max(0, state.lastStartedAt + policy.limits.minimumSpacingMs - now),
    waitForRequestWindow(state.events, now, MINUTE_MS, policy.limits.requestsPerMinute),
    waitForDailyWindow(state.events, now, policy.limits.requestsPerDay, policy.dailyWindow),
    waitForTokenWindow(state.events, now, inputTokens, policy.limits.inputTokensPerMinute),
  );
}

export function quotaCanOverflow(
  policy: QuotaPolicy,
  state: QuotaRuntimeState,
  inputTokens: number,
  overflowLimits: ReadonlySet<QuotaLimitName>,
  contextWindowTokens = Number.POSITIVE_INFINITY,
  now = Date.now(),
): boolean {
  if (!quotaCanEverHandle(policy, inputTokens, contextWindowTokens)) return false;
  if (policy.maxConcurrent !== null && state.active >= policy.maxConcurrent) return false;

  state.events = pruneQuotaEvents(state.events, now, policy.dailyWindow);

  if (state.blockedUntil > now) return false;
  if (state.lastStartedAt + policy.limits.minimumSpacingMs > now) return false;

  const waits: Record<QuotaLimitName, number> = {
    requestsPerMinute: waitForRequestWindow(state.events, now, MINUTE_MS, policy.limits.requestsPerMinute),
    inputTokensPerMinute: waitForTokenWindow(state.events, now, inputTokens, policy.limits.inputTokensPerMinute),
    requestsPerDay: waitForDailyWindow(state.events, now, policy.limits.requestsPerDay, policy.dailyWindow),
  };

  let overflowNeeded = false;
  for (const [limit, waitMs] of Object.entries(waits) as Array<[QuotaLimitName, number]>) {
    if (waitMs <= 0) continue;
    if (!Number.isFinite(waitMs) || !overflowLimits.has(limit)) return false;
    overflowNeeded = true;
  }
  return overflowNeeded;
}

export function reserveQuota(policy: QuotaPolicy, state: QuotaRuntimeState, inputTokens: number, now = Date.now()): void {
  state.events = pruneQuotaEvents(state.events, now, policy.dailyWindow);
  state.events.push({ at: now, inputTokens });
  state.lastStartedAt = now;
  state.active += 1;
}

export function releaseQuota(state: QuotaRuntimeState): void {
  state.active = Math.max(0, state.active - 1);
}

export function parseRetryAfterMs(value: string | null, fallbackMs: number): number {
  if (!value) return fallbackMs;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return Math.max(1_000, Math.ceil(seconds * 1000));
  const date = Date.parse(value);
  if (Number.isFinite(date)) return Math.max(1_000, date - Date.now());
  return fallbackMs;
}
