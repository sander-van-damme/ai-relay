import { appendFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

type Level = "debug" | "info" | "warn" | "error";

export const logStartedAt = new Date();
export const sessionLogPath = join(tmpdir(), `ai-relay-${process.pid}-${logStartedAt.toISOString().replace(/[:.]/g, "-")}.jsonl`);
export const sessionLogFilename = `ai-relay-logs-${logStartedAt.toISOString().replace(/[:.]/g, "-")}.jsonl`;

const LEVEL_ORDER: Record<Level, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

function configuredLevel(): Level {
  const value = (process.env.LOG_LEVEL ?? "info").toLowerCase();
  return value === "debug" || value === "warn" || value === "error"
    ? value
    : "info";
}

export function log(level: Level, event: string, fields: Record<string, unknown> = {}): void {
  if (LEVEL_ORDER[level] < LEVEL_ORDER[configuredLevel()]) return;

  const line = JSON.stringify({
    timestamp: new Date().toISOString(),
    level,
    event,
    ...fields,
  });

  appendFileSync(sessionLogPath, `${line}\n`, { encoding: "utf8", mode: 0o600 });

  if (level === "error") {
    console.error(line);
  } else if (level === "warn") {
    console.warn(line);
  } else {
    console.log(line);
  }
}
