import type { QuotaPolicy } from "../shared/quota.ts";
import { OpenAICompatibleProvider, type ManagedModel } from "../shared/openai-compatible.ts";

const GOOGLE_DAY = { type: "calendar-day", timeZone: "America/Los_Angeles" } as const;
const DEFAULT_RETRY_MS = 5_000;
const PROVIDER_FAILURE_COOLDOWN_MS = 15_000;

function quota(rpm: number, tpm: number, rpd: number): QuotaPolicy {
  return {
    maxConcurrent: null,
    dailyWindow: GOOGLE_DAY,
    limits: {
      requestsPerMinute: rpm,
      inputTokensPerMinute: tpm,
      requestsPerDay: rpd,
      minimumSpacingMs: 0,
    },
  };
}

// Free-tier limits mirror the AI Studio project limits used by this relay.
// Effective single-request capacity is min(contextWindowTokens, TPM).
const GOOGLE_MODELS: readonly ManagedModel[] = [
  {
    id: "google/gemma-4-26b-a4b-it",
    upstreamModel: "gemma-4-26b-a4b-it",
    contextWindowTokens: 262_144,
    quota: quota(30, 16_000, 14_400),
  },
  {
    id: "google/gemma-4-31b-it",
    upstreamModel: "gemma-4-31b-it",
    contextWindowTokens: 262_144,
    quota: quota(30, 16_000, 14_400),
  },
  {
    id: "google/gemini-3.5-flash-lite",
    upstreamModel: "gemini-3.5-flash-lite",
    contextWindowTokens: 1_048_576,
    quota: quota(15, 250_000, 500),
  },
  {
    id: "google/gemini-3.1-flash-lite",
    upstreamModel: "gemini-3.1-flash-lite",
    contextWindowTokens: 1_048_576,
    quota: quota(15, 250_000, 500),
  },
  {
    id: "google/gemini-3.8-flash",
    upstreamModel: "gemini-3.8-flash",
    contextWindowTokens: 1_048_576,
    quota: quota(5, 250_000, 20),
  },
  {
    id: "google/gemini-3.7-flash",
    upstreamModel: "gemini-3.7-flash",
    contextWindowTokens: 1_048_576,
    quota: quota(5, 250_000, 20),
  },
  {
    id: "google/gemini-3.6-flash",
    upstreamModel: "gemini-3.6-flash",
    contextWindowTokens: 1_048_576,
    quota: quota(5, 250_000, 20),
  },
  {
    id: "google/gemini-2.5-flash-lite",
    upstreamModel: "gemini-2.5-flash-lite",
    contextWindowTokens: 1_048_576,
    quota: quota(10, 250_000, 20),
  },
  {
    id: "google/gemini-2.5-flash",
    upstreamModel: "gemini-2.5-flash",
    contextWindowTokens: 1_048_576,
    quota: quota(5, 250_000, 20),
  },
];

export function createGoogleProvider(): OpenAICompatibleProvider {
  return new OpenAICompatibleProvider({
    id: "google",
    priority: 10,
    credentialEnv: "GEMINI_API_KEY",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
    defaultRetryMs: DEFAULT_RETRY_MS,
    providerFailureCooldownMs: PROVIDER_FAILURE_COOLDOWN_MS,
    models: GOOGLE_MODELS,
  });
}
