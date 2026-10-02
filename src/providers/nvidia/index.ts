import type { QuotaPolicy } from "../shared/quota.ts";
import { OpenAICompatibleProvider, type ManagedModel } from "../shared/openai-compatible.ts";
import { countNvidiaInputTokens } from "./token-count.ts";

const DEFAULT_RETRY_MS = 5_000;
const PROVIDER_FAILURE_COOLDOWN_MS = 15_000;

// NVIDIA's hosted trial does not publish a stable per-account RPM/TPM/RPD
// contract that is safe to hard-code here. Treat those limits as upstream-owned
// and learn temporary availability from actual 429 responses instead.
const HOSTED_TRIAL_POLICY: QuotaPolicy = {
  maxConcurrent: null,
  dailyWindow: { type: "rolling" },
  limits: {
    requestsPerMinute: null,
    inputTokensPerMinute: null,
    requestsPerDay: null,
    minimumSpacingMs: 0,
  },
};

const NVIDIA_MODELS: readonly ManagedModel[] = [
  {
    id: "nvidia/openai/gpt-oss-20b",
    upstreamModel: "openai/gpt-oss-20b",
    contextWindowTokens: 131_072,
    quota: HOSTED_TRIAL_POLICY,
  },
];

export function createNvidiaProvider(): OpenAICompatibleProvider {
  return new OpenAICompatibleProvider({
    id: "nvidia",
    priority: 20,
    credentialEnv: "NVIDIA_API_KEY",
    baseUrl: "https://integrate.api.nvidia.com/v1",
    defaultRetryMs: DEFAULT_RETRY_MS,
    providerFailureCooldownMs: PROVIDER_FAILURE_COOLDOWN_MS,
    countInputTokens: countNvidiaInputTokens,
    models: NVIDIA_MODELS,
  });
}
