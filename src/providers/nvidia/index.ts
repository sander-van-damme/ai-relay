import type { QuotaPolicy } from "../shared/quota.ts";
import { OpenAICompatibleProvider, type ManagedModel } from "../shared/openai-compatible.ts";
import { countNvidiaInputTokens } from "./token-count.ts";

const DEFAULT_RETRY_MS = 5_000;
const PROVIDER_FAILURE_COOLDOWN_MS = 15_000;

const UNMETERED_LOCAL_POLICY: QuotaPolicy = {
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
    id: "nvidia/openai/gpt-oss-120b",
    upstreamModel: "openai/gpt-oss-120b",
    contextWindowTokens: 131_072,
    quota: UNMETERED_LOCAL_POLICY,
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
