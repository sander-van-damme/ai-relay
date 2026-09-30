import type { QuotaPolicy } from "../quota.ts";
import type { ServerConfig } from "../types.ts";
import { OpenAICompatibleProvider, type ManagedModel } from "./openai-compatible.ts";

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

export function createNvidiaProvider(server: ServerConfig): OpenAICompatibleProvider {
  return new OpenAICompatibleProvider({
    id: "nvidia",
    priority: 20,
    credentialEnv: "NVIDIA_API_KEY",
    baseUrl: "https://integrate.api.nvidia.com/v1",
    defaultRetryMs: server.retrySeconds * 1000,
    providerFailureCooldownMs: 15_000,
    models: NVIDIA_MODELS,
  });
}
