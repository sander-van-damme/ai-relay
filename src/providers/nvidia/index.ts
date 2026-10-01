import type { QuotaPolicy } from "../shared/quota.ts";
import { OpenAICompatibleProvider, type ManagedModel } from "../shared/openai-compatible.ts";
import type { OfferRequest, ProviderOfferResult } from "../shared/types.ts";
import { countNvidiaInputTokens } from "./token-count.ts";

const DEFAULT_RETRY_MS = 5_000;
const PROVIDER_FAILURE_COOLDOWN_MS = 15_000;

const NVIDIA_DISABLED_DETAIL =
  "NVIDIA provider is intentionally disabled until its provider implementation is completed.";

// TODO(nvidia-provider): This implementation is intentionally kept as scaffolding,
// but it does not yet satisfy the provider contract in ../README.md and must not
// participate in routing.
//
// Before enabling NVIDIA offers:
// - evaluate NVIDIA's official SDK/native APIs versus the OpenAI-compatible API;
// - expand and verify the supported model catalog and model-specific capabilities;
// - implement verified RPM, TPM, RPD, concurrency, spacing, and account/tier limits;
// - verify context/input capacities and authoritative model-specific token counting;
// - implement NVIDIA-specific cooldown, failure, and overflow semantics where justified;
// - add the provider-contract tests required by ../README.md.
//
// The current model, tokenizer, transport, and placeholder quota configuration below
// are preserved only as a starting point for that later implementation.
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

class NvidiaProvider extends OpenAICompatibleProvider {
  override async getBestOffer(
    _request: OfferRequest,
    _now = Date.now(),
  ): Promise<ProviderOfferResult> {
    return {
      status: "no_offer",
      providerId: this.id,
      reason: "no_eligible_model",
      detail: NVIDIA_DISABLED_DETAIL,
    };
  }
}

export function createNvidiaProvider(): OpenAICompatibleProvider {
  return new NvidiaProvider({
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
