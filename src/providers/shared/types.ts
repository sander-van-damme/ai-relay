import type { ChatCompletionRequest } from "../../types.ts";

export type ProviderOfferKind = "standard" | "overflow";

export interface OfferRequest {
  offerKind: ProviderOfferKind;
  body: ChatCompletionRequest;
  requestedModel: string;
  maxOptimizationWaitMs: number;
  excludedModelIds: ReadonlySet<string>;
}

export interface ProviderOffer {
  kind: ProviderOfferKind;
  providerId: string;
  providerPriority: number;
  modelId: string;
  inputTokens: number;
  inputCapacityTokens: number;
  availableAt: number;
}

export type ProviderNoOfferReason =
  | "provider_not_configured"
  | "no_eligible_model"
  | "request_exceeds_capacity"
  | "token_count_failed"
  | "offer_evaluation_failed";

export type ProviderOfferResult =
  | {
      status: "offer";
      offer: ProviderOffer;
    }
  | {
      status: "no_offer";
      providerId: string;
      reason: ProviderNoOfferReason;
      detail?: string;
    };

export interface ProviderModelInfo {
  id: string;
  providerId: string;
  inputCapacityTokens: number;
}

export interface ProviderStatus {
  id: string;
  configured: boolean;
  blockedUntil: number | null;
  models: Array<{
    id: string;
    active: number;
    blockedUntil: number | null;
    overflowBlockedUntil: number | null;
  }>;
}

export type ProviderFailureScope = "provider" | "model";

export type ProviderExecutionResult =
  | {
      status: "success";
      response: Response;
      release: () => void;
    }
  | {
      status: "retryable";
      scope: ProviderFailureScope;
      reason: string;
      retryAt: number;
    }
  | {
      status: "rejected";
      scope: ProviderFailureScope;
      httpStatus: number;
      bodyText: string;
    };

export interface Provider {
  readonly id: string;
  readonly priority: number;
  isConfigured(): boolean;
  listModels(): readonly ProviderModelInfo[];
  getBestOffer(request: OfferRequest, now?: number): Promise<ProviderOfferResult>;
  execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    stream: boolean,
    signal: AbortSignal,
  ): Promise<ProviderExecutionResult>;
  status(now?: number): ProviderStatus;
}
