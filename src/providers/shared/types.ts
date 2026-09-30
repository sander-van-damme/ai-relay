import type { ChatCompletionRequest } from "../../types.ts";

export type ProviderOfferKind = "standard" | "overflow";

export interface OfferRequest {
  offerKind: ProviderOfferKind;
  requestedModel: string;
  estimatedInputTokens: number;
  maxOptimizationWaitMs: number;
  excludedModelIds: ReadonlySet<string>;
}

export interface ProviderOffer {
  kind: ProviderOfferKind;
  providerId: string;
  providerPriority: number;
  modelId: string;
  inputCapacityTokens: number;
  availableAt: number;
}

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
  getBestOffer(request: OfferRequest, now?: number): ProviderOffer | null;
  execute(
    offer: ProviderOffer,
    body: ChatCompletionRequest,
    stream: boolean,
    estimatedInputTokens: number,
    signal: AbortSignal,
  ): Promise<ProviderExecutionResult>;
  status(now?: number): ProviderStatus;
}
