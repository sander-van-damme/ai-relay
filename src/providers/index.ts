import { createGoogleProvider } from "./google/index.ts";
import { createNvidiaProvider } from "./nvidia/index.ts";
import type { Provider } from "./shared/types.ts";

export function createProviders(): Provider[] {
  return [createGoogleProvider(), createNvidiaProvider()];
}

export type { OfferRequest, Provider, ProviderExecutionResult, ProviderFailureScope, ProviderModelInfo, ProviderOffer, ProviderStatus } from "./shared/types.ts";
