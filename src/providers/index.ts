import type { ServerConfig } from "../types.ts";
import { createGoogleProvider } from "./google.ts";
import { createNvidiaProvider } from "./nvidia.ts";
import type { Provider } from "./types.ts";

export function createProviders(server: ServerConfig): Provider[] {
  return [createGoogleProvider(server), createNvidiaProvider(server)];
}

export type { OfferRequest, Provider, ProviderExecutionResult, ProviderModelInfo, ProviderOffer, ProviderStatus } from "./types.ts";
