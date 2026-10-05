import type { ProviderModelInfo } from "./providers/index.ts";

const RELAY_MODEL_CREATED = Math.floor(Date.now() / 1_000);

export interface ModelCatalogSource {
  listConfiguredModels(): readonly ProviderModelInfo[];
  autoInputCapacityTokens(now?: number): number | null;
}

export interface RelayApiModel {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
  shutdown_date: null;
  /**
   * Relay extension: maximum input tokens that one request can send through
   * this route after applying model-context and hard single-request quota caps.
   */
  input_capacity_tokens: number;
}

function modelObject(
  id: string,
  ownedBy: string,
  inputCapacityTokens: number,
  created = RELAY_MODEL_CREATED,
): RelayApiModel {
  return {
    id,
    object: "model",
    created,
    owned_by: ownedBy,
    shutdown_date: null,
    input_capacity_tokens: inputCapacityTokens,
  };
}

export function modelsPayload(
  catalog: ModelCatalogSource,
  now = Date.now(),
  created = RELAY_MODEL_CREATED,
): { object: "list"; data: RelayApiModel[] } {
  const concrete = catalog.listConfiguredModels().map((model) =>
    modelObject(model.id, model.providerId, model.inputCapacityTokens, created)
  );
  const autoCapacity = catalog.autoInputCapacityTokens(now);
  const auto = autoCapacity === null
    ? []
    : [modelObject("auto", "ai-relay", autoCapacity, created)];

  return { object: "list", data: [...auto, ...concrete] };
}

export function modelPayload(
  catalog: ModelCatalogSource,
  modelId: string,
  now = Date.now(),
  created = RELAY_MODEL_CREATED,
): RelayApiModel | null {
  if (modelId === "auto") {
    const capacity = catalog.autoInputCapacityTokens(now);
    return capacity === null ? null : modelObject("auto", "ai-relay", capacity, created);
  }

  const model = catalog.listConfiguredModels().find((candidate) => candidate.id === modelId);
  return model
    ? modelObject(model.id, model.providerId, model.inputCapacityTokens, created)
    : null;
}
