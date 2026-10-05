import assert from "node:assert/strict";
import test from "node:test";
import {
  modelPayload,
  modelsPayload,
  type ModelCatalogSource,
} from "../src/models-api.ts";

function catalog(autoCapacity: number | null = 1_048_576): ModelCatalogSource {
  return {
    listConfiguredModels: () => [
      { id: "google/model", providerId: "google", inputCapacityTokens: 250_000 },
      { id: "nvidia/model", providerId: "nvidia", inputCapacityTokens: 1_048_576 },
    ],
    autoInputCapacityTokens: () => autoCapacity,
  };
}

test("models payload keeps the OpenAI model shape and adds relay input capacity", () => {
  const payload = modelsPayload(catalog(), 1_000_000, 123);

  assert.equal(payload.object, "list");
  assert.deepEqual(payload.data, [
    {
      id: "auto",
      object: "model",
      created: 123,
      owned_by: "ai-relay",
      shutdown_date: null,
      input_capacity_tokens: 1_048_576,
    },
    {
      id: "google/model",
      object: "model",
      created: 123,
      owned_by: "google",
      shutdown_date: null,
      input_capacity_tokens: 250_000,
    },
    {
      id: "nvidia/model",
      object: "model",
      created: 123,
      owned_by: "nvidia",
      shutdown_date: null,
      input_capacity_tokens: 1_048_576,
    },
  ]);
});

test("auto disappears when no currently usable route can back it", () => {
  const payload = modelsPayload(catalog(null), 1_000_000, 123);

  assert.deepEqual(payload.data.map((model) => model.id), [
    "google/model",
    "nvidia/model",
  ]);
  assert.equal(modelPayload(catalog(null), "auto", 1_000_000, 123), null);
});

test("retrieve model returns concrete configured models and rejects unknown IDs", () => {
  assert.deepEqual(modelPayload(catalog(), "google/model", 1_000_000, 123), {
    id: "google/model",
    object: "model",
    created: 123,
    owned_by: "google",
    shutdown_date: null,
    input_capacity_tokens: 250_000,
  });
  assert.equal(modelPayload(catalog(), "missing/model", 1_000_000, 123), null);
});
