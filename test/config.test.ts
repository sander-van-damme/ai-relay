import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { parseConfig } from "../src/config.ts";

test("checked-in model config parses provider and model quota policies", async () => {
  const raw = JSON.parse(await readFile(new URL("../config/models.json", import.meta.url), "utf8")) as unknown;
  const config = parseConfig(raw);
  assert.equal(config.models.length, 2);
  assert.equal(config.models[0]?.provider, "google");
  assert.equal(config.models[1]?.provider, "nvidia");
  assert.equal(config.providers.google.maxConcurrent, null);
  assert.equal(config.providers.google.limits.requestsPerMinute, null);
});

test("provider API-key names are not part of models.json", async () => {
  const text = await readFile(new URL("../config/models.json", import.meta.url), "utf8");
  assert.doesNotMatch(text, /apiKeyEnv|GEMINI_API_KEY|NVIDIA_API_KEY/);
});

test("duplicate relay model IDs are rejected", () => {
  assert.throws(() => parseConfig({
    providers: {
      google: { baseUrl: "https://example.com" },
      nvidia: { baseUrl: "https://example.com" },
    },
    models: [
      { id: "same", provider: "google", upstreamModel: "a" },
      { id: "same", provider: "nvidia", upstreamModel: "b" },
    ],
  }), /duplicate model id/);
});

test("provider-level quota fields parse independently from model quotas", () => {
  const config = parseConfig({
    providers: {
      google: {
        baseUrl: "https://example.com",
        maxConcurrent: null,
        limits: { requestsPerMinute: 10, inputTokensPerMinute: 1000, requestsPerDay: 100 },
      },
      nvidia: { baseUrl: "https://example.com" },
    },
    models: [
      {
        id: "google/a",
        provider: "google",
        upstreamModel: "a",
        maxConcurrent: 2,
        limits: { requestsPerMinute: 3, inputTokensPerMinute: 500 },
      },
    ],
  });

  assert.equal(config.providers.google.limits.requestsPerMinute, 10);
  assert.equal(config.providers.google.limits.inputTokensPerMinute, 1000);
  assert.equal(config.models[0]?.limits.requestsPerMinute, 3);
  assert.equal(config.models[0]?.maxConcurrent, 2);
});
