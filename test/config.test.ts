import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { parseConfig } from "../src/config.ts";

test("checked-in model config parses", async () => {
  const raw = JSON.parse(await readFile(new URL("../config/models.json", import.meta.url), "utf8")) as unknown;
  const config = parseConfig(raw);
  assert.equal(config.models.length, 2);
  assert.equal(config.models[0]?.provider, "google");
  assert.equal(config.models[1]?.provider, "nvidia");
});

test("duplicate relay model IDs are rejected", () => {
  assert.throws(() => parseConfig({
    providers: {
      google: { baseUrl: "https://example.com", apiKeyEnv: ["KEY"] },
      nvidia: { baseUrl: "https://example.com", apiKeyEnv: ["KEY2"] },
    },
    models: [
      { id: "same", provider: "google", upstreamModel: "a" },
      { id: "same", provider: "nvidia", upstreamModel: "b" },
    ],
  }), /duplicate model id/);
});
