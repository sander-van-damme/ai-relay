import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parseConfig } from "../src/config.ts";

test("checked-in config contains only relay/server settings", async () => {
  const text = await readFile(new URL("../config/relay.json", import.meta.url), "utf8");
  const config = parseConfig(JSON.parse(text) as unknown);
  assert.equal(config.server.port, 8787);
  assert.doesNotMatch(text, /gemini|nvidia|requestsPerDay|inputTokensPerMinute|retrySeconds/);
});
