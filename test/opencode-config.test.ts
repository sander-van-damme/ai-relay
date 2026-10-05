import assert from "node:assert/strict";
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { mkdtemp } from "node:fs/promises";
import ts from "typescript";
import {
  applyOptimizations,
  optimizeConfig,
  restoreConfig,
  type ConfigPaths,
} from "../src/opencode-config.ts";

function toValue(node: ts.Expression): unknown {
  if (ts.isObjectLiteralExpression(node)) {
    return Object.fromEntries(node.properties.map((property) => {
      assert(ts.isPropertyAssignment(property));
      assert(ts.isStringLiteral(property.name) || ts.isIdentifier(property.name));
      return [property.name.text, toValue(property.initializer)];
    }));
  }
  if (ts.isArrayLiteralExpression(node)) return node.elements.map(toValue);
  if (ts.isStringLiteral(node)) return node.text;
  if (ts.isNumericLiteral(node)) return Number(node.text);
  if (node.kind === ts.SyntaxKind.TrueKeyword) return true;
  if (node.kind === ts.SyntaxKind.FalseKeyword) return false;
  if (node.kind === ts.SyntaxKind.NullKeyword) return null;
  throw new Error(`Unsupported test JSONC node: ${ts.SyntaxKind[node.kind]}`);
}

function parseJsonc(text: string): Record<string, unknown> {
  const source = ts.parseJsonText("test.jsonc", text);
  const statement = source.statements[0];
  assert(statement && ts.isExpressionStatement(statement));
  const result = toValue(statement.expression);
  assert(result && typeof result === "object" && !Array.isArray(result));
  return result as Record<string, unknown>;
}

async function fixture(contents?: string): Promise<{ directory: string; paths: ConfigPaths }> {
  const directory = await mkdtemp(join(tmpdir(), "ai-relay-opencode-"));
  const configPath = join(directory, "nested", "opencode.jsonc");
  const paths = { configPath, backupPath: `${configPath}.backup` };
  if (contents !== undefined) {
    await mkdir(join(directory, "nested"), { recursive: true });
    await writeFile(configPath, contents);
  }
  return { directory, paths };
}

function assertOptimized(value: Record<string, unknown>): void {
  assert.equal(value.warming, false);
  assert.deepEqual(value.tool_output, { max_lines: 1000, max_bytes: 32768 });
  assert.deepEqual(value.agents, { title: { disabled: true } });
  assert.equal(value.compaction, undefined);
  assert.equal(value.permission, undefined);
}

test("creates a minimal config and missing parent directory", async () => {
  const { paths } = await fixture();
  const result = await optimizeConfig(paths);
  assert.equal(result.configCreated, true);
  assertOptimized(parseJsonc(await readFile(paths.configPath, "utf8")));
  await assert.rejects(access(paths.backupPath));
});

test("optimizes an empty config file and backs up its original contents", async () => {
  const { paths } = await fixture("");
  await optimizeConfig(paths);
  assertOptimized(parseJsonc(await readFile(paths.configPath, "utf8")));
  assert.equal(await readFile(paths.backupPath, "utf8"), "");
});

test("preserves unrelated configuration", () => {
  const original = `{
  "provider": { "custom": { "api": "https://example.test" } },
  "model": "custom/model",
  "plugins": ["example"]
}\n`;
  const value = parseJsonc(applyOptimizations(original).text);
  assert.deepEqual(value.provider, { custom: { api: "https://example.test" } });
  assert.equal(value.model, "custom/model");
  assert.deepEqual(value.plugins, ["example"]);
});

test("leaves user-defined compaction settings alone", () => {
  const value = parseJsonc(applyOptimizations(
    '{ "compaction": { "auto": false, "prune": true, "keep": { "messages": 4, "tokens": 12000 }, "buffer": 8000 } }',
  ).text);
  assert.deepEqual(value.compaction, {
    auto: false,
    prune: true,
    keep: { messages: 4, tokens: 12000 },
    buffer: 8000,
  });
});

test("does not add relay auto-model context or output limits", () => {
  const original = `{
  "provider": {
    "relay": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "relay",
      "models": {
        "auto": {
          "name": "auto"
        }
      }
    }
  }
}\n`;
  const value = parseJsonc(applyOptimizations(original).text);
  const provider = value.provider as Record<string, unknown>;
  const relay = provider.relay as Record<string, unknown>;
  const models = relay.models as Record<string, unknown>;
  const auto = models.auto as Record<string, unknown>;
  assert.equal(relay.npm, "@ai-sdk/openai-compatible");
  assert.equal(auto.name, "auto");
  assert.equal(auto.limit, undefined);
});

test("removes legacy relay-managed compaction, skill, and model-limit settings", () => {
  const original = `{
  "warming": false,
  "compaction": {
    "auto": true,
    "prune": true,
    "keep": { "messages": 4, "tokens": 3000 },
    "buffer": 4000
  },
  "tool_output": { "max_lines": 1000, "max_bytes": 32768 },
  "agents": { "title": { "disabled": true } },
  "permission": { "bash": "allow", "skill": "deny" },
  "provider": {
    "relay": {
      "models": {
        "auto": {
          "name": "auto",
          "limit": { "context": 18000, "input": 18000, "output": 4000 }
        }
      }
    }
  }
}\n`;
  const result = applyOptimizations(original);
  const value = parseJsonc(result.text);

  assert.deepEqual(value.compaction, {
    prune: true,
    keep: { messages: 4 },
  });
  assert.deepEqual(value.permission, { bash: "allow" });

  const provider = value.provider as Record<string, unknown>;
  const relay = provider.relay as Record<string, unknown>;
  const models = relay.models as Record<string, unknown>;
  const auto = models.auto as Record<string, unknown>;
  assert.deepEqual(auto, { name: "auto" });

  const removed = result.changes
    .filter((change) => change.after === undefined)
    .map((change) => change.path);
  assert.deepEqual(removed, [
    "compaction.auto",
    "compaction.keep.tokens",
    "compaction.buffer",
    "permission.skill",
    "provider.relay.models.auto.limit.context",
    "provider.relay.models.auto.limit.input",
    "provider.relay.models.auto.limit.output",
  ]);
});

test("preserves user-modified values at formerly managed paths", () => {
  const original = `{
  "compaction": { "auto": false, "keep": { "tokens": 9000 }, "buffer": 7000 },
  "permission": { "skill": "allow" },
  "provider": {
    "relay": {
      "models": {
        "auto": {
          "limit": { "context": 64000, "input": 64000, "output": 8000 }
        }
      }
    }
  }
}\n`;
  const value = parseJsonc(applyOptimizations(original).text);
  assert.deepEqual(value.compaction, { auto: false, keep: { tokens: 9000 }, buffer: 7000 });
  assert.deepEqual(value.permission, { skill: "allow" });

  const provider = value.provider as Record<string, unknown>;
  const relay = provider.relay as Record<string, unknown>;
  const models = relay.models as Record<string, unknown>;
  const auto = models.auto as Record<string, unknown>;
  assert.deepEqual(auto.limit, { context: 64000, input: 64000, output: 8000 });
});

test("preserves existing agents", () => {
  const value = parseJsonc(applyOptimizations('{ "agents": { "build": { "model": "custom/model" } } }').text);
  assert.deepEqual(value.agents, {
    build: { model: "custom/model" },
    title: { disabled: true },
  });
});

test("preserves skill permissions instead of managing them", () => {
  const value = parseJsonc(applyOptimizations('{ "permission": { "bash": "allow", "skill": "allow" } }').text);
  assert.deepEqual(value.permission, {
    bash: "allow",
    skill: "allow",
  });
});

test("retains JSONC comments and accepts trailing commas", () => {
  const original = `{
  // Provider settings must remain documented.
  "provider": {
    "custom": true,
  },
}\n`;
  const result = applyOptimizations(original).text;
  assert.match(result, /\/\/ Provider settings must remain documented\./);
  assert.equal((parseJsonc(result).provider as Record<string, unknown>).custom, true);
});

test("optimization is idempotent", () => {
  const first = applyOptimizations("{}\n");
  const second = applyOptimizations(first.text);
  assert.equal(second.text, first.text);
  assert.deepEqual(second.changes, []);
});

test("dry run calculates changes without writing config, directory, or backup", async () => {
  const { directory, paths } = await fixture();
  const result = await optimizeConfig(paths, { dryRun: true });
  assert.equal(result.changes.length, 4);
  await assert.rejects(access(join(directory, "nested")));
  await assert.rejects(access(paths.configPath));
  await assert.rejects(access(paths.backupPath));
});

test("creates one durable backup and does not overwrite it", async () => {
  const original = '{ "theme": "dark" }\n';
  const { paths } = await fixture(original);
  const result = await optimizeConfig(paths);
  assert.equal(result.backupCreated, true);
  assert.equal(await readFile(paths.backupPath, "utf8"), original);
  await writeFile(paths.configPath, '{ "warming": true }\n');
  await optimizeConfig(paths);
  assert.equal(await readFile(paths.backupPath, "utf8"), original);
});

test("restore replaces the config with the saved backup", async () => {
  const original = '{ "theme": "dark" }\n';
  const { paths } = await fixture(original);
  await optimizeConfig(paths);
  await restoreConfig(paths);
  assert.equal(await readFile(paths.configPath, "utf8"), original);
  assert.equal(await readFile(paths.backupPath, "utf8"), original);
});

test("restore can recover an originally empty config", async () => {
  const { paths } = await fixture("");
  await optimizeConfig(paths);
  await restoreConfig(paths);
  assert.equal(await readFile(paths.configPath, "utf8"), "");
});

test("restore fails clearly when no backup exists", async () => {
  const { paths } = await fixture();
  await assert.rejects(restoreConfig(paths), /No OpenCode configuration backup found/);
});
