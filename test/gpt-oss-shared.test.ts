import assert from "node:assert/strict";
import test from "node:test";
import {
  isHarmonyParserFailure,
  normalizeGptOssRequest,
} from "../src/providers/shared/gpt-oss.ts";

test("GPT-OSS replay normalization removes only assistant output-only reasoning fields", () => {
  const body = {
    messages: [
      { role: "user", content: "Explain <|start|> literally.", reasoning: "keep-user-field" },
      {
        role: "assistant",
        content: "final answer",
        reasoning: "private reasoning",
        reasoning_content: "provider reasoning",
        reasoningContent: { text: "alternate provider reasoning" },
        tool_calls: [{ id: "call-1", type: "function", function: { name: "x", arguments: "{}" } }],
      },
      { role: "tool", tool_call_id: "call-1", content: "ok" },
    ],
  };

  const normalized = normalizeGptOssRequest(body);
  assert.deepEqual(normalized, {
    messages: [
      { role: "user", content: "Explain <|start|> literally.", reasoning: "keep-user-field" },
      {
        role: "assistant",
        content: "final answer",
        tool_calls: [{ id: "call-1", type: "function", function: { name: "x", arguments: "{}" } }],
      },
      { role: "tool", tool_call_id: "call-1", content: "ok" },
    ],
  });

  assert.equal(body.messages[1]!.reasoning, "private reasoning");
});

test("GPT-OSS replay normalization preserves object identity when no cleanup is needed", () => {
  const body = { messages: [{ role: "assistant", content: "clean" }] };
  assert.equal(normalizeGptOssRequest(body), body);
});

test("Harmony parser failures are recognized without matching ordinary user text", () => {
  assert.equal(
    isHarmonyParserFailure(JSON.stringify({
      error: {
        message: 'unexpected tokens remaining in message header: Some("to=functions.bash")',
      },
    })),
    true,
  );
  assert.equal(
    isHarmonyParserFailure("openai_harmony.HarmonyError: Unknown role: <|start|><|start|>assistant"),
    true,
  );
  assert.equal(
    isHarmonyParserFailure("User asked what <|start|>assistant means"),
    false,
  );
});
