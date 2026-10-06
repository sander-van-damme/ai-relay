import type { ChatCompletionRequest } from "../../types.ts";

type JsonObject = Record<string, unknown>;

function object(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

const OUTPUT_ONLY_REASONING_FIELDS = [
  "reasoning",
  "reasoning_content",
  "reasoningContent",
] as const;

/**
 * GPT-OSS providers may return provider-specific reasoning fields on assistant
 * messages. Those fields are output metadata, not OpenAI Chat Completions input,
 * and replaying them can make Harmony reject a later multi-turn request.
 *
 * Keep ordinary message content untouched, including literal Harmony-looking
 * text supplied by a user. Only remove known output-only fields from assistant
 * messages.
 */
export function normalizeGptOssRequest(body: ChatCompletionRequest): ChatCompletionRequest {
  if (!Array.isArray(body.messages)) return body;

  let changed = false;
  const messages = body.messages.map((rawMessage) => {
    const message = object(rawMessage);
    if (!message || message.role !== "assistant") return rawMessage;

    let normalized: JsonObject | undefined;
    for (const field of OUTPUT_ONLY_REASONING_FIELDS) {
      if (!(field in message)) continue;
      normalized ??= { ...message };
      delete normalized[field];
      changed = true;
    }
    return normalized ?? rawMessage;
  });

  return changed ? { ...body, messages } : body;
}

/**
 * Known GPT-OSS serving stacks can surface Harmony parser crashes as either
 * 4xx or 5xx responses. Treat the failure as model/request-format specific so
 * auto routing can move to another model instead of suppressing the provider.
 */
export function isHarmonyParserFailure(bodyText: string): boolean {
  const normalized = bodyText.toLowerCase();
  return normalized.includes("unexpected tokens remaining in message header")
    || (
      normalized.includes("harmonyerror")
      && (
        normalized.includes("unexpected token")
        || normalized.includes("unknown role")
      )
    );
}
