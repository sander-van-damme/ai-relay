import { AutoTokenizer } from "@huggingface/transformers";
import { countChatCompletionTokens as countGptOss20bChatCompletionTokens } from "gpt-tokenizer/model/gpt-oss-20b";
import type { ChatCompletionRequest } from "../../types.ts";

export type NvidiaTokenizerSpec =
  | {
      kind: "gpt-oss-20b";
    }
  | {
      kind: "huggingface";
      repository: string;
      revision?: string;
    };

interface ChatTemplateTokenizer {
  apply_chat_template(
    conversation: unknown[],
    options?: Record<string, unknown>,
  ): unknown;
}

const huggingFaceTokenizerCache = new Map<string, Promise<ChatTemplateTokenizer>>();

function tokenizerCacheKey(spec: Extract<NvidiaTokenizerSpec, { kind: "huggingface" }>): string {
  return `${spec.repository}@${spec.revision ?? "main"}`;
}

function loadHuggingFaceTokenizer(
  spec: Extract<NvidiaTokenizerSpec, { kind: "huggingface" }>,
): Promise<ChatTemplateTokenizer> {
  const key = tokenizerCacheKey(spec);
  const cached = huggingFaceTokenizerCache.get(key);
  if (cached) return cached;

  const pending = AutoTokenizer.from_pretrained(spec.repository, {
    revision: spec.revision ?? "main",
  })
    .then((tokenizer) => tokenizer as unknown as ChatTemplateTokenizer)
    .catch((error) => {
      huggingFaceTokenizerCache.delete(key);
      throw error;
    });

  huggingFaceTokenizerCache.set(key, pending);
  return pending;
}

function tokenLength(value: unknown): number | null {
  if (Array.isArray(value)) return value.length;
  if (ArrayBuffer.isView(value)) {
    const length = Reflect.get(value, "length");
    return typeof length === "number" ? length : null;
  }

  if (typeof value !== "object" || value === null) return null;
  const inputIds = Reflect.get(value, "input_ids");
  if (inputIds !== undefined) return tokenLength(inputIds);
  return null;
}

function messages(body: ChatCompletionRequest): unknown[] {
  if (!Array.isArray(body.messages)) {
    throw new Error("NVIDIA token counting requires an OpenAI-style messages array.");
  }
  return body.messages;
}

function tools(body: ChatCompletionRequest): unknown[] | undefined {
  return Array.isArray(body.tools) ? body.tools : undefined;
}

function countGptOss20b(body: ChatCompletionRequest): number {
  const count = countGptOss20bChatCompletionTokens({
    ...body,
    model: "gpt-oss-20b",
  } as never);

  if (!Number.isSafeInteger(count) || count < 0) {
    throw new Error(`gpt-tokenizer returned an invalid GPT-OSS token count: ${count}`);
  }
  return count;
}

async function countHuggingFaceChat(
  body: ChatCompletionRequest,
  spec: Extract<NvidiaTokenizerSpec, { kind: "huggingface" }>,
): Promise<number> {
  const tokenizer = await loadHuggingFaceTokenizer(spec);
  const requestTools = tools(body);

  const rendered = tokenizer.apply_chat_template(messages(body), {
    tokenize: true,
    return_tensor: false,
    return_dict: false,
    add_generation_prompt: true,
    ...(requestTools ? { tools: requestTools } : {}),
  });

  const count = tokenLength(rendered);
  if (!Number.isSafeInteger(count) || count === null || count < 0) {
    throw new Error(
      `Hugging Face tokenizer ${spec.repository} returned an unsupported tokenized result.`,
    );
  }
  return count;
}

export async function countNvidiaInputTokens(
  body: ChatCompletionRequest,
  tokenizer: NvidiaTokenizerSpec,
): Promise<number> {
  switch (tokenizer.kind) {
    case "gpt-oss-20b":
      return countGptOss20b(body);
    case "huggingface":
      return countHuggingFaceChat(body, tokenizer);
  }
}
