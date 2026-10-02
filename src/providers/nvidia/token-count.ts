import { countChatCompletionTokens as countGptOss20bChatCompletionTokens } from "gpt-tokenizer/model/gpt-oss-20b";
import type { ChatCompletionRequest } from "../../types.ts";

const TOKENIZER_ASSET_TIMEOUT_MS = 15_000;

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
  chat_template?: unknown;
  get_chat_template?: (options?: Record<string, unknown>) => unknown;
  apply_chat_template(
    conversation: unknown[],
    options?: Record<string, unknown>,
  ): unknown;
}

const huggingFaceTokenizerCache = new Map<string, Promise<ChatTemplateTokenizer>>();

function tokenizerCacheKey(spec: Extract<NvidiaTokenizerSpec, { kind: "huggingface" }>): string {
  return `${spec.repository}@${spec.revision ?? "main"}`;
}

function hasChatTemplate(tokenizer: ChatTemplateTokenizer): boolean {
  try {
    if (typeof tokenizer.get_chat_template === "function") {
      const template = tokenizer.get_chat_template();
      if (typeof template === "string" && template.trim()) return true;
      if (Array.isArray(template) && template.length > 0) return true;
    }
  } catch {
    // A tokenizer without an installed template may throw here.
  }

  if (typeof tokenizer.chat_template === "string") return Boolean(tokenizer.chat_template.trim());
  return Array.isArray(tokenizer.chat_template) && tokenizer.chat_template.length > 0;
}

async function fetchTokenizerAsset(
  repository: string,
  revision: string,
  filename: string,
): Promise<string | null> {
  const response = await fetch(
    `https://huggingface.co/${repository}/resolve/${encodeURIComponent(revision)}/${filename}`,
    {
      headers: { "user-agent": "ai-relay/1.0" },
      signal: AbortSignal.timeout(TOKENIZER_ASSET_TIMEOUT_MS),
    },
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(
      `Hugging Face tokenizer asset ${repository}/${filename} returned HTTP ${response.status}.`,
    );
  }
  return response.text();
}

async function ensureChatTemplate(
  tokenizer: ChatTemplateTokenizer,
  spec: Extract<NvidiaTokenizerSpec, { kind: "huggingface" }>,
): Promise<void> {
  if (hasChatTemplate(tokenizer)) return;

  const revision = spec.revision ?? "main";
  const jinja = await fetchTokenizerAsset(spec.repository, revision, "chat_template.jinja");
  if (jinja?.trim()) {
    Reflect.set(tokenizer, "chat_template", jinja);
    return;
  }

  const jsonText = await fetchTokenizerAsset(spec.repository, revision, "chat_template.json");
  if (jsonText) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(jsonText) as unknown;
    } catch {
      throw new Error(
        `Hugging Face tokenizer ${spec.repository} returned invalid chat_template.json.`,
      );
    }

    const value = typeof parsed === "object" && parsed !== null
      ? Reflect.get(parsed, "chat_template")
      : undefined;
    const template = value ?? parsed;
    if (
      (typeof template === "string" && template.trim())
      || (Array.isArray(template) && template.length > 0)
    ) {
      Reflect.set(tokenizer, "chat_template", template);
      return;
    }
  }

  throw new Error(
    `Hugging Face tokenizer ${spec.repository} does not expose a usable chat template.`,
  );
}

function loadHuggingFaceTokenizer(
  spec: Extract<NvidiaTokenizerSpec, { kind: "huggingface" }>,
): Promise<ChatTemplateTokenizer> {
  const key = tokenizerCacheKey(spec);
  const cached = huggingFaceTokenizerCache.get(key);
  if (cached) return cached;

  const pending = (async () => {
    // Keep the current GPT-OSS path lightweight: Transformers.js and remote
    // tokenizer assets are loaded only after an HF-backed model is enabled.
    const { AutoTokenizer } = await import("@huggingface/transformers");
    const tokenizer = await AutoTokenizer.from_pretrained(spec.repository, {
      revision: spec.revision ?? "main",
    }) as unknown as ChatTemplateTokenizer;
    await ensureChatTemplate(tokenizer, spec);
    return tokenizer;
  })().catch((error) => {
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
