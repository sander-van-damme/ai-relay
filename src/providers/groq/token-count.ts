import { countChatCompletionTokens as countGptOssChatCompletionTokens } from "gpt-tokenizer/model/gpt-oss-20b";
import type { ChatCompletionRequest } from "../../types.ts";

const TOKENIZER_ASSET_TIMEOUT_MS = 15_000;
const QWEN_IMAGE_TOKENS = 2_048;

export type GroqTokenizerSpec =
  | { kind: "gpt-oss" }
  | {
      kind: "huggingface";
      repository: string;
      revision?: string;
      visionImageTokens?: number;
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

function tokenizerCacheKey(spec: Extract<GroqTokenizerSpec, { kind: "huggingface" }>): string {
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
  spec: Extract<GroqTokenizerSpec, { kind: "huggingface" }>,
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
  spec: Extract<GroqTokenizerSpec, { kind: "huggingface" }>,
): Promise<ChatTemplateTokenizer> {
  const key = tokenizerCacheKey(spec);
  const cached = huggingFaceTokenizerCache.get(key);
  if (cached) return cached;

  const pending = (async () => {
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
    throw new Error("Groq token counting requires an OpenAI-style messages array.");
  }
  return body.messages;
}

function tools(body: ChatCompletionRequest): unknown[] | undefined {
  return Array.isArray(body.tools) ? body.tools : undefined;
}

function object(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function imageCount(body: ChatCompletionRequest): number {
  let count = 0;
  for (const rawMessage of messages(body)) {
    const message = object(rawMessage);
    if (!message || !Array.isArray(message.content)) continue;
    for (const rawPart of message.content) {
      const part = object(rawPart);
      if (part?.type === "image_url" || part?.type === "image") count += 1;
    }
  }
  return count;
}

function qwenTemplateKwargs(body: ChatCompletionRequest): Record<string, unknown> {
  const value = body.chat_template_kwargs;
  const options = typeof value === "object" && value !== null && !Array.isArray(value)
    ? { ...value as Record<string, unknown> }
    : {};

  // Groq defaults Qwen 3.8 to non-reasoning mode, whereas the upstream Qwen
  // chat template defaults to thinking. Set the template switches explicitly
  // so local counting matches the request Groq actually serves.
  const effort = body.reasoning_effort;
  if (effort === undefined || effort === null || effort === "none") {
    options.enable_thinking = false;
  } else {
    options.enable_thinking = true;
    options.reasoning_effort = effort === "low" || effort === "medium"
      ? effort
      : "xhigh";
  }
  return options;
}

function countGptOss(body: ChatCompletionRequest): number {
  if (!countGptOssChatCompletionTokens) {
    throw new Error("gpt-tokenizer does not expose chat-completion counting for GPT-OSS.");
  }

  // GPT-OSS 20B and 120B share the same Harmony tokenizer and chat format.
  const count = countGptOssChatCompletionTokens({
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
  spec: Extract<GroqTokenizerSpec, { kind: "huggingface" }>,
): Promise<number> {
  const tokenizer = await loadHuggingFaceTokenizer(spec);
  const requestTools = tools(body);
  const requestDocuments = Array.isArray(body.documents) ? body.documents : undefined;

  const rendered = tokenizer.apply_chat_template(messages(body), {
    ...qwenTemplateKwargs(body),
    tokenize: true,
    return_tensor: false,
    return_dict: false,
    add_generation_prompt: true,
    ...(requestTools ? { tools: requestTools } : {}),
    ...(requestDocuments ? { documents: requestDocuments } : {}),
  });

  let count = tokenLength(rendered);
  if (count === null || !Number.isSafeInteger(count) || count < 0) {
    throw new Error(
      `Hugging Face tokenizer ${spec.repository} returned an unsupported tokenized result.`,
    );
  }

  const images = imageCount(body);
  if (images > 0 && spec.visionImageTokens) {
    // The Qwen chat template renders one <|image_pad|> token per image, while
    // Groq documents each image as consuming 2,048 input tokens. Replace the
    // single placeholder token with Groq's documented accounting value.
    count += images * (spec.visionImageTokens - 1);
  }
  return count;
}

export async function countGroqInputTokens(
  body: ChatCompletionRequest,
  tokenizer: GroqTokenizerSpec,
): Promise<number> {
  switch (tokenizer.kind) {
    case "gpt-oss":
      return countGptOss(body);
    case "huggingface":
      return countHuggingFaceChat(body, tokenizer);
  }
}

export const GROQ_QWEN_IMAGE_TOKENS = QWEN_IMAGE_TOKENS;
