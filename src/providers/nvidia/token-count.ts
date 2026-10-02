import { log } from "../../log.ts";
import type { ChatCompletionRequest } from "../../types.ts";

const NVIDIA_BASE_URL = "https://integrate.api.nvidia.com";
const TOKEN_PROBE_TIMEOUT_MS = 10_000;
const MAX_ERROR_DETAIL = 500;

type JsonObject = Record<string, unknown>;

export interface NvidiaTokenCountResult {
  count: number;
  method: "chat_render" | "tokenize";
}

interface ProbeResult {
  method: NvidiaTokenCountResult["method"];
  ok: boolean;
  count?: number;
  status?: number;
  detail?: string;
}

function object(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function detail(text: string): string {
  return text.replace(/\s+/g, " ").trim().slice(0, MAX_ERROR_DETAIL);
}

function tokenizerPayload(body: ChatCompletionRequest, upstreamModel: string): JsonObject {
  const payload: JsonObject = {
    model: upstreamModel,
    messages: body.messages,
  };
  if (body.tools !== undefined) payload.tools = body.tools;
  if (body.chat_template_kwargs !== undefined) payload.chat_template_kwargs = body.chat_template_kwargs;
  return payload;
}

async function postProbe(
  path: string,
  payload: JsonObject,
  apiKey: string,
  method: ProbeResult["method"],
  relayModelId: string,
): Promise<ProbeResult> {
  let response: Response;
  try {
    response = await fetch(`${NVIDIA_BASE_URL}${path}`, {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
        accept: "application/json",
        "user-agent": "ai-relay/1.0",
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(TOKEN_PROBE_TIMEOUT_MS),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("warn", `nvidia_tokenizer_probe_${method}_failed`, {
      provider: "nvidia",
      relay_model: relayModelId,
      reason: "network_error",
      detail: message,
    });
    return { method, ok: false, detail: message };
  }

  const bodyText = await response.text().catch(() => "");
  if (!response.ok) {
    log("warn", `nvidia_tokenizer_probe_${method}_failed`, {
      provider: "nvidia",
      relay_model: relayModelId,
      status: response.status,
      detail: detail(bodyText || response.statusText),
    });
    return {
      method,
      ok: false,
      status: response.status,
      detail: detail(bodyText || response.statusText),
    };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(bodyText) as unknown;
  } catch {
    const message = "Tokenizer endpoint returned invalid JSON.";
    log("warn", `nvidia_tokenizer_probe_${method}_failed`, {
      provider: "nvidia",
      relay_model: relayModelId,
      status: response.status,
      detail: message,
    });
    return { method, ok: false, status: response.status, detail: message };
  }

  const value = object(parsed);
  let count: number | undefined;
  if (method === "chat_render") {
    if (Array.isArray(value?.token_ids)) count = value.token_ids.length;
  } else {
    if (typeof value?.count === "number" && Number.isSafeInteger(value.count) && value.count >= 0) {
      count = value.count;
    } else if (Array.isArray(value?.tokens)) {
      count = value.tokens.length;
    }
  }

  if (count === undefined) {
    const message = method === "chat_render"
      ? "Render response did not contain token_ids."
      : "Tokenize response did not contain a valid count or tokens array.";
    log("warn", `nvidia_tokenizer_probe_${method}_failed`, {
      provider: "nvidia",
      relay_model: relayModelId,
      status: response.status,
      detail: message,
    });
    return { method, ok: false, status: response.status, detail: message };
  }

  log("info", `nvidia_tokenizer_probe_${method}_success`, {
    provider: "nvidia",
    relay_model: relayModelId,
    input_tokens: count,
  });
  return { method, ok: true, count, status: response.status };
}

export async function countNvidiaInputTokens(
  body: ChatCompletionRequest,
  relayModelId: string,
  upstreamModel: string,
  apiKey: string,
): Promise<NvidiaTokenCountResult> {
  const payload = tokenizerPayload(body, upstreamModel);
  const [render, tokenize] = await Promise.all([
    postProbe("/v1/chat/completions/render", payload, apiKey, "chat_render", relayModelId),
    postProbe("/tokenize", payload, apiKey, "tokenize", relayModelId),
  ]);

  if (render.ok && tokenize.ok) {
    if (render.count !== tokenize.count) {
      log("warn", "nvidia_tokenizer_probe_mismatch", {
        provider: "nvidia",
        relay_model: relayModelId,
        chat_render_tokens: render.count,
        tokenize_tokens: tokenize.count,
        selected_method: "chat_render",
      });
    } else {
      log("info", "nvidia_tokenizer_probe_agreement", {
        provider: "nvidia",
        relay_model: relayModelId,
        input_tokens: render.count,
      });
    }
    return { count: render.count!, method: "chat_render" };
  }

  if (render.ok) return { count: render.count!, method: "chat_render" };
  if (tokenize.ok) return { count: tokenize.count!, method: "tokenize" };

  throw new Error(
    `NVIDIA hosted token counting is unavailable for ${relayModelId}: `
    + `chat_render=${render.status ?? "network"} ${render.detail ?? "unknown error"}; `
    + `tokenize=${tokenize.status ?? "network"} ${tokenize.detail ?? "unknown error"}`,
  );
}
