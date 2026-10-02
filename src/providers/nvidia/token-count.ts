import { countChatCompletionTokens } from "gpt-tokenizer/model/gpt-oss-20b";
import type { InputTokenCounter } from "../shared/openai-compatible.ts";

export const countNvidiaInputTokens: InputTokenCounter = (body, model) => {
  if (model.upstreamModel !== "openai/gpt-oss-20b") {
    throw new Error(`Unsupported NVIDIA tokenizer model: ${model.upstreamModel}`);
  }
  if (!countChatCompletionTokens) {
    throw new Error("gpt-tokenizer does not expose chat-completion counting for gpt-oss-20b.");
  }
  const count = countChatCompletionTokens({
    ...body,
    model: "gpt-oss-20b",
  } as never);
  if (!Number.isSafeInteger(count) || count < 0) {
    throw new Error(`gpt-tokenizer returned an invalid token count for ${model.upstreamModel}: ${count}`);
  }
  return count;
};
