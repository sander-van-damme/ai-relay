export function estimateInputTokens(body: Record<string, unknown>): number {
  const copy = { ...body };
  delete copy.model;
  delete copy.stream;
  const serialized = JSON.stringify(copy);
  return Math.max(1, Math.ceil(Buffer.byteLength(serialized, "utf8") / 4));
}

