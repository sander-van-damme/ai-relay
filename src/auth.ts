import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage } from "node:http";

function safeEqual(left: string, right: string): boolean {
  const a = Buffer.from(left);
  const b = Buffer.from(right);
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

export function isAuthorized(request: IncomingMessage, keys: readonly string[]): boolean {
  const header = request.headers.authorization;
  if (!header?.startsWith("Bearer ")) return false;
  const token = header.slice(7).trim();
  return keys.some((key) => safeEqual(token, key));
}
