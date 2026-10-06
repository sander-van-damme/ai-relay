export interface SseEventDecision<T> {
  value?: T;
  forward?: boolean;
}

export interface ObservedSseResponse<T> {
  response: Response;
  observed: Promise<T | undefined>;
}

function eventData(event: string): string | null {
  const data = event
    .split(/\r\n|\r|\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n")
    .trim();
  return data || null;
}

function takeEvent(buffer: string): { event: string; rest: string } | null {
  const boundary = /(?:\r\n|\r|\n)(?:\r\n|\r|\n)/.exec(buffer);
  if (!boundary || boundary.index === undefined) return null;
  const end = boundary.index + boundary[0].length;
  return {
    event: buffer.slice(0, end),
    rest: buffer.slice(end),
  };
}

/**
 * Observe complete SSE events while exposing a single downstream body.
 *
 * The wrapper reads the upstream response only from the downstream stream's
 * pull() path. At most the current upstream chunk plus framed-but-not-yet-
 * forwarded data is buffered, so telemetry cannot drain the source ahead of
 * downstream backpressure as a cloned response branch can.
 */
export function observeSseResponse<T>(
  response: Response,
  inspect: (data: string) => SseEventDecision<T> | void,
): ObservedSseResponse<T> {
  if (!response.body) {
    return { response, observed: Promise.resolve(undefined) };
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let latest: T | undefined;
  let settled = false;
  let resolveObserved!: (value: T | undefined) => void;
  const observed = new Promise<T | undefined>((resolve) => {
    resolveObserved = resolve;
  });

  const settle = (value: T | undefined): void => {
    if (settled) return;
    settled = true;
    resolveObserved(value);
  };

  const inspectEvent = (event: string): boolean => {
    const data = eventData(event);
    if (!data) return true;
    const decision = inspect(data);
    if (decision?.value !== undefined) latest = decision.value;
    return decision?.forward !== false;
  };

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        for (;;) {
          const framed = takeEvent(buffer);
          if (framed) {
            buffer = framed.rest;
            if (inspectEvent(framed.event)) {
              controller.enqueue(encoder.encode(framed.event));
              return;
            }
            continue;
          }

          const { done, value } = await reader.read();
          if (!done) {
            buffer += decoder.decode(value, { stream: true });
            continue;
          }

          buffer += decoder.decode();
          if (buffer) {
            const trailing = buffer;
            buffer = "";
            if (inspectEvent(trailing)) controller.enqueue(encoder.encode(trailing));
          }
          settle(latest);
          controller.close();
          reader.releaseLock();
          return;
        }
      } catch (error) {
        settle(undefined);
        try {
          reader.releaseLock();
        } catch {
          // The lock may already have been released after an upstream failure.
        }
        controller.error(error);
      }
    },
    async cancel(reason) {
      try {
        await reader.cancel(reason);
      } finally {
        settle(undefined);
        try {
          reader.releaseLock();
        } catch {
          // Ignore cancellation races.
        }
      }
    },
  });

  return {
    response: new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    }),
    observed,
  };
}
