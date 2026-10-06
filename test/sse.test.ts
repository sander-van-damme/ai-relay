import assert from "node:assert/strict";
import test from "node:test";
import { observeSseResponse } from "../src/providers/shared/sse.ts";

test("SSE observation cannot drain the upstream stream ahead of a slow consumer", async () => {
  const encoder = new TextEncoder();
  const events = [
    'data: {"choices":[{"delta":{"content":"one"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"two"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"three"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"four"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"five"}}]}\n\n',
    'data: {"choices":[],"usage":{"completion_tokens":5}}\n\n',
    "data: [DONE]\n\n",
  ];

  let upstreamPulls = 0;
  let index = 0;
  const upstream = new Response(new ReadableStream<Uint8Array>({
    pull(controller) {
      upstreamPulls += 1;
      if (index >= events.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(events[index++]!));
    },
  }), { headers: { "content-type": "text/event-stream" } });

  const observed = observeSseResponse(upstream, (data) => {
    if (data.includes('"usage"')) return { value: 5 };
  });

  // Give the wrapper enough time to fill its own single downstream queue slot.
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.ok(upstreamPulls < events.length, `upstream was drained early after ${upstreamPulls} pulls`);

  const reader = observed.response.body!.getReader();
  const first = await reader.read();
  assert.equal(first.done, false);
  assert.match(new TextDecoder().decode(first.value), /"one"/);

  // Even after one slow downstream read, the observer may prefetch only bounded
  // stream data; it must not consume the complete provider response.
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.ok(upstreamPulls < events.length, `upstream was drained after one downstream read (${upstreamPulls} pulls)`);

  for (;;) {
    const { done } = await reader.read();
    if (done) break;
  }
  assert.equal(await observed.observed, 5);
});
