import assert from "node:assert/strict";
import test from "node:test";
import { createNvidiaProvider } from "../src/providers/nvidia/index.ts";

const baseRequest = {
  offerKind: "standard" as const,
  body: { messages: [] },
  requestedModel: "auto",
  maxOptimizationWaitMs: 15_000,
  excludedModelIds: new Set<string>(),
};

test("NVIDIA provider does not advertise offers while implementation is incomplete", async () => {
  const originalKey = process.env.NVIDIA_API_KEY;
  process.env.NVIDIA_API_KEY = "test";

  try {
    const provider = createNvidiaProvider();
    const requests = [
      baseRequest,
      { ...baseRequest, requestedModel: "nvidia/openai/gpt-oss-120b" },
      { ...baseRequest, offerKind: "overflow" as const },
    ];

    for (const request of requests) {
      const result = await provider.getBestOffer(request, Date.now());
      assert.deepEqual(result, {
        status: "no_offer",
        providerId: "nvidia",
        reason: "no_eligible_model",
        detail: "NVIDIA provider is intentionally disabled until its provider implementation is completed.",
      });
    }
  } finally {
    if (originalKey === undefined) delete process.env.NVIDIA_API_KEY;
    else process.env.NVIDIA_API_KEY = originalKey;
  }
});
