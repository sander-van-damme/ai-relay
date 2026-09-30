import assert from "node:assert/strict";
import test from "node:test";
import {
  calendarDayBounds,
  effectiveInputCapacity,
  emptyQuotaState,
  estimateInputTokens,
  quotaDelayMs,
  reserveQuota,
  type QuotaPolicy,
} from "../src/quota.ts";

const rolling: QuotaPolicy = {
  maxConcurrent: null,
  dailyWindow: { type: "rolling" },
  limits: {
    requestsPerMinute: 2,
    inputTokensPerMinute: 100,
    requestsPerDay: 10,
    minimumSpacingMs: 0,
  },
};

test("input token estimate ignores relay model and stream fields", () => {
  const a = estimateInputTokens({ model: "a", stream: true, messages: [{ role: "user", content: "hello" }] });
  const b = estimateInputTokens({ model: "b", stream: false, messages: [{ role: "user", content: "hello" }] });
  assert.equal(a, b);
  assert.ok(a > 0);
});

test("effective request capacity is capped by TPM", () => {
  assert.equal(effectiveInputCapacity(rolling, 1_000_000), 100);
});

test("rolling RPM and TPM delays remain enforced", () => {
  const state = emptyQuotaState();
  reserveQuota(rolling, state, 40, 1_000);
  state.active = 0;
  reserveQuota(rolling, state, 40, 2_000);
  state.active = 0;
  assert.equal(quotaDelayMs(rolling, state, 10, 2_500), 58_500);
});

test("Google-style calendar day resets at Pacific midnight", () => {
  const policy: QuotaPolicy = {
    maxConcurrent: null,
    dailyWindow: { type: "calendar-day", timeZone: "America/Los_Angeles" },
    limits: {
      requestsPerMinute: null,
      inputTokensPerMinute: null,
      requestsPerDay: 1,
      minimumSpacingMs: 0,
    },
  };
  const state = emptyQuotaState();
  const requestAt = Date.parse("2026-10-01T06:30:00Z"); // Sep 30 23:30 PDT
  reserveQuota(policy, state, 1, requestAt);
  state.active = 0;
  const now = Date.parse("2026-10-01T06:45:00Z");
  assert.equal(quotaDelayMs(policy, state, 1, now), 15 * 60_000);
  assert.equal(calendarDayBounds(now, "America/Los_Angeles").end, Date.parse("2026-10-01T07:00:00Z"));
  assert.equal(quotaDelayMs(policy, state, 1, Date.parse("2026-10-01T07:00:01Z")), 0);
});
