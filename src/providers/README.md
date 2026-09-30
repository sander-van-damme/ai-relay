# Provider implementation contract

This directory contains concrete providers plus optional shared implementation helpers.

- Concrete providers live in `src/providers/<provider>/`.
- Reusable helpers live in `src/providers/shared/`.
- Shared helpers are implementation details, not mandatory provider behavior.
- A provider may use an official SDK, an OpenAI-compatible API, direct HTTP, or another transport.

## Core invariant

A `ProviderOffer` is a claim that a concrete model can handle the request and gives the earliest time the provider currently believes it can start.

Never return an offer without applying every known hard constraint. A provider that knows a route is blocked must not keep advertising that route as immediately usable.

## Model capacity

`listModels()` must expose stable relay model IDs and an effective single-request input capacity.

Capacity must account for all hard limits that can make one request impossible, including:

- context/input window limits;
- provider token-per-request ceilings, plus token-per-minute ceilings only when they make a single request impossible rather than merely delaying it;
- account/tier restrictions known to the provider.

A request that can never fit must not produce an offer.

## Input token counting

Input token counting is provider-owned and model-specific.

The relay must not apply a provider-independent token estimate. Before advertising an offer, a provider must determine the input token count for the concrete model through `countInputTokens()`. It may use a local model tokenizer, an official SDK, or an authoritative provider counting endpoint.

`countInputTokens()` is asynchronous because some providers can only determine the authoritative count through I/O. Providers with local tokenizers may still return the result immediately through the same async interface.

The count must reflect the actual request representation seen by that model, including provider-specific chat formatting, system/developer instructions, tool definitions and other input that consumes the model's context. If a provider cannot determine an authoritative count for a request, it must not fall back to a generic bytes/characters heuristic.

`ProviderOffer.inputTokens` is the authoritative count used for that offer. Execution and quota accounting must reuse that exact value rather than tokenizing the request again.

Providers may cache token counts per request/model so repeated standard/overflow evaluation does not repeat expensive tokenization or network calls. A failed count must not be cached permanently when a later retry could succeed.

The current Google provider uses the official `@google/genai` `countTokens` API for counting while generation still uses Google's OpenAI-compatible transport. The current NVIDIA GPT-OSS provider uses the model's `o200k_harmony` tokenizer locally.

## Offer selection

`getBestOffer()` is asynchronous and owns selection inside one provider. It must consider:

- requested model and `auto`;
- the model-specific input token count;
- context/capacity limits;
- RPM, TPM, RPD or equivalent quotas;
- provider-wide quotas;
- concurrency limits;
- model/provider cooldowns;
- unavailable or unsupported models;
- whether the provider is configured.

If a request can be handled later, `availableAt` must describe the earliest realistic start time. Cross-provider comparison belongs only to the relay scheduler.

## Standard and overflow offers

The scheduler requests offers in two modes:

- `standard`: normal provider behavior using the provider's known quota and health state;
- `overflow`: an optional speculative last-resort attempt after no standard offer is usable inside the scheduler's normal optimization window.

A provider must return the same kind in `ProviderOffer.kind` that was requested through `OfferRequest.offerKind`.

Overflow offers are a relay-side speculative behavior, not a capability that an upstream provider is expected to support or document. They should only be enabled for a concrete provider and quota dimension when we have empirical reason to suspect that the upstream may sometimes accept requests beyond the limit we normally track. Do not enable overflow merely because another provider has shown similar behavior.

An overflow offer may only relax quota dimensions that the relay implementation has explicitly marked as overflow-eligible based on such observations. It must never ignore:

- model context/input capacity;
- a hard single-request token ceiling;
- concurrency limits;
- minimum request spacing;
- provider/model health cooldowns;
- quota dimensions that were not explicitly marked as overflowable.

For example, the current Google integration may use speculative overflow for requests-per-day because observed behavior has sometimes allowed requests beyond the nominal daily limit. This is not a public or documented Google capability, and it must not be treated as guaranteed behavior. RPM and TPM exhaustion continue to behave as normal queueing constraints.

Capacity rules still apply during overflow selection. For a 200,000-token request, a 16,000-token model is never eligible, while an otherwise quota-exhausted model with at least 200,000 tokens of effective request capacity may be eligible for an overflow offer.

A provider should only expose an overflow offer when the request would otherwise be delayed specifically by an opted-in quota. It should not expose speculative offers while the same model is normally available.

The scheduler always prefers a standard offer that is usable inside its optimization window. Only when no such standard offer exists does it ask providers for overflow offers. If no overflow offer is available, the scheduler returns to the earliest standard offer and keeps the request queued.

### Learning a hard quota cap

A failed speculative attempt can provide stronger evidence than the local quota model.

When an overflow request receives a provider quota response such as HTTP `429`, the provider may remember that the attempted model has reached a hard cap and temporarily stop advertising overflow for that model.

The hard-cap observation TTL is provider-defined. The current Google/OpenAI-compatible implementation uses 24 hours. Its cache is process-local and only suppresses overflow offers; normal offers still become available at their usual quota reset time.

Network failures, timeouts and `5xx` responses must not mark the quota as a hard cap because they do not prove that the quota caused the failure.

A successful overflow request does not mark the model as capped. If its normal quota remains exhausted, the provider may offer another speculative attempt later.

For `auto` routing, a model-scoped overflow failure should return to the scheduler rather than starting a hidden retry loop inside `execute()`. The scheduler can ask the provider for another overflow offer with the failed model excluded, allowing the provider to return its next eligible model. This keeps each upstream attempt visible to queue fairness, retry budgets and logging.

## Quota ownership

Quota semantics belong to the provider. `shared/quota.ts` can be reused when its semantics match, but a provider should implement its own rules when they differ.

Do not generalize shared quota behavior just to force a provider into it. Calendar-day resets, rolling windows, minimum spacing and account-specific limits must follow the upstream provider.

## Provider SDKs and transports

Prefer an official provider SDK when it exposes capabilities or models that the OpenAI-compatible endpoint does not.

The rest of the relay must not depend on whether a provider uses an official SDK, the shared OpenAI-compatible helper, direct HTTP or another transport.

`shared/openai-compatible.ts` is a convenience implementation, not the provider abstraction itself.

## Failures

Every non-successful execution must be classified by scope:

- `model`: another model on the same provider may still work;
- `provider`: switching models on that provider is unlikely to help.

Typical model-scoped failures include model rate limits or model-specific rejection. Typical provider-scoped failures include invalid credentials, provider outages, network failures and provider-wide limits.

A transient failure must update cooldown/health state before returning `retryable`. A permanent error must return `rejected`, not an endless retry.

Cooldowns represent provider health. They must not be tuned around scheduler constants such as the optimization wait.

The scheduler separately enforces a finite per-request budget of three retryable failures for each provider/model path. Once exhausted, that path is excluded for that request. This guarantees that a persistently failing path cannot keep a request alive forever.

## Provider state versus request state

Provider-global state may include quota usage, cooldowns, rate-limit reset times and consecutive upstream failures.

Per-request routing history, retry budgets and exclusions belong to the scheduler.

## Required tests

A provider is incomplete until tests cover at least:

1. Requests larger than hard capacity produce no offer.
2. Quota exhaustion changes `availableAt` correctly.
3. Provider and model cooldowns are respected.
4. Model-scoped failures do not unnecessarily disable the provider.
5. Provider-scoped failures do not get retried through another model on the same provider.
6. Authentication/configuration errors are not endlessly retried.
7. Repeated transient failures cannot create an infinite request loop.
8. Successful execution releases concurrency state and records quota usage correctly.
9. Explicit model selection stays on the requested model.
10. `auto` returns the provider's best valid offer according to that provider's rules.
11. Standard offers are preferred over overflow offers.
12. Overflow never exceeds hard request capacity or non-opted-in quota dimensions.
13. A quota rejection from an overflow attempt suppresses repeated probes for the provider-defined hard-cap TTL.
14. Network/`5xx` failures do not incorrectly create a hard-cap observation.
15. Token counting is model-specific and no provider-independent estimate is used.
16. The exact token count on the selected offer is reused for quota accounting and execution.
17. Repeated offer evaluation reuses a cached successful token count when appropriate.

## Adding a provider

A new provider should normally require changes only under `src/providers/`, registration in `src/providers/index.ts`, credentials documentation and tests.

If adding a provider requires `if (provider === ...)` logic in `relay.ts`, reconsider the provider interface first.

Google currently uses `@google/genai` for authoritative token counting. A later migration of generation itself to the native Google SDK should remain entirely inside the Google provider so provider-native functionality does not leak into the scheduler.
