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
- provider token-per-request or token-per-minute ceilings when they bound one request;
- account/tier restrictions known to the provider.

A request that can never fit must not produce an offer.

## Offer selection

`getBestOffer()` owns selection inside one provider. It must consider:

- requested model and `auto`;
- estimated input tokens;
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

Overflow support is optional and provider-specific. Do not assume that an upstream accepts requests beyond a documented quota simply because another provider sometimes does.

An overflow offer may only relax quota dimensions that the concrete provider has explicitly opted into. It must never ignore:

- model context/input capacity;
- a hard single-request token ceiling;
- concurrency limits;
- minimum request spacing;
- provider/model health cooldowns;
- quota dimensions that were not explicitly marked as overflowable.

For example, the current Google provider allows speculative overflow for requests-per-day only. RPM and TPM exhaustion continue to behave as normal queueing constraints.

Capacity rules still apply during overflow selection. For a 200,000-token request, a 16,000-token model is never eligible, while an otherwise quota-exhausted model with at least 200,000 tokens of effective request capacity may be eligible for an overflow offer.

A provider should only expose an overflow offer when the request would otherwise be delayed specifically by an opted-in quota. It should not expose speculative offers while the same model is normally available.

The scheduler always prefers a standard offer that is usable inside its optimization window. Only when no such standard offer exists does it ask providers for overflow offers. If no overflow offer is available, the scheduler returns to the earliest standard offer and keeps the request queued.

### Learning a hard quota cap

A failed speculative attempt can provide stronger evidence than the local quota model.

When an overflow request receives a provider quota response such as HTTP `429`, the provider may remember that the attempted model has reached a hard cap and temporarily stop advertising overflow for that model.

The current Google/OpenAI-compatible implementation caches that observation for 24 hours. The cache is process-local and only suppresses overflow offers; normal offers still become available at their usual quota reset time.

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
8. Successful execution releases concurrency/quota state.
9. Explicit model selection stays on the requested model.
10. `auto` returns the provider's best valid offer according to that provider's rules.
11. Standard offers are preferred over overflow offers.
12. Overflow never exceeds hard request capacity or non-opted-in quota dimensions.
13. A quota rejection from an overflow attempt suppresses repeated probes for the provider-defined hard-cap TTL.
14. Network/`5xx` failures do not incorrectly create a hard-cap observation.

## Adding a provider

A new provider should normally require changes only under `src/providers/`, registration in `src/providers/index.ts`, credentials documentation and tests.

If adding a provider requires `if (provider === ...)` logic in `relay.ts`, reconsider the provider interface first.

Google is a candidate for a later migration to `@google/genai`. That migration should remain entirely inside the Google provider so provider-native functionality does not leak into the scheduler.
