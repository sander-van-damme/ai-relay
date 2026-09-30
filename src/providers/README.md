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

## Adding a provider

A new provider should normally require changes only under `src/providers/`, registration in `src/providers/index.ts`, credentials documentation and tests.

If adding a provider requires `if (provider === ...)` logic in `relay.ts`, reconsider the provider interface first.

Google is a candidate for a later migration to `@google/genai`. That migration should remain entirely inside the Google provider so provider-native functionality does not leak into the scheduler.
