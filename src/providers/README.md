# Provider implementation contract

This directory contains concrete providers plus optional shared implementation helpers.

- Concrete providers live in `src/providers/<provider>/`.
- Reusable helpers live in `src/providers/shared/`.
- Shared helpers are implementation details, not mandatory provider behavior.
- A provider owns its transport choice, such as an official SDK, direct HTTP, or another provider-specific API.

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

Input token counting is provider-owned and model-specific. It is an implementation detail of offer evaluation, not a capability exposed through the `Provider` interface.

The relay must not apply a provider-independent token estimate or ask a provider to count a model directly. Before advertising an offer, a provider must determine the input token count for each concrete model it evaluates. It may use a local model tokenizer, an official SDK, or an authoritative provider counting endpoint.

Counting may be asynchronous because some providers can only determine the authoritative count through I/O. The count must reflect the actual request representation seen by that model, including provider-specific chat formatting, system/developer instructions, tool definitions and other input that consumes the model's context. If a provider cannot determine an authoritative count for a request, it must not fall back to a generic bytes/characters heuristic.

`ProviderOffer.inputTokens` is the authoritative count used for that offer. Execution and quota accounting must reuse that exact value rather than tokenizing the request again.

Token-counting implementations and caches remain provider-owned even when two providers expose the same underlying model. Providers can apply different chat templates, injected instructions, tool serialization, special tokens or deployment-specific preprocessing, so a count produced for one provider must not be reused by another unless their rendered-input semantics have been explicitly proven identical. Sharing a low-level tokenizer library is fine; sharing a higher-level rendered-request count is not the default.

Providers may cache token counts per request/model so repeated standard/overflow evaluation does not repeat expensive tokenization or network calls. A failed count must not be cached permanently when a later retry could succeed.

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

`getBestOffer()` returns a structured `ProviderOfferResult`. A successful result contains the concrete offer. A no-offer result explains why the provider could not make one, such as missing configuration, no eligible model, hard request capacity, or authoritative token-count failure. The relay may use those reasons for terminal error classification, but it must not reproduce the provider's feasibility checks itself.

## Standard and overflow offers

The scheduler requests offers in two modes:

- `standard`: normal provider behavior using the provider's known quota and health state;
- `overflow`: an optional speculative last-resort attempt after no standard offer is usable inside the scheduler's normal optimization window.

A provider must return the same kind in `ProviderOffer.kind` that was requested through `OfferRequest.offerKind`.

Overflow offers are a relay-side speculative behavior, not a capability that an upstream provider is expected to support or document. They should only be implemented when observation or testing gives us reason to suspect that an upstream may sometimes accept requests beyond a limit that would normally block the request. Do not infer overflow behavior from another provider, model, account, or limit.

The provider contract does not globally define which kinds of limits may or may not be overflowed. A concrete provider may implement overflow for any specific constraint where observed upstream behavior justifies doing so, including request, token, rate, capacity, context, concurrency, spacing, or other limits. The exact scope and behavior are provider-specific.

An overflow offer may relax only the constraints that the concrete provider implementation has deliberately made overflowable. All other constraints continue to apply normally. This is an implementation decision based on observed behavior, not a claim that the upstream publicly supports or guarantees exceeding that limit.

A provider should only expose an overflow offer when the request would otherwise be blocked or delayed by a constraint that its overflow implementation is specifically prepared to probe. It should not expose speculative offers while the same request is normally available.

The scheduler always prefers a standard offer that is usable inside its optimization window. Only when no such standard offer exists does it ask providers for overflow offers. If no overflow offer is available, the scheduler returns to the earliest standard offer and keeps the request queued.

### Learning an overflow boundary

A failed speculative attempt can provide stronger evidence than the provider's local model of an upstream limit.

When an overflow attempt receives a response that clearly shows the probed limit was reached, the provider may remember that observation and temporarily stop advertising equivalent overflow attempts for that model or constraint.

How such observations are classified, scoped and expired is provider-defined. Failures that do not establish that the probed limit caused the rejection, such as unrelated network failures, timeouts or upstream `5xx` responses, must not be treated as evidence of an overflow boundary.

A successful overflow request does not establish a new guaranteed limit. The provider may continue making speculative overflow offers according to its own implementation and observations.

For `auto` routing, a model-scoped overflow failure should return to the scheduler rather than starting a hidden retry loop inside `execute()`. The scheduler can ask the provider for another overflow offer with the failed model excluded, allowing the provider to return its next eligible model. This keeps each upstream attempt visible to queue fairness, retry budgets and logging.

## Quota ownership

Quota semantics belong to the provider. `shared/quota.ts` can be reused when its semantics match, but a provider should implement its own rules when they differ.

Do not generalize shared quota behavior just to force a provider into it. Calendar-day resets, rolling windows, minimum spacing and account-specific limits must follow the upstream provider.

## Provider SDKs and transports

Prefer an official provider SDK when it exposes capabilities or models that are not available through the provider's other APIs.

The rest of the relay must not depend on whether a provider uses an official SDK, direct HTTP, or another transport.

## Failures

Every non-successful execution must be classified by scope:

- `model`: another model on the same provider may still work;
- `provider`: switching models on that provider is unlikely to help.

Typical model-scoped failures include model rate limits or model-specific rejection. Typical provider-scoped failures include invalid credentials, provider outages, network failures and provider-wide limits.

A transient failure must update cooldown/health state before returning `retryable`. A permanent error must return `rejected`, not an endless retry.

Cooldowns represent provider health. They must not be tuned around scheduler constants such as the optimization wait.

The scheduler separately enforces a finite per-request budget of three retryable failures for each provider/model path. Once exhausted, that path is excluded for that request. This guarantees that a persistently failing path cannot keep a request alive forever.

An unexpected exception escaping `provider.execute()` is contained by the scheduler and treated as a provider-scoped retryable failure for that request. It consumes the same three-attempt path budget. For `auto`, the next routing pass avoids the throwing provider when an alternative is available; for an explicitly selected model, the same provider may be retried only until that finite budget is exhausted, after which the request terminates with a relay error. Providers should still classify expected upstream failures themselves rather than throwing.

## Observability boundary

Observability must not create provider-specific hooks or inspect concrete provider implementations. The relay may observe only:

- facts generated by the relay or scheduler itself; and
- values exposed through the generic `Provider` contract, including `listModels()`, `status()`, `ProviderOfferResult`, and `ProviderExecutionResult`.

`ProviderOffer.inputTokens` is the routing/preflight count for the concrete offer. `ProviderExecutionResult.usage`, when present, is a separate upstream-reported measurement whose input, output and total dimensions are independent. Providers must expose every token dimension the upstream actually reports and leave absent dimensions undefined; they must not fabricate missing upstream usage from the routing count or from other usage dimensions. Observability tracks a separate reported-success count for each dimension so partial aggregate coverage stays explicit.

Provider-specific transports, quota internals, candidate evaluation details, token-counting implementations and private upstream payloads remain implementation details. If additional observability data is genuinely required across providers, add it to the generic contract only when it is also part of normal relay/provider interaction rather than a logging-only capability.

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
12. Overflow relaxes only the constraints that the concrete provider has explicitly implemented as overflowable, while all other provider constraints continue to apply.
13. A rejection that clearly identifies the probed overflow boundary can suppress equivalent speculative probes according to provider-defined behavior.
14. Unrelated failures such as network errors, timeouts or `5xx` responses do not incorrectly create an overflow-boundary observation.
15. Token counting is model-specific and no provider-independent estimate is used.
16. The exact token count on the selected offer is reused for quota accounting and execution.
17. Repeated offer evaluation reuses a cached successful token count when appropriate.
18. Token counting is not exposed through the public `Provider` interface.
19. No-offer results distinguish hard capacity from configuration, eligibility and counting failures.

## Adding a provider

A new provider should normally require changes only under `src/providers/`, registration in `src/providers/index.ts`, credentials documentation and tests.

If adding a provider requires `if (provider === ...)` logic in `relay.ts`, reconsider the provider interface first.

