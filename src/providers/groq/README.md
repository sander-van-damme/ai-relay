# Groq provider

The Groq provider is an isolated direct-HTTP implementation of Groq's OpenAI-compatible Chat Completions API.

It deliberately does **not** use a shared OpenAI-compatible transport abstraction. Groq-specific request support, model capabilities, rate-limit headers, quota reconciliation, error classification, token counting, and retry behavior remain under this directory.

## Configuration

Set:

```text
GROQ_API_KEY=...
```

The provider sends requests to:

```text
POST https://api.groq.com/openai/v1/chat/completions
Authorization: Bearer <GROQ_API_KEY>
```

## Active model catalog

The checked-in catalog reflects Groq's Free plan as verified on 2026-10-03 and contains the current general-purpose Chat Completions routes relevant to this relay:

| Relay model | Upstream model | Context | Max output | Free limits | Notes |
| --- | --- | ---: | ---: | --- | --- |
| `groq/openai/gpt-oss-120b` | `openai/gpt-oss-120b` | 131,072 | 65,536 | 30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD | text, reasoning |
| `groq/qwen/qwen3.8-27b` | `qwen/qwen3.8-27b` | 131,072 | 16,384 | 30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD | text + vision, reasoning, preview |
| `groq/openai/gpt-oss-20b` | `openai/gpt-oss-20b` | 131,072 | 65,536 | 30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD | text, reasoning |

The relay's effective single-request input capacity for these Free-plan routes is **8,000 tokens**, not 131,072, because a request that exceeds the Free-plan TPM ceiling cannot be admitted as a single request.

Specialized safety, prompt-guard, speech-to-text, and text-to-speech models are intentionally not exposed as normal chat routes.

Groq changes its hosted catalog over time. Before adding, removing, or enabling a model, verify both the current model page and the current rate-limit table rather than copying historical model IDs.

## Token counting

Token counting is provider-owned and model-specific.

### GPT-OSS

GPT-OSS 20B and 120B use the Harmony tokenizer/chat format. The provider counts both locally with `gpt-tokenizer/model/gpt-oss-20b`; the two hosted GPT-OSS variants share that tokenizer format.

The successful upstream response is later compared with Groq's `usage.prompt_tokens`. A mismatch is logged so drift between local rendering and the hosted route is observable.

### Qwen 3.8

Qwen uses the official `Qwen/Qwen3.8-27B` tokenizer through Transformers.js and its chat template.

Groq's reasoning defaults differ from the upstream template defaults, so local rendering explicitly maps Groq's `reasoning_effort` behavior before tokenizing.

For vision requests, Groq documents each image as consuming 2,048 input tokens. The tokenizer renders an image placeholder; the provider replaces that placeholder accounting with Groq's documented per-image charge.

No generic bytes/characters estimate is used. If authoritative model-specific counting cannot be performed, the provider returns `token_count_failed`.

Successful counts are cached per request object and concrete model.

## Request capabilities

The provider forwards the caller's OpenAI-style Chat Completions request, replaces `model` with the selected upstream Groq model, and sets `stream` to the relay-selected mode.

Known incompatible request shapes are rejected during offer evaluation rather than advertised and then predictably rejected upstream. In particular:

- `logprobs`, `top_logprobs`, and `logit_bias` are not routed;
- `messages[].name` is not routed;
- `n` must be absent or `1`;
- non-zero frequency/presence penalties are not routed;
- `metadata` and `store` are not routed;
- `include_reasoning` and `reasoning_format` may not both be set;
- `documents`, `chat_template_kwargs`, Compound/search extensions, and deprecated `functions` / `function_call` are not routed because this provider does not have a proven matching token-count/rendering path for them;
- tools, when present, must be OpenAI function tools;
- Free-plan routing accepts only the default/on-demand service-tier behavior (`auto`, `on_demand`, or omitted);
- none of the current Free-plan chat routes advertise requests that explicitly require parallel local function calls; for ordinary tool requests the provider sends `parallel_tool_calls=false` to avoid Groq's default `true` on models that do not support it;
- images route only to Qwen, with at most three images per request;
- `max_completion_tokens` / `max_tokens`, when supplied, must fit the selected model's published max output.

Reasoning support is model-specific:

- GPT-OSS: `low`, `medium`, `high`;
- Qwen 3.8: `none`, `default`, `low`, `medium`, `high`.

## Quotas and availability

The shared quota implementation is reused unchanged for mechanics that match Groq:

- 30 requests/minute;
- 1,000 requests/day;
- 8,000 token/minute hard input admission capacity;
- active-request accounting;
- model cooldown state.

Groq's Free plan also has a **200,000 token/day** limit that is not represented by the shared quota contract. That limit is therefore tracked inside this provider.

The public Groq documentation does not define a calendar boundary for the Free-plan daily counters. Until a more authoritative reset rule is available, local RPD/TPD prediction uses the existing rolling-window mechanics. Upstream responses remain authoritative: Groq's rate-limit/reset headers and `429` responses can extend local blocking when the service reports a stricter current state.

Groq limits are organization-level, so multiple API keys in one organization do not create independent quota pools.

## Combined-token and cached-token accounting

Groq TPM/TPD consumption is based on token usage rather than input alone, while the relay must decide whether to offer a route before output exists.

The provider therefore:

1. reserves the authoritative input count before execution;
2. asks Groq to include usage;
3. after success, reconciles that reservation to actual charged usage:
   `uncached prompt tokens + completion tokens`.

Groq prompt caching can report `usage.prompt_tokens_details.cached_tokens`. Cached prompt tokens are subtracted from local charged quota because Groq documents cached tokens as not counting toward rate limits.

For streaming requests the provider forces `stream_options.include_usage=true` upstream so quota accounting can be reconciled from the terminal usage event. The relay observes that event on the same backpressured stream; the usage-only SSE event is forwarded only when the client requested `stream_options.include_usage=true`. Groq's reported `prompt_tokens`, `completion_tokens`, and `total_tokens` are exposed independently through `ProviderUsage`, so observability can aggregate each dimension with explicit coverage without inventing missing values.

## Rate-limit headers

The provider observes Groq's rate-limit response headers when available:

- `x-ratelimit-remaining-requests`
- `x-ratelimit-reset-requests`
- `x-ratelimit-remaining-tokens`
- `x-ratelimit-reset-tokens`
- `retry-after`

Reset values such as `7.66s` or compound duration strings are parsed provider-locally. If Groq reports zero remaining requests or tokens, the affected model is not advertised as immediately available before that reset.

## Failure classification

- `429`: retryable, model-scoped rate limit; `Retry-After` and reset headers are respected.
- `498 capacity_exceeded`: retryable, model-scoped. The provider does not request Flex itself, but recognizes this Groq-specific response defensively.
- `408` and `5xx`: retryable, provider-scoped.
- network/transport failures: retryable, provider-scoped.
- `401`, `403`, and Groq `blocked_api_access`: rejected, provider-scoped.
- other upstream `4xx`: rejected, model-scoped.

Repeated transient failures use bounded exponential cooldowns. A successful request resets the corresponding consecutive failure counters.

## Overflow

Groq overflow offers are intentionally **not implemented**.

The relay's overflow mode is speculative and should only exist after observed upstream behavior justifies probing a documented/local limit. No such Groq behavior has been established for this provider.

## Sources

Current provider assumptions should be checked against Groq's official documentation:

- https://console.groq.com/docs/models
- https://console.groq.com/docs/rate-limits
- https://console.groq.com/docs/api-reference
- https://console.groq.com/docs/openai
- https://console.groq.com/docs/reasoning
- https://console.groq.com/docs/vision
- https://console.groq.com/docs/prompt-caching
- https://console.groq.com/docs/deprecations
