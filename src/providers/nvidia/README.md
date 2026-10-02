# NVIDIA provider

The NVIDIA provider is implemented as an isolated provider using direct `fetch` calls to NVIDIA's hosted chat-completions API at `https://integrate.api.nvidia.com`.

## Current scope

The source catalog contains the 15 general chat/text free-endpoint candidates selected for this relay, in coding-preference order. Only one model is enabled while the hosted transport and authoritative token counting are being verified:

```text
nvidia/openai/gpt-oss-20b
```

The remaining models stay in the NVIDIA source catalog with `enabled: false`. Their upstream IDs, context limits, request compatibility, token counting and capabilities must be verified individually before enabling them.

NVIDIA does not currently have a hard-coded RPM, TPM, RPD, credit, or concurrency ceiling in the relay. A normal HTTP 429 is treated as model-scoped and triggers exponential model backoff. `Retry-After` is honored when present. Network failures, timeouts and 5xx responses are provider-scoped health failures. The relay does not infer an account-wide rate limit merely because several models are throttled.

Overflow offers are disabled until NVIDIA-specific overflow behavior is actually observed and justified.

## Token-count diagnostic

For the enabled model, the first offer evaluation for each request/model probes both hosted NIM/vLLM token-counting paths:

```text
POST https://integrate.api.nvidia.com/v1/chat/completions/render
POST https://integrate.api.nvidia.com/tokenize
```

Both probes receive the chat messages and tools rather than a manually concatenated prompt. Successful counts are cached per request/model so repeated offer evaluation does not repeat the probes.

The diagnostic emits deliberately visible structured log events:

```text
nvidia_tokenizer_probe_chat_render_success
nvidia_tokenizer_probe_chat_render_failed
nvidia_tokenizer_probe_tokenize_success
nvidia_tokenizer_probe_tokenize_failed
nvidia_tokenizer_probe_agreement
nvidia_tokenizer_probe_mismatch
```

If both methods work, the render count is selected because it comes from the full chat-completion rendering path. If only one works, that result is used. If both fail, NVIDIA returns a structured `token_count_failed` no-offer result and inference is not attempted.

The ordinary NVIDIA 429 response is also logged as `nvidia_rate_limit_response` so the actual hosted error body and `Retry-After` behavior can be observed before adding more specific quota logic.

## Configuration

Set:

```text
NVIDIA_API_KEY
```

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.
