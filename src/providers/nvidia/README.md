# NVIDIA provider

The NVIDIA provider is implemented as an isolated provider using direct `fetch` calls to NVIDIA's hosted chat-completions API at `https://integrate.api.nvidia.com`.

## Current scope

The source catalog contains the 15 general chat/text free-endpoint candidates selected for this relay, in coding-preference order. Only one model is enabled while the hosted transport and model behavior are being verified:

```text
nvidia/openai/gpt-oss-20b
```

The remaining models stay in the NVIDIA source catalog with `enabled: false`. Their upstream IDs, context limits, request compatibility and local token counts must still be validated against NVIDIA before enabling them.

NVIDIA does not currently have a hard-coded RPM, TPM, RPD, credit, or concurrency ceiling in the relay. A normal HTTP 429 is treated as model-scoped and triggers exponential model backoff. `Retry-After` is honored when present. Network failures, timeouts and 5xx responses are provider-scoped health failures. The relay does not infer an account-wide rate limit merely because several models are throttled.

Overflow offers are disabled until NVIDIA-specific overflow behavior is actually observed and justified.

## Token counting

The hosted NVIDIA service at `integrate.api.nvidia.com` was tested with both self-hosted NIM/vLLM helper endpoints:

```text
POST /v1/chat/completions/render
POST /tokenize
```

Both returned HTTP 404, so hosted token counting is not used.

Token counting is therefore local and provider-owned. The provider contract intentionally forbids reusing rendered-request token counts or count caches across providers merely because they expose the same underlying model: provider-specific chat templates, tool serialization, injected instructions or special tokens can change the actual input.

Each NVIDIA catalog model carries its own tokenizer specification:

| Relay model | Local tokenizer source |
| --- | --- |
| `nvidia/deepseek-ai/deepseek-v4.1-flash` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| `nvidia/z-ai/glm-5-3` | `zai-org/GLM-5.3` |
| `nvidia/moonshotai/kimi-k3` | `Xenova/Kimi-K3-tokenizer` |
| `nvidia/z-ai/glm-5-3-flash` | `zai-org/GLM-5.3-Flash` |
| `nvidia/nvidia/nemotron-3-ultra-550b-a55b` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` |
| `nvidia/meta/muse-glimmer-30b` | `meta-models/Muse-Glimmer-30B` |
| `nvidia/poolside/laguna-xs-2.1` | `poolside/Laguna-XS-2.1` |
| `nvidia/google/gemma-4-31b-it` | `nvidia/Gemma-4-31B-IT-NVFP4` |
| `nvidia/nvidia/nemotron-3-super-120b-a12b` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` |
| `nvidia/google/diffusiongemma-26b-a4b-it` | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` |
| `nvidia/nvidia/nemotron-3.5-lightning-30b-a3b` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| `nvidia/openai/gpt-oss-20b` | local `gpt-tokenizer/model/gpt-oss-20b` |
| `nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| `nvidia/meta/llama-3.2-90b-vision-instruct` | public Llama 3.2 Vision tokenizer mirror `alpindale/Llama-3.2-90B-Vision-Instruct` |
| `nvidia/meta/llama-3.2-11b-vision-instruct` | public Llama 3.2 Vision tokenizer mirror `alpindale/Llama-3.2-11B-Vision-Instruct` |

GPT-OSS uses the lightweight local `gpt-tokenizer` implementation and does not need a network request during offer evaluation.

The other tokenizer implementations use `@huggingface/transformers` and `AutoTokenizer.apply_chat_template()` with the request's messages, tools, `chat_template_kwargs`, and an assistant generation prompt. Transformers.js itself is dynamically imported only when an HF-backed model is evaluated. Tokenizer instances are lazy-loaded and cached per tokenizer repository. If a tokenizer does not expose an embedded template, the provider loads that repository's `chat_template.jinja` or `chat_template.json` once and installs it on the tokenizer. Because the other models are disabled, none of these assets are downloaded during normal operation yet. Enabling a model requires validating that this locally rendered count matches the NVIDIA deployment for representative plain-chat, tool-use, template-option and long-context requests.

Kimi K3 uses the Hugging Face staff-maintained standalone tokenizer repository because the original Kimi tokenizer historically required custom Python tokenizer code. The standalone repository supplies a normal tokenizer JSON and chat template usable from JavaScript.

The Llama Vision models use the public `alpindale` mirrors because the official Meta repositories can require Hugging Face access approval. NVIDIA documents these VLMs as supporting text-only queries, but the mirror templates must still be checked against NVIDIA's hosted rendering before either model is enabled.

## Rate-limit diagnostics

An NVIDIA 429 response is logged as:

```text
nvidia_rate_limit_response
nvidia_model_rate_limit
```

The log includes the upstream error body and `Retry-After` value when present so NVIDIA-specific behavior can be refined from observed responses rather than guessed quotas.

## Configuration

Set:

```text
NVIDIA_API_KEY
```

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.
