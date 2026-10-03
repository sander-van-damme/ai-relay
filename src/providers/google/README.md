# Google provider

Google execution is implemented on the official `@google/genai` SDK and the native Interactions API via `interactions.create()`. Authoritative token counting normally uses the SDK's `models.countTokens()`; when the request includes count-relevant configuration such as system instructions, tools, or response schema, the provider calls the Gemini Developer API `:countTokens` REST endpoint with a full nested `generateContentRequest`. This is required because the pinned SDK's `CountTokensConfig` path does not support that configuration in Gemini Developer API mode.

The checked-in catalog starts from the non-zero AI Studio quotas supplied for this relay project on 2026-10-01 and excludes routes that the Developer API subsequently reports as unavailable. In particular, both `gemini-2.5-flash` and `gemini-2.5-flash-lite` are omitted after the API returned 404 responses stating that they are no longer available to new users. The active catalog otherwise includes text-capable Gemini Flash/Flash-Lite models, Gemma 4 26B/31B, and Gemini Robotics ER 2 Preview. The Antigravity implementation is retained but its catalog entry is temporarily commented out while the new emulation path is validated. TTS, Live/audio, image-generation, embedding, video/music, zero-quota routes, and agents that do not fit the relay's chat-completions contract are excluded. Model-specific request capabilities such as supported thinking levels are treated as hard routing constraints.

Each Google route has an explicit `preference` value. Capacity and availability remain the primary routing criteria. Preference is consulted only when candidates have the same effective input capacity and the same availability, with the larger value preferred. Values are deliberately spaced in increments of 100 so they can be adjusted later based on observed coding/task quality without changing the routing algorithm or pretending that release date alone determines quality.

## Tool-history recovery

Provider-native continuation state is preferred whenever the incoming OpenAI message prefix matches a stored Google continuation.

When that state is unavailable, the relay recovers without treating provider-native state as authoritative:

- Gemini model routes use GenerateContent for reconstructed OpenAI tool history. External function-call parts receive Google's documented `skip_thought_signature_validator` migration signature, allowing a trace from another model/API to be replayed. Once Gemini responds, the relay preserves Google's native model content and thought signatures and uses that native GenerateContent history on later turns.
- A stored continuation is ignored if the caller has introduced a new assistant tool-call trace after that continuation boundary; that trace is treated as external history and recovered instead of being injected into the old native chain.

Token counting follows the same request plan as execution, so transcript bootstraps and replay-safe GenerateContent requests are counted in the shape actually used for recovery.

## Antigravity

The Antigravity implementation uses a deliberately different compatibility strategy from the model-backed Google routes. Antigravity is a managed agent with its own sandbox and native tool loop, so the relay does not register caller-provided Chat Completions tools as Antigravity tools and does not try to make the Google sandbox impersonate the caller's environment.

Instead, each outer Chat Completions request is handled as a one-shot manual emulation task:

1. The relay provisions a fresh Antigravity environment with `environment: "remote"`.
2. It omits the `tools` field entirely, leaving Antigravity's native/default capabilities intact.
3. It serializes the complete raw Chat Completions request into a provider-owned prompt and tells Antigravity that it is manually acting as the response-producing side of an OpenAI-compatible Chat Completions endpoint.
4. Tool definitions inside that raw JSON remain data describing the external caller's runtime. If the correct outer response is a tool call, Antigravity writes that tool call into the manual Chat Completions response instead of executing the outer tool itself.
5. Antigravity emits the semantic response inside `<manual_openai_chat_completion_response>...</manual_openai_chat_completion_response>`.
6. The relay extracts and validates the JSON. Transport metadata such as `id`, `object`, `created`, `model`, and `usage` is relay-owned; any versions supplied by Antigravity are ignored and replaced.
7. If validation fails, the relay may continue the same temporary interaction for up to two correction attempts, supplying the exact validator error. This continuation exists only inside the processing of that single outer request. If the agent still cannot produce a valid completion, the route is treated as temporarily unavailable: the relay records a model-level retryable failure and applies the same exponential model cooldown used for other temporary model failures.
8. After success or failure, the relay makes a best-effort attempt to delete the temporary Google environment. Cleanup failure is logged but never changes an otherwise valid caller response.

Across outer Chat Completions requests, Antigravity is fully stateless. The relay never reuses an Antigravity `previous_interaction_id` or `environment_id`; the next request already carries its own complete conversation history.

Streaming callers are also handled as one complete Antigravity emulation turn. Internal Antigravity progress and native tool activity are not forwarded. After the final manual response validates, the relay emits that completed semantic response as OpenAI-compatible SSE chunks.

The Antigravity catalog entry remains disabled while this path is being validated. The underlying implementation and tests stay in place so it can be re-enabled without reconstructing the integration.


## Quotas and continuation

Google quota accounting uses the AI Studio RPM, TPM and RPD limits. RPD resets at midnight in `America/Los_Angeles`. Speculative overflow is enabled only for RPD because observed project usage can exceed that displayed daily limit. RPM, TPM, context capacity and provider/model health remain hard constraints. If an overflow attempt receives a Google `429` that explicitly identifies the daily/RPD quota, overflow for that route is suppressed only until the next Pacific midnight, matching the same calendar-day boundary used by local RPD accounting. The learned overflow block is purged on the next provider evaluation/status read after that boundary. Generic `429` responses, unrelated network errors, and `5xx` responses do not create that observation.

Model-backed Google Interactions are stored so compatible follow-up requests can use `previous_interaction_id` and preserve Google-native state. GenerateContent continuations likewise preserve native model content and thought signatures. This continuation cache is not used for Antigravity: Antigravity is intentionally one-shot and stateless across outer Chat Completions requests.

## Configuration

Set the Google API key in the environment:

```text
GEMINI_API_KEY
```

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.

For the provider implementation contract, see `../README.md`.
