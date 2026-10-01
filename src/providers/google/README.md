# Google provider

Google execution is implemented on the official `@google/genai` SDK and the native Interactions API via `interactions.create()`. The provider does not use Google's OpenAI-compatible Chat Completions endpoint. Authoritative token counting normally uses the SDK's `models.countTokens()`; when the request includes count-relevant configuration such as system instructions, tools, or response schema, the provider calls the Gemini Developer API `:countTokens` REST endpoint with a full nested `generateContentRequest`. This is required because the pinned SDK's `CountTokensConfig` path does not support that configuration in Gemini Developer API mode.

The checked-in catalog starts from the non-zero AI Studio quotas supplied for this relay project on 2026-10-01 and excludes routes that the Developer API subsequently reports as unavailable. In particular, both `gemini-2.5-flash` and `gemini-2.5-flash-lite` are omitted after the API returned 404 responses stating that they are no longer available to new users. The catalog otherwise includes text-capable Gemini Flash/Flash-Lite models, Gemma 4 26B/31B, Gemini Robotics ER 2 Preview, and the Antigravity managed agent. TTS, Live/audio, image-generation, embedding, video/music, zero-quota routes, and agents that do not fit the relay's chat-completions contract are excluded. Model-specific request capabilities such as supported thinking levels are treated as hard routing constraints.

Each Google route has an explicit `preference` value. Capacity and availability remain the primary routing criteria. Preference is consulted only when candidates have the same effective input capacity and the same availability, with the larger value preferred. Values are deliberately spaced in increments of 100 so they can be adjusted later based on observed coding/task quality without changing the routing algorithm or pretending that release date alone determines quality.

## Tool-history recovery

Provider-native continuation state is preferred whenever the incoming OpenAI message prefix matches a stored Google continuation.

When that state is unavailable, the relay recovers without treating provider-native state as authoritative:

- Antigravity cannot accept reconstructed function-call history in stateless mode. If the OpenAI history already contains tool calls/results but no safe Antigravity continuation exists, the relay starts a fresh Antigravity interaction with the supplied conversation flattened into a plain-text historical transcript. Historical tool activity is labeled as already completed context, while the caller's current function tools are still supplied normally. The resulting interaction ID is stored so later turns resume statefully with `previous_interaction_id`.
- Gemini model routes use GenerateContent for reconstructed OpenAI tool history. External function-call parts receive Google's documented `skip_thought_signature_validator` migration signature, allowing a trace from another model/API to be replayed. Once Gemini responds, the relay preserves Google's native model content and thought signatures and uses that native GenerateContent history on later turns.
- A stored continuation is ignored if the caller has introduced a new assistant tool-call trace after that continuation boundary; that trace is treated as external history and recovered instead of being injected into the old native chain.

Token counting follows the same request plan as execution, so transcript bootstraps and replay-safe GenerateContent requests are counted in the shape actually used for recovery.

## Antigravity

`google/antigravity-preview-09-2026` is exposed through the same relay `/v1/chat/completions` API as the model-backed routes. The provider invokes `agent: "antigravity-preview-09-2026"` and pins the agent's underlying reasoning model to `gemini-3.8-flash`.

Antigravity's managed-agent API requires an environment on every interaction. For a fresh interaction the relay supplies `environment: "remote"`, which asks Google to provision the required provider-owned sandbox. The returned `environment_id` is stored with the interaction ID and reused on stateful continuations. The relay does not mount caller files, repositories, credentials, or other sources into that environment.

The relay also replaces Antigravity's default tool set with an explicit list containing:

- Google Search, for public-information grounding;
- any function tools supplied by the incoming Chat Completions request.

Because the tool list is explicit, Antigravity does not receive its default `code_execution` or URL Context tools. Caller-provided functions remain caller-owned: Google may request them, but the relay never executes them itself. The provider sandbox is not treated as the caller's authoritative workspace; the provider-owned system instruction continues to direct the agent to use caller-supplied tools for caller-environment actions.

A short provider-owned system instruction only establishes that Antigravity is serving a chat-completions interface, that caller-provided functions operate on the caller's authoritative external environment, and that no filesystem/shell/runtime should be assumed beyond explicitly supplied tools. Client system/developer instructions are appended unchanged after that prefix.

Antigravity's AI Studio quota is configured as 60 RPM, 100K input TPM, and 100 RPD. Its documented 1,048,576-token input context is therefore capped to an effective single-request capacity of 100K by TPM. Authoritative request counting uses Google's `:countTokens` endpoint with a full `generateContentRequest` for the underlying `gemini-3.8-flash` model, including the relay prefix, Google Search tool declaration, and caller function definitions. When Chat Completions supplies `max_completion_tokens` (or legacy `max_tokens`), the relay maps it to Antigravity's total agent budget as `offer.inputTokens + max completion tokens`, so the caller's completion allowance is added on top of the authoritative initial input count. If no completion cap is supplied, `agent_config.max_total_tokens` is omitted. Other generation controls that the agent API cannot honor make the route ineligible rather than being silently ignored.

## Quotas and continuation

Google quota accounting uses the AI Studio RPM, TPM and RPD limits. RPD resets at midnight in `America/Los_Angeles`. Speculative overflow is enabled only for RPD because observed project usage can exceed that displayed daily limit. RPM, TPM, context capacity and provider/model health remain hard constraints. If an overflow attempt receives a Google `429` that explicitly identifies the daily/RPD quota, overflow for that route is suppressed only until the next Pacific midnight, matching the same calendar-day boundary used by local RPD accounting. The learned overflow block is purged on the next provider evaluation/status read after that boundary. Generic `429` responses, unrelated network errors, and `5xx` responses do not create that observation.

Interactions are stored so follow-up requests can use `previous_interaction_id`, preserving Google-native multi-turn/tool state. Antigravity continuations also store and resend the associated `environment_id`, because Google's managed-agent API requires both conversation state and environment state on chained interactions. Continuation keys distinguish model interactions from agent interactions even when Antigravity uses Gemini 3.8 Flash underneath. Continuation state is bounded to one hour and reused only on the same upstream target.

## Configuration

Set the Google API key in the environment:

```text
GEMINI_API_KEY
```

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.

For the provider implementation contract, see `../README.md`.
