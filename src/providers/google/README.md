# Google provider

Google is implemented directly on the official `@google/genai` SDK and the native Interactions API via `interactions.create()`. The provider does not use Google's OpenAI-compatible Chat Completions endpoint.

The checked-in catalog mirrors the non-zero AI Studio quotas supplied for this relay project on 2026-10-01. It includes text-capable Gemini Flash/Flash-Lite models, Gemma 4 26B/31B, Gemini Robotics ER 2 Preview, and the Antigravity managed agent. TTS, Live/audio, image-generation, embedding, video/music, zero-quota routes, and agents that do not fit the relay's chat-completions contract are excluded. Model-specific request capabilities such as supported thinking levels are treated as hard routing constraints.

Each Google route has an explicit `preference` value. Capacity and availability remain the primary routing criteria. Preference is consulted only when candidates have the same effective input capacity and the same availability, with the larger value preferred. Values are deliberately spaced in increments of 100 so they can be adjusted later based on observed coding/task quality without changing the routing algorithm or pretending that release date alone determines quality.

## Antigravity

`google/antigravity-preview-09-2026` is exposed through the same relay `/v1/chat/completions` API as the model-backed routes. The provider invokes `agent: "antigravity-preview-09-2026"` and pins the agent's underlying reasoning model to `gemini-3.8-flash`.

The relay intentionally does not provide an Antigravity `environment`. It also replaces Antigravity's default tool set with an explicit list containing:

- Google Search, for public-information grounding;
- any function tools supplied by the incoming Chat Completions request.

Because the tool list is explicit, Antigravity does not receive its default `code_execution` or URL Context tools, and without an `environment` it does not receive Google's filesystem tools. Caller-provided functions remain caller-owned: Google may request them, but the relay never executes them itself. This keeps the relay generic and lets clients such as coding agents continue to own their actual workspace and tool execution.

A short provider-owned system instruction only establishes that Antigravity is serving a chat-completions interface, that caller-provided functions operate on the caller's authoritative external environment, and that no filesystem/shell/runtime should be assumed beyond explicitly supplied tools. Client system/developer instructions are appended unchanged after that prefix.

Antigravity's AI Studio quota is configured as 60 RPM, 100K input TPM, and 100 RPD. Its documented 1,048,576-token input context is therefore capped to an effective single-request capacity of 100K by TPM. Authoritative request counting uses Google's `countTokens` with the underlying `gemini-3.8-flash` model and includes the relay prefix plus caller function definitions. When Chat Completions supplies `max_completion_tokens` (or legacy `max_tokens`), the relay maps it to Antigravity's total agent budget as `offer.inputTokens + max completion tokens`, so the caller's completion allowance is added on top of the authoritative initial input count. If no completion cap is supplied, `agent_config.max_total_tokens` is omitted. Other generation controls that the agent API cannot honor make the route ineligible rather than being silently ignored.

## Quotas and continuation

Google quota accounting uses the AI Studio RPM, TPM and RPD limits. RPD resets at midnight in `America/Los_Angeles`. Speculative overflow is enabled only for RPD because observed project usage can exceed that displayed daily limit. RPM, TPM, context capacity and provider/model health remain hard constraints. If an overflow attempt receives a Google `429` that explicitly identifies the daily/RPD quota, overflow for that route is suppressed only until the next Pacific midnight, matching the same calendar-day boundary used by local RPD accounting. The learned overflow block is purged on the next provider evaluation/status read after that boundary. Generic `429` responses, unrelated network errors, and `5xx` responses do not create that observation.

Interactions are stored so follow-up requests can use `previous_interaction_id`, preserving Google-native multi-turn/tool state. Continuation keys distinguish model interactions from agent interactions even when Antigravity uses Gemini 3.8 Flash underneath. Continuation state is bounded to one hour and reused only on the same upstream target.

## Configuration

Set the Google API key in the environment:

```text
GEMINI_API_KEY
```

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.

For the provider implementation contract, see `../README.md`.
