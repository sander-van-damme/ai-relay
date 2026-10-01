# Google provider

Google is implemented directly on the official `@google/genai` SDK and the native Interactions API via `interactions.create()`, including Gemini Robotics ER 2 Preview. The provider does not use Google's OpenAI-compatible Chat Completions endpoint.

The checked-in catalog mirrors the non-zero AI Studio quotas supplied for this relay project on 2026-10-01. It includes only endpoints that accept text and produce text: Gemini Flash/Flash-Lite text models, Gemma 4 26B/31B, and Gemini Robotics ER 2 Preview. TTS, Live/audio, image-generation, embedding, video/music, and agent endpoints are intentionally excluded. Model-specific request capabilities such as supported thinking levels are also treated as hard routing constraints.

Google quota accounting uses the AI Studio RPM, TPM and RPD limits. RPD resets at midnight in `America/Los_Angeles`. Speculative overflow is enabled only for RPD because the observed project usage can exceed that displayed daily limit. RPM, TPM, context capacity and provider/model health remain hard constraints. If an overflow attempt receives a Google `429` that explicitly identifies the daily/RPD quota, overflow for that model is suppressed for exactly 24 hours. Generic `429` responses, unrelated network errors, and `5xx` responses do not create that observation.

Interactions are stored so follow-up requests can use `previous_interaction_id`, preserving Google-native multi-turn/tool state. Continuation state is bounded to one hour and reused only on the same upstream model.

## Configuration

Set the Google API key in the environment:

```text
GEMINI_API_KEY
```

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.

For the provider implementation contract, see `../README.md`.
