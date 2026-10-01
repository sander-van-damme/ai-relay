# AI Relay

AI Relay is a small local OpenAI-compatible Chat Completions relay. It exposes one endpoint on `127.0.0.1`, asks each configured provider for its best current offer, and keeps requests queued while capacity is temporarily unavailable.

Provider/model knowledge lives in TypeScript provider modules. Machine-specific configuration is limited to server settings and provider API keys.

## Requirements and installation

Requirements:

- Linux with systemd
- Node.js 24.12+

Install dependencies, add your provider keys, and install the service:

```bash
npm install
cp .env.example .env
# Edit .env and set GEMINI_API_KEY and/or NVIDIA_API_KEY.
sudo npm run install-service
```

`install-service` creates the `ai-relay` system user when needed, copies the application to `/opt/ai-relay`, installs/enables the systemd unit, copies `.env` to `/etc/ai-relay.env`, restarts the service, and checks `/health`.

On later updates:

```bash
git pull
sudo npm run install-service
```

## Architecture

The relay core does not know provider transports or provider-specific quota semantics. A provider owns:

- its model catalog and effective request capacities;
- RPM, TPM and RPD accounting;
- daily reset semantics;
- provider/model cooldowns after upstream failures;
- the technical implementation of the upstream request;
- selection of one best offer for a relay request.

The scheduler only compares one offer per provider, waits when that buys a smaller-capacity model within the optimization cutoff, and fails over when an execution attempt fails.

Concrete provider implementations live under `src/providers/<provider>/`. Reusable implementation helpers live under `src/providers/shared/`; using them is optional. A provider may use an official SDK, an OpenAI-compatible transport, or its own implementation without changing the scheduler. See `src/providers/README.md` for the provider implementation contract. The checked-in `config/relay.json` contains only server settings. Provider retry/quota rules stay with the provider and secrets stay in the environment.

## Offer routing

For `"model": "auto"`, each provider counts the request with the concrete candidate model's own tokenizer or authoritative provider-native counting mechanism and returns at most one offer for the requested offer kind. The relay does not use a provider-independent token estimate.

```text
kind + provider + model + exact input tokens + effective input capacity + available-at time
```

Token counting is asynchronous at the provider boundary so a provider can use either a fast local tokenizer or an official counting API. The selected offer carries the authoritative input token count, and execution reuses that same value for quota accounting.

The scheduler asks for `standard` offers first. Providers may optionally expose an `overflow` offer as a speculative last resort when their own tracked quota says a request should wait.

Effective single-request capacity is the smaller of the model context limit and its configured TPM limit. This prevents a request from being routed to a model whose context is large enough but whose per-minute quota can never admit that request.

Within a provider, the smallest-capacity model that can handle the request is preferred, provided it becomes available inside the current optimization wait. Across providers the same rule is applied. Provider priority is only a tie-breaker; Google currently has priority over NVIDIA for otherwise equivalent offers.

The initial optimization wait is 15 seconds. Every failed execution halves it:

```text
15s -> 7.5s -> 3.75s -> 1.875s -> ...
```

If no standard offer is usable inside the cutoff, the scheduler checks for immediate overflow offers. If none is available, it uses the earliest standard offer and keeps the request queued rather than failing.

## Failure handling and queue fairness

Providers maintain their own health/cooldown state. A network failure or upstream `5xx` temporarily suppresses that provider; `429` cools down the affected model with `Retry-After` support and exponential backoff. Retryable failures are scoped to a provider or model. The scheduler prefers another eligible path after a failure and enforces a finite per-request budget of three retryable failures for the same path, so a permanently broken route cannot keep one request alive forever.

After a failed execution the request keeps its original queue age, but sets a one-shot yield flag. If a younger request is runnable, exactly one younger dispatch may pass before the failed request becomes eligible again. This avoids both extremes: sending a failed request to the back of the queue, or allowing one unstable request to monopolize all dispatches.

The existing work-conserving queue behavior remains: blocked requests may be bypassed so usable quota is not wasted, with a starvation barrier after repeated bypasses.

## Google provider

Google is implemented directly on the official `@google/genai` SDK and its Interactions API. The provider does not use Google's OpenAI-compatible Chat Completions endpoint.

The checked-in catalog mirrors the non-zero AI Studio quotas supplied for this relay project on 2026-10-01. It includes only endpoints that accept text and produce text: Gemini Flash/Flash-Lite text models, Gemma 4 26B/31B, and Gemini Robotics ER 2 Preview. TTS, Live/audio, image-generation, embedding, video/music, and agent endpoints are intentionally excluded.

Google quota accounting uses the AI Studio RPM, TPM and RPD limits. RPD resets at midnight in `America/Los_Angeles`. Speculative overflow is enabled only for RPD because the observed project usage can exceed that displayed daily limit. RPM, TPM, context capacity and provider/model health remain hard constraints. If an overflow attempt receives a Google `429`, overflow for that model is suppressed for exactly 24 hours; unrelated network errors and `5xx` responses do not create that observation.

Interactions are stored so follow-up requests can use `previous_interaction_id`, preserving Google-native multi-turn/tool state such as thought signatures. The relay keeps only a bounded one-hour in-memory mapping from an OpenAI-style conversation prefix to the matching Google interaction; continuation is reused only on the same upstream model.

## API

AI Relay implements:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

Example:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Reply with exactly: relay works"}]
  }'
```

For streaming, add `"stream": true`.

There is no relay authentication. The service binds to `127.0.0.1` and is intended only for processes on the same machine. An `Authorization` header from a client is ignored.

## Configuration

`config/relay.json` contains server settings only:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8787,
    "heartbeatSeconds": 15,
    "upstreamTimeoutSeconds": 300,
    "bodyLimitBytes": 10485760
  }
}
```

Provider credentials:

```text
GEMINI_API_KEY
NVIDIA_API_KEY
```

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.

## Logs

```bash
journalctl -u ai-relay -f
journalctl -u ai-relay --since '1 hour ago'
systemctl status ai-relay
```

Useful scheduler fields include `failure_count`, `optimization_wait_ms`, `queue_bypasses`, provider/model selection, and total request time.

## Development

```bash
npm install
npm run check
```

Runtime dependencies are pinned in `package.json`. Production runs the TypeScript source directly using Node's built-in type stripping.
