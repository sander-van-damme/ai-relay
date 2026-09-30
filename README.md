# AI Relay

A small self-hosted OpenAI-compatible relay for a fixed list of LLMs.

This project is the deliberately simplified successor to `cloudflare-ai-relay`. It is designed to run as a normal long-lived Node.js service on a Linux server, with logs available through `journalctl` and no Cloudflare runtime dependencies.

## Design

The relay intentionally does **not** discover, benchmark, rank, probe, or cache models dynamically.

- models are listed manually in `config/models.json`;
- Google Gemini and NVIDIA are the only providers currently implemented;
- model order in the JSON file is the `auto` preference order;
- requests enter one global FIFO queue;
- the queue dispatches the oldest request first when one of its eligible models has quota and concurrency available;
- rate limits are enforced with rolling 60-second request/token windows and a rolling 24-hour request window;
- streaming and non-streaming requests both receive bytes immediately while queued;
- upstream `429`, `408`, and `5xx` responses are retried without losing the client's place in the queue;
- stdout/stderr use structured JSON, which systemd stores in the journal.

There is no runtime dependency on an AI SDK. Both providers expose OpenAI-compatible Chat Completions endpoints, so the relay uses the built-in Node.js `fetch` API directly.

## Requirements

- Node.js 24.12+ (runs the TypeScript source directly using Node's built-in type stripping)
- Linux/systemd for the supplied service file

Runtime dependencies: **none**. `typescript` and `@types/node` are development-only packages used for checks.

## Configuration

Copy/edit `config/models.json`. A model entry looks like:

```json
{
  "id": "google/gemini-3.8-flash",
  "provider": "google",
  "upstreamModel": "gemini-3.8-flash",
  "enabled": true,
  "maxConcurrent": 1,
  "limits": {
    "requestsPerMinute": 10,
    "inputTokensPerMinute": 250000,
    "requestsPerDay": 21500,
    "minimumSpacingMs": 0
  }
}
```

The numbers above are only an example of how to enter a quota. Set a limit to `null` when you do not want the relay to enforce that dimension. Quotas change by model/account/tier, so the checked-in file deliberately does **not** guess your account's limits. Put the values shown for your project/provider account into this file.

`inputTokensPerMinute` uses a conservative local estimate (`JSON bytes / 4`) before dispatch. It is useful for pacing but it is not a provider tokenizer. If exact provider accounting matters, leave headroom below the published limit.

The daily limiter is a rolling 24-hour window. That is slightly more conservative than providers that reset at a fixed clock time, but it is deterministic and avoids a burst immediately after a reset.

### Automatic routing

Use:

```json
{ "model": "auto" }
```

`auto` tries enabled models in the order they appear in `config/models.json`. There is no hidden quality score. If the first model is temporarily rate-limited, another configured model can take the request while preserving FIFO queue order.

Explicit routing uses the configured relay ID, for example:

```text
google/gemini-3.8-flash
nvidia/openai/gpt-oss-120b
```

## Secrets

```bash
cp .env.example /etc/ai-relay.env
sudo chmod 600 /etc/ai-relay.env
```

At least one relay bearer key is mandatory. The service accepts `RELAY_API_KEY` and numbered keys such as `RELAY_API_KEY_1`, which makes key rotation easy.

Provider secrets currently used:

```text
GEMINI_API_KEY
NVIDIA_API_KEY
```

The Gemini provider also accepts the legacy aliases `GOOGLE_API_KEY` and `GOOGLE_GENERATIVE_AI_API_KEY` if they are listed in the provider config.

Older provider/key names are preserved in [`docs/LEGACY_PROVIDERS.md`](docs/LEGACY_PROVIDERS.md).

## Run locally

```bash
npm install
export RELAY_API_KEY='local-secret'
export GEMINI_API_KEY='...'
export NVIDIA_API_KEY='...'
npm start
```

Then:

```bash
curl http://127.0.0.1:8787/v1/models \
  -H 'Authorization: Bearer local-secret'

curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Reply with exactly: relay works"}]
  }'
```

For streaming, add `"stream": true`.

## Queue behavior

The client connection starts immediately:

- SSE requests receive an SSE comment such as `: ai-relay queued`;
- non-streaming requests receive JSON whitespace, which is legal before the final JSON value.

The request then remains in the FIFO queue until an eligible model can start. Heartbeats keep the connection active while it waits.

A model can be delayed by:

- `maxConcurrent` active calls;
- `requestsPerMinute`;
- `inputTokensPerMinute`;
- `requestsPerDay`;
- `minimumSpacingMs`;
- a provider `Retry-After` value after a `429`;
- the configured retry delay after transient network/5xx failures.

Quota state is intentionally in memory. Restarting the service clears the local rolling windows, so avoid rapid restart loops when you are already close to a provider quota.

## API

### `GET /health`

Returns queue depth and per-model active/block state.

### `GET /v1/models`

Returns `auto` plus the enabled models from the JSON configuration.

### `POST /v1/chat/completions`

OpenAI-compatible Chat Completions proxy. The relay changes only the upstream `model` field; the rest of the request body is forwarded as-is.

All endpoints require:

```text
Authorization: Bearer <relay key>
```

## systemd

A hardened example unit is in `deploy/ai-relay.service`.

Suggested layout:

```text
/opt/ai-relay/            repository checkout
/etc/ai-relay.env         secrets
/etc/ai-relay/models.json model/rate-limit configuration
```

Install:

```bash
sudo useradd --system --home /opt/ai-relay --shell /usr/sbin/nologin ai-relay
sudo mkdir -p /opt/ai-relay
sudo chown -R ai-relay:ai-relay /opt/ai-relay

sudo cp deploy/ai-relay.service /etc/systemd/system/ai-relay.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-relay
```

Logs:

```bash
journalctl -u ai-relay -f
journalctl -u ai-relay --since '1 hour ago'
```

Restart after editing the model config:

```bash
sudo systemctl restart ai-relay
```

## Development

```bash
npm install
npm run check
```

The code is kept compatible with Node's erasable-TypeScript subset, so production does not need a compile/build step.
