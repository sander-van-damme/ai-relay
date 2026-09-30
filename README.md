# AI Relay

AI Relay is a small local OpenAI-compatible Chat Completions relay. It exposes one endpoint on `127.0.0.1`, routes requests over a fixed list of configured models, and queues work when a model or provider is temporarily out of quota.

The model list and quota rules live in the repository. A machine only needs its own provider API keys.

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

`install-service` creates the `ai-relay` system user when needed, copies the application to `/opt/ai-relay`, installs/enables the systemd unit, copies `.env` to `/etc/ai-relay.env`, restarts the service, and checks the local `/health` endpoint.

On later updates, pull the repository and run the same service installer again:

```bash
git pull
sudo npm run install-service
```

If no local `.env` exists, an empty `/etc/ai-relay.env` is created on first install. You can edit that file directly and restart the service with `sudo systemctl restart ai-relay`.

## Configuration

`config/models.json` is the checked-in application configuration. It contains the server settings, providers, models, routing order, and quota limits. Secrets do **not** belong in this file.

Provider credentials are read from the environment:

```text
GEMINI_API_KEY
NVIDIA_API_KEY
```

For local development, `npm start` automatically reads a local `.env` file when present. The systemd service reads `/etc/ai-relay.env`.

### Automatic model routing

Use `"model": "auto"` to let the relay choose a model.

The order of enabled entries in the `models` array is one **global preference order across all providers**. For example, Google → NVIDIA → Google is a valid order. There is no hidden ranking or provider-specific sub-order.

An explicit relay model ID bypasses automatic model selection, for example:

```text
google/gemini-3.8-flash
nvidia/openai/gpt-oss-120b
```

### Quotas

Quota policies can be configured at two levels:

- **provider level**: shared by every model using that provider;
- **model level**: applied only to that individual model.

Both policies must have capacity before a request can start. Supported fields are:

```json
{
  "maxConcurrent": null,
  "limits": {
    "requestsPerMinute": null,
    "inputTokensPerMinute": null,
    "requestsPerDay": null,
    "minimumSpacingMs": 0
  }
}
```

`null` means that AI Relay does not enforce that dimension. The checked-in values intentionally do not guess provider/account limits; set them to the limits that apply to your account.

`inputTokensPerMinute` uses a conservative local estimate (`JSON bytes / 4`) before dispatch. It is a pacing estimate, not a provider tokenizer, so leave headroom when exact accounting matters.

The daily limiter is a rolling 24-hour window. Quota state is kept in memory and resets when the service restarts.

## Queue behavior

Requests enter one queue. The oldest runnable request is dispatched first, but a temporarily blocked large request does not waste capacity: younger requests that still fit the available quota may pass it.

To prevent starvation, a blocked request may be bypassed by at most eight younger requests. After that it becomes a temporary queue barrier until it can run. This allows a small request to use remaining minute budget while a larger request waits for enough tokens to expire from the rolling window.

A request can wait because of model-level or provider-level concurrency, RPM, TPM, daily quota, minimum spacing, an upstream `Retry-After`, or a transient upstream failure. Upstream HTTP `429` cooldowns are applied conservatively to the provider as well as the selected model.

Streaming and non-streaming connections receive heartbeat bytes while queued so clients do not time out while waiting.

## API

AI Relay implements the OpenAI-compatible Chat Completions shape:

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

There is no relay authentication. The service binds to `127.0.0.1` and is intended only for processes on the same machine. If a client library sends an `Authorization: Bearer ...` header, AI Relay simply ignores it; any value or no header at all is accepted.

## Logs

The service writes structured JSON to stdout/stderr and systemd stores it in the journal.

Follow logs live:

```bash
journalctl -u ai-relay -f
```

Inspect recent logs:

```bash
journalctl -u ai-relay --since '1 hour ago'
```

Service status and recent failures are also visible with:

```bash
systemctl status ai-relay
```

## Development

Run locally:

```bash
npm install
cp .env.example .env
# Add the provider keys you want to use.
npm start
```

Run checks:

```bash
npm run check
```

The runtime has no npm dependencies. Production runs the TypeScript source directly using Node's built-in type stripping.
