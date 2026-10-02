# AI Relay

AI Relay is a small OpenAI-compatible Chat Completions relay for a trusted local network. It listens on all interfaces by default, asks each configured provider for its best current offer, and keeps requests queued while capacity is temporarily unavailable.

Provider/model knowledge lives in TypeScript provider modules. Machine-specific configuration is limited to server settings and provider API keys.

## Requirements and installation

Requirements:

- Linux with systemd
- Node.js 24.12+

Add your provider keys and install the service:

```bash
cp .env.example .env
# Edit .env and configure the providers you want to use.
sudo npm run install-service
```

The service installer installs the production dependencies it needs under `/opt/ai-relay`; a preliminary `npm install` in the repository is not required for production installation.

`install-service` creates the `ai-relay` system user when needed, copies the application to `/opt/ai-relay`, installs its production npm dependencies there, installs/enables the systemd unit, copies `.env` to `/etc/ai-relay.env`, restarts the service, and checks `/health`. A separate manual `npm install` in `/opt/ai-relay` is not required.

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

> **Provider documentation:** Do not add provider-specific setup, models, quotas, transport details, or implementation status to this README. Keep that information in `src/providers/<provider>/README.md`.

## Offer routing

For `"model": "auto"`, each provider counts the request with the concrete candidate model's own tokenizer or authoritative provider-native counting mechanism and returns at most one offer for the requested offer kind. The relay does not use a provider-independent token estimate.

```text
kind + provider + model + exact input tokens + effective input capacity + available-at time
```

Token counting stays inside provider offer evaluation and may be asynchronous, so a provider can use either a fast local tokenizer or an official counting API. The relay only sees the resulting offer (or a structured no-offer reason). The selected offer carries the authoritative input token count, and execution reuses that same value for quota accounting.

The scheduler asks for `standard` offers first. Providers may optionally expose an `overflow` offer as a speculative last resort when their own tracked quota says a request should wait.

Effective single-request capacity is the smaller of the model context limit and its configured TPM limit. This prevents a request from being routed to a model whose context is large enough but whose per-minute quota can never admit that request.

Within a provider, the smallest-capacity model that can handle the request is preferred, provided it becomes available inside the current optimization wait. Across providers the same rule is applied. Provider priority is only a tie-breaker for otherwise equivalent offers.

The initial optimization wait is 15 seconds. Every failed execution halves it:

```text
15s -> 7.5s -> 3.75s -> 1.875s -> ...
```

If no standard offer is usable inside the cutoff, the scheduler checks for immediate overflow offers. If none is available, it uses the earliest standard offer and keeps the request queued rather than failing.

## Failure handling and queue fairness

Providers maintain their own health/cooldown state. A network failure or upstream `5xx` temporarily suppresses that provider; `429` cools down the affected model with `Retry-After` support and exponential backoff. Retryable failures are scoped to a provider or model. The scheduler prefers another eligible path after a failure and enforces a finite per-request budget of three retryable failures for the same path, so a permanently broken route cannot keep one request alive forever.

After a failed execution the request keeps its original queue age, but sets a one-shot yield flag. If a younger request is runnable, exactly one younger dispatch may pass before the failed request becomes eligible again. This avoids both extremes: sending a failed request to the back of the queue, or allowing one unstable request to monopolize all dispatches.

The existing work-conserving queue behavior remains: blocked requests may be bypassed so usable quota is not wasted, with a starvation barrier after repeated bypasses.

## API

AI Relay implements:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
GET  /observability
GET  /observability/stats
GET  /observability/logs
```

Example:

```bash
curl http://<server-lan-ip>:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Reply with exactly: relay works"}]
  }'
```

For streaming, add `"stream": true`.

There is no relay authentication. The service binds to `0.0.0.0`, so any machine that can reach the server on the LAN can use `http://<server-lan-ip>:8787`. LAN exposure is intentional; restrict access using your network configuration if necessary. An `Authorization` header from a client is ignored.

### Observability

Open `http://<server-lan-ip>:8787/observability` for a dependency-free dashboard that refreshes every five seconds. Its counters cover only the lifetime of the current server process and reset on restart. They distinguish completed client requests, terminal request failures, client cancellations, and individual upstream attempts that failed before failover. Token totals use each selected provider offer's authoritative input count and provider-reported output/total usage when available. Provider and concrete-model rows include zero-count registered routes and their live configuration, activity, and cooldown state. The requested-model table distinguishes `auto` from every explicitly requested model.

The **Download logs** button downloads `/observability/logs`, a JSONL file containing exactly the structured log events emitted since this process started. It observes the configured `LOG_LEVEL` and the same redaction as console/journald logging; it does not query journald or retain an unlimited in-memory history.

## Configuration

`config/relay.json` contains server settings only:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8787,
    "heartbeatSeconds": 15,
    "upstreamTimeoutSeconds": 300,
    "bodyLimitBytes": 10485760
  }
}
```

Provider credentials are configured through environment variables. See the README inside each `src/providers/<provider>/` directory for that provider's required variables and current status.

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.

## Logs

```bash
journalctl -u ai-relay -f
journalctl -u ai-relay --since '1 hour ago'
systemctl status ai-relay
```

Useful scheduler fields include `failure_count`, `optimization_wait_ms`, `queue_bypasses`, provider/model selection, and total request time.

Provider offer evaluation is logged structurally. At the default `LOG_LEVEL=info`, dispatches, upstream responses, failures, cooldown transitions, recovery, and confirmed/expired overflow boundaries are visible without logging every candidate evaluation. Google emits `google_history_recovery` when it has to bootstrap Antigravity from a transcript or replay external tool history through GenerateContent. Model/provider rejections include a bounded `detail` field derived from the upstream error body; structured `error.code`, `error.status`, and `error.message` are preferred, common credential patterns are redacted, and fallback text is capped at 1,000 characters. Set `LOG_LEVEL=debug` to also see `provider_offer` and `provider_no_offer` events for every provider offer pass, including offer kind, model, authoritative input tokens, effective capacity, availability, and structured no-offer reasons.

## Development

```bash
npm install
npm run check
```

Runtime dependencies are pinned in `package.json`. Production runs the TypeScript source directly using Node's built-in type stripping.
