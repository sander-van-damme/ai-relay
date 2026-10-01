# NVIDIA provider

The NVIDIA implementation is retained as unfinished scaffolding but is intentionally disabled. Its `getBestOffer()` currently always returns a structured `no_offer` result with reason `no_eligible_model`, so the scheduler cannot select NVIDIA even when `NVIDIA_API_KEY` is configured.

Before enabling it, the NVIDIA provider still needs to follow the full contract in `../README.md`: evaluate the official SDK/native APIs versus the OpenAI-compatible endpoint, expand and verify the model catalog and capabilities, implement verified request/token/daily/concurrency limits, validate authoritative token counting and capacities, implement NVIDIA-specific failure/cooldown/overflow behavior, and add the required provider tests.

## Configuration

The reserved environment variable is:

```text
NVIDIA_API_KEY
```

It is currently unused for routing because the NVIDIA provider is intentionally disabled pending completion.

For local development, `npm start` reads `.env` when present. The systemd service reads `/etc/ai-relay.env`.
