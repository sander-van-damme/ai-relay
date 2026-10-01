# Legacy provider inventory

This file preserves the provider/API-key inventory from the previous `cloudflare-ai-relay` project. These providers are **not active** in the new relay unless support is deliberately added later.

| Provider | Legacy environment variable(s) | Notes |
| --- | --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` | OpenAI-compatible gateway |
| OpenCode Inference | `OPENCODE_API_KEY` | Previous relay only used `*-free` models |
| Groq | `GROQ_API_KEY` | OpenAI-compatible API |
| Cerebras | `CEREBRAS_API_KEY` | OpenAI-compatible API |
| Chutes | `CHUTES_API_KEY` | OpenAI-compatible API |
| Vercel AI Gateway | `VERCEL_API_KEY` | OpenAI-compatible gateway |
| Mistral Studio | `MISTRAL_API_KEY` | OpenAI-compatible API |
| Arli AI | `ARLIAI_API_KEY` | Legacy provider |
| NVIDIA NIM / API Catalog | `NVIDIA_API_KEY` | Implementation retained but intentionally disabled pending provider-contract work |
| Kilo AI Gateway | `KILO_API_KEY` | Legacy provider |
| Kenari | `KENARI_API_KEY` | Legacy provider |
| LLM7 | `LLM7_API_KEY` | Legacy provider |
| Ollama Cloud | `OLLAMA_API_KEY` | Legacy provider |
| Hugging Face | `HUGGINGFACE_API_KEY` | Inference Providers router |
| AION Labs | `AION_API_KEY` | Legacy provider |
| Cohere | `COHERE_API_KEY` | Compatibility API |
| Google Gemini | `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY` | Active in the new relay |
| Cloudflare Workers AI REST | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_KEY` | Deliberately not migrated |
| OVHcloud AI Endpoints | none | Previously used anonymous/free tier |
| Artificial Analysis | `ARTIFICIAL_ANALYSIS_API_KEY` | Benchmark ranking was deliberately removed |

The old relay also used `MODEL_DENYLIST` and Cloudflare-specific KV/Workers bindings. Those concepts are intentionally absent from the new design.
