# Shared provider code

Files in this directory must contain only provider-neutral code that is genuinely shared across providers or clearly reusable across multiple providers.

## Rules

- Do not add provider-specific behavior to this directory.
- Do not add branches or special cases for a provider, model, endpoint, SDK, request format, response format, error shape, quota rule, or transport detail.
- Do not change shared code to solve a problem that is specific to one provider. Fix that problem inside `src/providers/<provider>/` instead.
- Only move or add code here when its semantics are provider-independent and sharing it does not couple providers together.
- Prefer some duplication inside provider implementations over a shared abstraction that may cause changes for one provider to affect others.
- Shared code must not inspect provider names, provider IDs, or model IDs to decide behavior.
- If you are unsure whether something belongs here, keep it in the provider-specific implementation.

`types.ts` is the provider-neutral contract. `quota.ts` contains reusable quota mechanics where those mechanics match a provider's semantics. Providers remain responsible for their own behavior and may implement provider-specific logic themselves when shared mechanics do not fit.
