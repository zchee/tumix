# ExecPlan: Provider-Specific LLM Parameters

- **Owner:** codex
- **Status:** draft
- **Problem:** `model.LLM` only accepts `*model.LLMRequest`, so Anthropic/OpenAI/xAI adapters cannot receive provider-only parameters (e.g., `anthropic.BetaMessageNewParams` fields, `openai.ResponseNewParams`/metadata, extra xAI chat options).
- **Goal:** Add a backward-compatible extension point so callers can attach provider-specific tunables to an `LLMRequest`, and have each adapter apply them when constructing SDK request params.
- **Non-goals:** Changing the ADK `model.LLM` interface, reworking message/parts conversion, or introducing new network dependencies.
- **Constraints:** Preserve existing defaults/behavior; keep API ergonomic and typed (avoid raw `map[string]any` where possible); no vendor edits.

## Approach
1) Define a typed carrier `ProviderParams` stored in `LLMRequest.Tools["gollm:provider-params"]`.
2) Provide per-provider option structs:
   - `AnthropicProviderParams` with `[]AnthropicParamMutator` (`func(*anthropic.MessageNewParams)`).
   - `OpenAIProviderParams` with `[]OpenAIParamMutator` (`func(*openai.ChatCompletionNewParams)`).
   - `XAIProviderParams` with `[]xai.ChatOption` to append post-normalization.
3) Helpers: `SetProviderParams(req, params)` and `providerParams(req)` to safely fetch/normalize values (accept struct or pointer).
4) Adapter wiring:
   - After building default params, apply mutators (skip nil) before calling SDK.
   - For xAI, append extra chat options before `Chat.Create`.
5) Tests:
   - Unit tests per provider verifying mutations are applied and base behavior unchanged when absent.
6) Docs: brief usage note in code comments.

## Steps
1) Add `provider_params.go` with carrier types, mutator aliases, helper setters/getters.
2) Wire `anthropicLLM.buildParams` to apply Anthropic mutators.
3) Wire `openAILLM.chatCompletionParams` to apply OpenAI mutators.
4) Wire xAI generate/generateStream to append `XAIProviderParams.Options`.
5) Add table-driven tests for each provider covering: no-op when absent, single mutator, multiple mutators preserving base fields.
6) gofmt + gofumpt; run unit tests for touched packages.

## Risks / Mitigations
- **Type assertion misuse:** Guard with ok checks and tolerate non-typed entries to avoid panics.
- **Behavior drift:** Apply mutators after base mapping to keep defaults unless overridden.
- **Future provider APIs:** Mutator pattern keeps forward-compatibility without new interface changes.

## Validation
- `./tools/bin/gotestsum -f standard-verbose -- -race -count=1 -shuffle=on -cover ./gollm/...`
- Spot-check integration via existing rr recordings remain unchanged when no provider params attached.
