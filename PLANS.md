# ExecPlan: Provider-Specific LLM Parameters (Anthropic Beta, OpenAI Responses, xAI)

This ExecPlan is a living document. Keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current. Follow the ExecPlan conventions in `~/.config/agent/instructions/ExecPlan.md`.

## Purpose / Big Picture

Extend `gollm` adapters so callers can attach provider-specific knobs to `model.LLMRequest` and have Anthropic (beta), OpenAI Responses, and xAI adapters apply them without breaking existing behavior (stop sequences, multiple candidates, tool outputs). A novice should be able to prove success by running the gollm test suite and exercising stop-sequence trimming and function-call output handling via unit tests.

## Progress

- [x] (2025-12-06 15:30Z) Added `ProviderParams` carrier and mutator/option types; stored on `LLMRequest.Tools["gollm:provider-params"]`.
- [x] (2025-12-06 15:45Z) Wired Anthropic, OpenAI, xAI adapters to read provider params; converted Anthropic path to beta APIs; migrated OpenAI path to Responses API.
- [x] (2025-12-06 16:10Z) Added provider param unit tests for all three adapters; recorded replays for Anthropic/xAI beta paths.
- [x] (2025-12-06 16:30Z) Test suite `go test ./gollm/...` green on current branch.
- [x] (2025-12-06 17:30Z) Fix Anthropic mutator application to allow full beta params without field loss.
- [x] (2025-12-06 17:45Z) Support OpenAI Responses: parse shell/function outputs, trim stop sequences, and emulate candidate_count via loop; ignore seed without failing.
- [x] (2025-12-06 17:50Z) Stream path uses stop trimming and updated aggregator signature.
- [x] (2025-12-06 17:55Z) Added unit tests for stop trimming, shell_call_output, and candidate_count loop.
- [x] (2025-12-06 17:58Z) Re-ran `go test ./gollm/...` successfully.
- [ ] (TODO) Update Outcomes & Decision Log; summarize lessons.

## Surprises & Discoveries

- OpenAI Responses API omits `stop`/`seed`/`candidate_count` knobs; naive migration hard-fails common callers. Evidence: current `responseParams` returns errors when these fields are set.
- Anthropic beta mutators were proxied through a limited copy, dropping any fields beyond the copied subset (e.g., ToolChoice, Safety settings).
- Responses output union uses `shell_call_output` rather than `function_call_output`; raw function outputs must be handled via the output union, not the conversation item type.

## Decision Log

- Decision: Keep Responses API as primary path but layer compatibility shims (stop trimming, multi-candidate loop, graceful seed ignore) rather than reverting to Chat Completions.
  Rationale: preserves future-proofing while reducing regressions for existing callers.
  Date/Author: 2025-12-06 / codex
- Decision: Handle function/tool outputs by parsing `shell_call_output` union and mapping to `FunctionResponse`, leaving other tool types to default conversion.
  Rationale: Minimal change to surface tool outputs without reintroducing legacy chat API.
  Date/Author: 2025-12-06 / codex

## Outcomes & Retrospective

- Anthropic mutators now operate in place, so any beta params (tool choice, metadata) can be set via ProviderParams.
- OpenAI Responses path now trims stop sequences, tolerates seeds, loops to deliver multiple candidates, and converts shell_call_output into `FunctionResponse`.
- New tests cover stop trimming, tool output conversion, and candidate_count loop; `go test ./gollm/...` passes.
- Remaining: monitor for additional Responses output variants (file/web search) if needed.

## Context and Orientation

- Repo root `/Users/zchee/go/src/github.com/zchee/tumix-worktrees/genai-convert`.
- Key files:
  - `gollm/provider_params.go` defines provider param carriers and mutator aliases.
  - `gollm/anthropic.go`, `gollm/internal/adapter/anthropic_*` for Anthropic beta path.
  - `gollm/openai.go`, `gollm/internal/adapter/openai_*` for OpenAI Responses path.
  - `gollm/xai.go` for xAI path; `gollm/xai/chat_options.go` uses invopop/jsonschema.
  - Tests in `gollm/*_test.go` and `gollm/internal/adapter/*_test.go`; replays under `gollm/testdata/`.
- Current state: provider params exist, but Anthropic mutators lose fields; OpenAI Responses lacks support for stop sequences, candidate_count>1, and function_call_output items; streaming paths don’t trim stops.

## Plan of Work

1) Fix Anthropic mutator plumbing:
   - Apply mutators directly to `anthropic.BetaMessageNewParams` without proxy copying so all fields can be set.
   - Ensure nil-safe and order-preserving.
2) OpenAI Responses compatibility:
   - Content conversion: handle `function_call_output` items by emitting `genai.FunctionResponse` with CallID and payload.
   - Stop sequences: implement trimming for non-stream and stream outputs; set FinishReason=Stop when trimmed.
   - CandidateCount: for non-stream requests, loop `count` times with identical params and yield multiple responses; keep stream path single-candidate with clear error/warning.
   - Seed: ignore gracefully (document in comments) rather than hard error.
3) Streaming aggregator:
   - Accept stop sequence list; trim final text and set FinishReason appropriately; tolerate tool-call outputs in final conversion.
4) Tests:
   - Add unit tests for function_call_output conversion, stop-sequence trimming, and candidate_count looping; adjust replays if needed.
5) Validation:
   - Run `go test ./gollm/...`.
   - Confirm new tests fail before fixes and pass after.

## Concrete Steps

1) Edit `gollm/anthropic.go`: simplify `applyAnthropicProviderParams` to mutate `*anthropic.BetaMessageNewParams` in place; remove proxy copy; keep nil guards.
2) Edit `gollm/internal/adapter/openai_response.go`:
   - Extend `OpenAIResponseToLLM` to accept stop sequences and convert `function_call_output` items to `genai.Part{FunctionResponse:...}`.
   - Post-process text parts to trim at earliest stop sequence; set FinishReason to Stop when trimming occurs.
   - Add helper to trim text and detect stop.
3) Edit `gollm/internal/adapter/openai_stream_test.go` and related tests: update for new aggregator constructor signature and new behaviors.
4) Edit `gollm/internal/adapter/openai_response.go` & `gollm/internal/adapter/openai_test.go`: adjust tests and helper inputs for stop trimming and function_call_output.
5) Edit `gollm/openai.go`:
   - Accept stop sequence slice and candidate count handling.
   - For non-stream and count>1, issue multiple `Responses.New` calls and yield sequentially.
   - Ignore seed, stop_seq errors; pass stop list to response converter and streaming aggregator.
6) Edit streaming aggregator in `gollm/internal/adapter/openai_response.go`: store stop sequences; trim final aggregated text; set finish reason accordingly; expose constructor signature.
7) Add/adjust tests covering stop trimming, function_call_output, candidate_count loop; update replays only if assertions require.
8) Run `go test ./gollm/...` from repo root.
9) Update this plan’s `Progress`, `Surprises`, `Decision Log`, and `Outcomes` with actual results.

## Validation and Acceptance

- Command (repo root): `go test ./gollm/...` must pass.
- Unit tests:
  - `TestOpenAIResponseToLLM_StopSequences` should show trimmed text and FinishReason=Stop.
  - `TestOpenAIResponseToLLM_FunctionCallOutput` should produce a `FunctionResponse` part with CallID and payload.
  - `TestOpenAILLM_Generate_CandidateCount` should emit multiple responses when `CandidateCount=2`.
- Manual reasoning: stop sequences should not appear in final content; candidate loop should yield the expected number of responses.

## Idempotence and Recovery

- Changes are additive and can be reapplied; running `go test` is safe repeatedly.
- If replays need regeneration, re-run specific tests to rewrite fixtures; otherwise no external state.
- If a change introduces failures, revert the specific file edits; no migrations involved.

## Artifacts and Notes

- Keep new helper functions small; prefer comments near compatibility shims (stop trimming, candidate loop) to explain rationale.
- Note: OpenAI Responses lacks native stop/candidate support; compatibility is emulated locally.

## Interfaces and Dependencies

- Anthropic: `anthropic.BetaMessageNewParams` mutators; ensure tool choice and metadata can be set via ProviderParams.
- OpenAI: `responses.ResponseNewParams` primary path; manual stop/candidate handling layered on top.
- xAI: only provider options append; no further interface change expected.
