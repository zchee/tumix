# TUMIX alignment with paper philosophy

This ExecPlan is a living document. It must be kept current while we align the Go implementation of tumix to the TUMIX paper’s philosophy: diverse tool-use agents, iterative sharing, LLM-based early stopping, majority-vote finalization, and cost-aware execution that keeps per-run spend near $0.01 (≈$500/month at 50K runs).

If PLANS.md existed it would govern, but none is present; follow this plan directly.

## Purpose / Big Picture
Enable tumix to mirror the paper’s behaviour: run 15 pre-designed agents (plus optional LLM-designed agents), share answers across rounds, stop when refinement gains vanish (min 2 rounds), and finalize via majority vote with a Gemini judge only deciding whether to stop. Add cost guards so typical runs stay under one cent using Gemini-2.5-flash while preserving quality.

## Progress
- [x] (2025-12-03T22:35Z) Read project instructions, ExecPlan rules, Go style, and current tumix code.
- [x] (2025-12-03T22:42Z) Downloaded and skimmed TUMIX paper; extracted agent roster and termination philosophy.
- [x] (2025-12-03T23:35Z) Finalized gap analysis: missing Guided+ agents, no cost cap, judge over-selects answer, no stats sharing.
- [x] (2025-12-03T23:55Z) Implemented roster (added Guided+ and auto agents), stats propagation, judge prompt rewrite, cost cap, CLI flags, and tests; all tests pass.
- [ ] (pending) Update docs beyond README snippet if needed and re-run checks after any follow-ups.

## Surprises & Discoveries
- TUMIX paper uses majority vote after early-stop decision; current code lets judge pick the answer directly, which risks single-model bias.
- Existing code built only 12 of the 15 agents and lacked cost/round gating; now expanded and budget-capped, but token estimates remain heuristic.

## Decision Log
- (2025-12-03) Keep judge as stop/continue arbiter; final answer chosen via majority (with judge recommendation fallback) to reduce single-model bias.
- (2025-12-03) Add Guided+ variants and optional auto-designed agents to reach/extend the 15-agent diversity described in the paper.
- (2025-12-03) Introduce per-run budget flag `-max_cost_usd` defaulting to $0.01 with conservative token estimates to satisfy the $500/month @50K runs constraint.

## Outcomes & Retrospective
- (pending) Summarize results and remaining gaps once implementation and tests land.

## Context and Orientation
- Entry point: `main.go` constructs 12 agents and a judge, runs fixed rounds (max_rounds flag, min rounds hard-coded to 2), and uses judge + finalize tool for early stop and answer selection.
- Agent definitions: `agent/agent.go` defines Base, CoT, CoT code, Search, Code, Code+, Dual-Tool (gs/llm/com), Guided (gs/llm/com), plus judge/orchestrator. Shared context is injected via session state (`question`, `joined_answers`, `round_num`).
- Tests: `main_test.go` covers env parsing and token checks; `agent/orchestrator_test.go` covers stop and majority fallback. No coverage for auto-agents or cost guards.
- Tooling: vendored ADK + go-genai; tests run with `./tools/bin/gotestsum -f standard-verbose -- -race -count=1 -shuffle=on -cover -covermode=atomic ./...`.

## Plan of Work
Narrative sequence to deliver the aligned behaviour:
1) Confirm cost target: per-run budget ≤$0.01 using price table in `main.go` (flash default). Use this to derive default `max_cost_usd` and dynamic round cap.
2) Define round-level stats (unique answers, vote margin, entropy, coverage) and persist them in session state for prompt injection.
3) Expand agent roster: add Guided+ variants (gs/llm/com) with hinted prompts; ensure all 15 pre-designed agents share the common context template.
4) Add optional LLM-designed agents: a designer routine that asks the model for N agent blueprints (name + description + tool emphasis) and instantiates llmagent configs accordingly; make it opt-in via CLI flag.
5) Rework orchestration: candidates run in parallel; after each round compute stats and store them; judge reads stats and only signals stop (not the final answer); when stopping (or hitting max rounds/budget) choose final answer via majority vote over latest candidate responses, with judge recommendation as tiebreaker if provided.
6) Implement cost-aware scheduler: estimate per-call token cost from model prices + max_tokens/heuristic; derive allowed rounds given max_cost_usd; monitor cumulative estimated cost during execution and stop early if exceeded.
7) Extend CLI/env: add `-min_rounds`, `-max_cost_usd`, `-auto_agents`, and `-budget_tokens` (optional prompt budget) with validation and defaults aligned to the paper (min rounds 2, default cost budget ~0.01).
8) Update shared context and judge instructions to include stats and clearer early-stop criteria (diversity collapse, vote margin).
9) Add tests: stats computation, majority voting with normalization, cost cap rounding, guided+ construction, auto-agent designer stub (using deterministic fake LLM), and judge stop behaviour.
10) Refresh README with new flags, cost guidance, and alignment notes; run gofmt/gofumpt and full test suite.

## Concrete Steps
Run from repo root unless noted.
1. Add round stats helpers and state keys in `agent/agent.go`.
2. Implement Guided+ agent constructors and add them to the loader list in `main.go`.
3. Implement auto-agent designer (new Go file in `agent/` or `internal/`) and integrate into loader when `-auto_agents > 0`.
4. Rework orchestrator to compute stats each round, pass to judge via state, let judge only decide stop, and finalize via majority vote (judge answer as tiebreaker).
5. Add cost budget calculation in `main.go` (config parsing + dynamic round cap + runtime guard) and surface new CLI/env flags.
6. Strengthen shared/judge instructions with stats placeholders and early-stop guidance.
7. Write tests for stats, cost guard, guided+ agents, auto-agent designer, and majority/stop behaviour.
8. Update README with new flags and cost guidance; ensure pricing table mention.
9. Format (`gofmt -w`, `gofumpt`, goimports) and run `./tools/bin/gotestsum -f standard-verbose -- -race -count=1 -shuffle=on -cover -covermode=atomic ./...` (fallback: `go test ./...` if tools missing).

## Validation and Acceptance
- Unit tests for new helpers and orchestrator pass.
- Running tumix with `-max_cost_usd 0.01 -model gemini-2.5-flash -max_rounds 6 -min_rounds 2 "<prompt>"` stops no later than cost/round cap and emits majority-voted final answer.
- Batch mode respects cost cap per prompt and reports any early-stop due to budget.
- README documents new flags; CLI `-h` shows them.

## Idempotence and Recovery
- Config parsing and cost guards are pure functions; rerunning with same inputs yields same round cap.
- Auto-agent generation is optional; when enabled, results are cached per run but do not mutate repo.
- No migrations; tests and gofmt can be rerun safely.

## Artifacts and Notes
- Pricing defaults remain in `main.go` but can be overridden via `TUMIX_PRICING_FILE` and new flags.
- Keep progress timestamps updated when major milestones are completed.

## Interfaces and Dependencies
- Model: `google.golang.org/genai` via ADK `model.LLM`.
- New CLI flags: `-min_rounds`, `-max_cost_usd`, `-auto_agents`, `-budget_tokens` (name may adjust during implementation); env overrides prefixed with `TUMIX_`.
- Agent constructors exposed from `agent/agent.go`; auto-agent designer helper to return `[]agent.Agent`.
