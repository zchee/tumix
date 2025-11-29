# Refactoring Plan for @gollm/**

## Objective
Refactor the `gollm` package to improve maintainability, standardization, and performance. The current state involves monolithic adapter files (`anthropic.go`, `openai.go`, `xai.go`) and a complex custom xAI client. We aim to modularize logic, standardize infrastructure (tracing, HTTP clients), and optimize the xAI implementation.

## Phase 1: Modularization & Decoupling (High Priority)
**Goal:** Break down monolithic adapter files into logic-specific components to improve readability and testability.

- [ ] **Refactor Anthropic Adapter** (`gollm/anthropic.go`)
    - [ ] Extract message conversion (`genaiToAnthropicMessages`) to `gollm/anthropic_messages.go`.
    - [ ] Extract tool conversion (`genaiToolsToAnthropic`) to `gollm/anthropic_tools.go`.
    - [ ] Extract response mapping (`anthropicMessageToLLMResponse`) to `gollm/anthropic_response.go`.
    - [ ] Keep `NewAnthropicLLM` and `GenerateContent` high-level flow in `gollm/anthropic.go`.

- [ ] **Refactor OpenAI Adapter** (`gollm/openai.go`)
    - [ ] Extract message conversion (`genaiToOpenAIMessages`) to `gollm/openai_messages.go`.
    - [ ] Extract tool conversion (`genaiToolsToOpenAI`) to `gollm/openai_tools.go`.
    - [ ] Extract response mapping (`openAIResponseToLLM` & stream aggregation) to `gollm/openai_response.go`.

- [ ] **Refactor xAI Adapter** (`gollm/xai.go`)
    - [ ] Extract message conversion (`GenAIContentsToMessages` - currently in `gollm/xai/genai_bridge.go`, move/consolidate logic if needed) and response mapping (`xai2LLMResponse`) into `gollm/xai_mapping.go` (or similar) to keep the root `gollm` package clean. *Note: `xai.go` currently imports `gollm/xai` package. We should clarify if the bridge logic belongs in the client package or the adapter package.*

## Phase 2: Infrastructure & Standardization (High Impact)
**Goal:** Ensure consistent behavior for Observability, Authentication, and HTTP Transport across all REST-based providers.

- [ ] **Standardize HTTP Client**
    - [ ] Create `gollm/internal/httputil` package.
    - [ ] Implement `NewClient` factory that accepts `http.Client` options and injects OTel transport and standardized User-Agent.
    - [ ] Update `anthropic` and `openai` adapters to use this factory, resolving the TODOs regarding OTel tracing.

- [ ] **Unified Configuration Mapping**
    - [ ] (Optional) Create internal helpers to map `genai.GenerateContentConfig` fields (Temperature, TopP, etc.) to provider-specific types to reduce boilerplate in `GenerateContent` methods.

## Phase 3: xAI Client Optimization (`gollm/xai/`)
**Goal:** Validate and polish the custom xAI gRPC client implementation.

- [ ] **Review & Optimize Codec**
    - [ ] Audit `gollm/xai/internal/grpccodec` for potential memory leaks or unsafe usage of `buffer_pool`.
    - [ ] Verify `vtprotobuf` integration is actually triggering.

- [ ] **Refine Chat Session**
    - [ ] Review `ChatSession` in `gollm/xai` for thread-safety if it's intended to be shared (currently looks mutable and not safe). Add documentation or locking if needed.

## Phase 4: Testing & Validation
- [ ] **Benchmarks**: Add benchmarks for message conversion logic to ensure refactoring didn't introduce regressions.
- [ ] **Integration Tests**: Ensure all providers pass the `adk` compliance tests (implied).
