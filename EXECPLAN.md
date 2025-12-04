# ExecPlan: Add CLI Flag for LLM Backend Selection

I will add a new CLI flag `-backend` to allow users to select the LLM backend (Gemini, OpenAI, Anthropic, xAI).

## 1. Modify `config` struct in `main.go`
- [x] Add `LLMBackend` field to `config` struct.

## 2. Update `parseConfig` in `main.go`
- [x] Initialize `LLMBackend` with `envOrDefault("TUMIX_BACKEND", "gemini")`.
- [x] Add `flag.StringVar` for `-backend` flag.
- [x] Validate that `LLMBackend` is one of the supported values (`gemini`, `openai`, `anthropic`, `xai`).

## 3. Update `buildModel` in `main.go`
- [x] Import `github.com/zchee/tumix/gollm`.
- [x] Switch on `cfg.LLMBackend` to instantiate the appropriate LLM.
    - `gemini`: Use `gemini.NewModel` (existing logic).
    - `openai`: Use `gollm.NewOpenAILLM`.
    - `anthropic`: Use `gollm.NewAnthropicLLM`.
    - `xai`: Use `gollm.NewXAILLM`.
- [x] Pass `cfg.APIKey` wrapped in `gollm.AuthMethodAPIKey` where appropriate.

## 4. Verification
- [x] Run `go build ./...` to check for compilation errors.
