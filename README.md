# tumix

tumix implements a [TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture](https://arxiv.org/abs/2510.01279) in Go.

## Tracing

HTTP client tracing is off by default to minimize overhead. Set `TUMIX_HTTP_TRACE=1` to enable OpenTelemetry spans for outgoing LLM HTTP calls; clients will keep using the shared pooled transport either way.

## Provider-specific parameters

`model.LLMRequest` now accepts provider overrides via `SetProviderParams`. This keeps the ADK `LLM` interface unchanged while letting each adapter mutate SDK-specific params just before the request is sent.

```go
req := &model.LLMRequest{
	Model:    "gpt-4o",
	Contents: genai.Text("Translate this."),
	Config:   &genai.GenerateContentConfig{},
}

SetProviderParams(req, &ProviderParams{
	OpenAI: &OpenAIProviderParams{
		Mutate: []OpenAIParamMutator{
			func(p *openai.ChatCompletionNewParams) {
				p.Store = openai.Bool(true)
				p.PromptCacheKey = openai.String("txn-123")
			},
		},
	},
})
```

## CLI flags (snapshot)

- `-model` (default `gemini-2.5-flash`)
- `-max_rounds` (default 3; higher improves quality, raises cost)
- `-min_rounds` (default 2; judge cannot stop before this)
- `-temperature` / `-top_p` / `-top_k` / `-max_tokens` / `-seed`
- `-json` (emit final answer as JSON on stdout)
- `-session_dir` (persist sessions to disk; default in-memory)
- `TUMIX_SESSION_SQLITE` env to use sqlite-backed store instead of session_dir
- `-batch_file` with `-concurrency` (one prompt per line)
- `-http_trace` (enable HTTP spans)
- `-otlp_endpoint` (export traces)
- `-max_cost_usd` (default 0.01; per-run budget, caps rounds)
- `-auto_agents` (add N auto-designed agents for diversity)
- `-budget_tokens` (override per-call prompt token estimate for budgeting)
- `-bench_local` to run synthetic local benchmark (no LLM calls)
- `-max_prompt_chars` to fail fast on oversized prompts
- `-max_prompt_tokens` tokenizer-backed guard (CountTokens) with heuristic fallback; pricing override via `TUMIX_PRICING_FILE`
- `-metrics_addr` serve `/healthz`, `/debug/vars`, `/metrics` (Prometheus text)

Env overrides: `GOOGLE_API_KEY`, `TUMIX_MODEL`, `TUMIX_MAX_ROUNDS`, `TUMIX_TEMPERATURE`, `TUMIX_TOP_P`, `TUMIX_TOP_K`, `TUMIX_MAX_TOKENS`, `TUMIX_SESSION_DIR`, `TUMIX_HTTP_TRACE`, `TUMIX_CALL_WARN`, `TUMIX_CONCURRENCY`.

## Quick recipes

- Low cost: `./tumix -model gemini-2.5-flash -max_rounds 2 -temperature 0.2 "Explain X"`
- High quality: `./tumix -model gemini-2.5-pro -max_rounds 3 -top_p 0.95 -max_tokens 512 "Explain Y"`
- Batch: `./tumix -batch_file prompts.txt -concurrency 4 -json`
- Persist & observe: `TUMIX_SESSION_SQLITE=/tmp/tumix.db ./tumix -metrics_addr :9090 -http_trace`
- CI smoke: `./tools/bin/gotestsum -f standard-verbose -- -race -count=1 -shuffle=on -cover ./...`

## Observability

- `/metrics` emits counters: `tumix_requests`, `tumix_input_tokens`, `tumix_output_tokens`, `tumix_cost_usd`.
- `/debug/vars` exposes the same data via expvar; `/healthz` returns `ok`.
