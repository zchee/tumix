# tumix

tumix implements a [TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture](https://arxiv.org/abs/2510.01279) in Go.

## Tracing

HTTP client tracing is off by default to minimize overhead. Set `TUMIX_HTTP_TRACE=1` to enable OpenTelemetry spans for outgoing LLM HTTP calls; clients will keep using the shared pooled transport either way.

## CLI flags (snapshot)

- `-model` (default `gemini-2.5-flash`)
- `-max_rounds` (default 3; higher improves quality, raises cost)
- `-temperature` / `-top_p` / `-top_k` / `-max_tokens` / `-seed`
- `-json` (emit final answer as JSON on stdout)
- `-session_dir` (persist sessions to disk; default in-memory)
- `TUMIX_SESSION_SQLITE` env to use sqlite-backed store instead of session_dir
- `-batch_file` with `-concurrency` (one prompt per line)
- `-http_trace` (enable HTTP spans)
- `-otlp_endpoint` (export traces)
- `-bench_local` to run synthetic local benchmark (no LLM calls)
- `-max_prompt_chars` to fail fast on oversized prompts
- `-max_prompt_tokens` heuristic guard; pricing override via `TUMIX_PRICING_FILE`
- `-metrics_addr` serve `/healthz` and `/debug/vars`

Env overrides: `GOOGLE_API_KEY`, `TUMIX_MODEL`, `TUMIX_MAX_ROUNDS`, `TUMIX_TEMPERATURE`, `TUMIX_TOP_P`, `TUMIX_TOP_K`, `TUMIX_MAX_TOKENS`, `TUMIX_SESSION_DIR`, `TUMIX_HTTP_TRACE`, `TUMIX_CALL_WARN`, `TUMIX_CONCURRENCY`.

## Quick recipes

- Low cost: `./tumix -model gemini-2.5-flash -max_rounds 2 -temperature 0.2 "Explain X"`
- High quality: `./tumix -model gemini-2.5-pro -max_rounds 3 -top_p 0.95 -max_tokens 512 "Explain Y"`
- Batch: `./tumix -batch_file prompts.txt -concurrency 4 -json`
- Persist & observe: `TUMIX_SESSION_SQLITE=/tmp/tumix.db ./tumix -metrics_addr :9090 -http_trace`
