# tumix

tumix implements a [TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture](https://arxiv.org/abs/2510.01279) in Go.

## Tracing

HTTP client tracing is off by default to minimize overhead. Set `TUMIX_HTTP_TRACE=1` to enable OpenTelemetry spans for outgoing LLM HTTP calls; clients will keep using the shared pooled transport either way.
