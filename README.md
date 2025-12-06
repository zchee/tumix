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
