// Copyright 2025 The tumix Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package gollm

import (
	"context"
	"errors"
	"iter"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/gollm/internal/httputil"
	"github.com/zchee/tumix/internal/version"
	"github.com/zchee/tumix/telemetry"
)

var anthropicTracer = otel.Tracer("github.com/zchee/tumix/gollm/anthropic")

// anthropicLLM implements the adk [model.LLM] interface using the Anthropic SDK.
type anthropicLLM struct {
	client         *anthropic.Client
	name           string
	userAgent      string
	providerParams *ProviderParams
}

var _ model.LLM = (*anthropicLLM)(nil)

// NewAnthropicLLM creates a new Anthropic-backed LLM.
//
// If authKey is nil, the Anthropic SDK falls back to the ANTHROPIC_API_KEY environment variable.
//
//nolint:unparam
func NewAnthropicLLM(_ context.Context, apiKey, modelName string, params *ProviderParams, opts ...option.RequestOption) (model.LLM, error) {
	userAgent := version.UserAgent("anthropic")

	httpClient := httputil.NewClient(3 * time.Minute)
	ropts := []option.RequestOption{
		option.WithHTTPClient(httpClient),
		option.WithHeader("User-Agent", userAgent),
		option.WithMaxRetries(2),
	}
	if apiKey != "" {
		ropts = append(ropts, option.WithAPIKey(apiKey))
	}

	// opts are allowed to override by order
	opts = append(ropts, opts...)
	client := anthropic.NewClient(opts...)

	return &anthropicLLM{
		client:         &client,
		name:           modelName,
		userAgent:      userAgent,
		providerParams: params,
	}, nil
}

// Name implements [model.LLM].
func (m *anthropicLLM) Name() string { return m.name }

// GenerateContent implements [model.LLM].
func (m *anthropicLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	ctx, span := anthropicTracer.Start(ctx, "gollm.anthropic.GenerateContent")
	cfg := adapter.NormalizeRequest(req, m.userAgent)

	system, msgs, err := adapter.GenAIToAnthropicMessages(cfg.SystemInstruction, req.Contents)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			defer func() { telemetry.End(span, err) }()
			yield(nil, err)
		}
	}

	params, err := m.buildParams(req, system, msgs)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			defer func() { telemetry.End(span, err) }()
			yield(nil, err)
		}
	}

	if stream {
		return m.stream(ctx, params)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		var spanErr error
		defer func() { telemetry.End(span, spanErr) }()

		resp, err := m.client.Beta.Messages.New(ctx, *params)
		if err != nil {
			spanErr = err
			yield(nil, err)
			return
		}
		llmResp, convErr := adapter.AnthropicMessageToLLMResponse(resp)
		if convErr != nil {
			spanErr = convErr
			yield(nil, convErr)
			return
		}
		yield(llmResp, nil)
	}
}

// buildParams prepares Anthropic Beta message parameters from a genai request.
//
// It validates presence of messages, maps config knobs, and sets defaults
// required by the Anthropic API.
func (m *anthropicLLM) buildParams(req *model.LLMRequest, system []anthropic.BetaTextBlockParam, msgs []anthropic.BetaMessageParam) (*anthropic.BetaMessageNewParams, error) {
	if len(msgs) == 0 {
		return nil, errors.New("no messages")
	}

	params := &anthropic.BetaMessageNewParams{
		Model:         anthropic.Model(adapter.ModelName(m.name, req)),
		Messages:      msgs,
		System:        system,
		MaxTokens:     int64(req.Config.MaxOutputTokens),
		StopSequences: req.Config.StopSequences,
	}

	if params.MaxTokens == 0 {
		// Anthropic requires max_tokens; fall back to a conservative default.
		params.MaxTokens = 1024
	}
	if req.Config.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*req.Config.Temperature))
	}
	if req.Config.TopP != nil {
		params.TopP = param.NewOpt(float64(*req.Config.TopP))
	}
	if req.Config.TopK != nil {
		params.TopK = param.NewOpt(int64(*req.Config.TopK))
	}
	if len(req.Config.Tools) > 0 {
		tools, tc := adapter.GenAIToolsToAnthropic(req.Config.Tools, req.Config.ToolConfig)
		params.Tools = tools
		if tc != nil {
			params.ToolChoice = *tc
		}
	}

	applyAnthropicProviderParams(req, m.providerParams, params)

	return params, nil
}

// stream consumes Anthropic streaming responses and converts them to LLM responses.
//
// It accumulates incremental deltas, yielding partial text as it arrives and a final
// response once the stream signals completion.
func (m *anthropicLLM) stream(ctx context.Context, params *anthropic.BetaMessageNewParams) iter.Seq2[*model.LLMResponse, error] {
	span := trace.SpanFromContext(ctx)

	stream := m.client.Beta.Messages.NewStreaming(ctx, *params)
	acc := &anthropic.BetaMessage{}

	return func(yield func(*model.LLMResponse, error) bool) {
		defer stream.Close()

		var spanErr error
		defer func() { telemetry.End(span, spanErr) }()

		for stream.Next() {
			event := stream.Current()
			if err := acc.Accumulate(event); err != nil {
				spanErr = err
				yield(nil, err)
				return
			}

			switch ev := event.AsAny().(type) {
			case anthropic.BetaRawContentBlockDeltaEvent:
				if delta := ev.Delta.AsAny(); delta != nil {
					if t, ok := delta.(anthropic.BetaTextDelta); ok && t.Text != "" {
						resp := &model.LLMResponse{
							Content: &genai.Content{
								Role: genai.RoleModel,
								Parts: []*genai.Part{
									genai.NewPartFromText(adapter.AccText(acc)),
								},
							},
							Partial: true,
						}
						if !yield(resp, nil) {
							return
						}
					}
				}

			case anthropic.BetaRawMessageStopEvent:
				resp, err := adapter.AnthropicMessageToLLMResponse(acc)
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(resp, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil {
			spanErr = err
			yield(nil, err)
		}
	}
}

func applyAnthropicProviderParams(req *model.LLMRequest, defaults *ProviderParams, params *anthropic.BetaMessageNewParams) {
	pp, ok := effectiveProviderParams(req, defaults)
	if !ok || pp.Anthropic == nil {
		return
	}

	for _, mutate := range pp.Anthropic.Mutate {
		if mutate == nil {
			continue
		}
		mutate(params)
	}
}
