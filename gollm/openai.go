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
	"fmt"
	"io"
	"iter"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/adk/model"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/gollm/internal/httputil"
	"github.com/zchee/tumix/internal/version"
	"github.com/zchee/tumix/telemetry"
)

var openaiTracer = otel.Tracer("github.com/zchee/tumix/gollm/openai")

// openAILLM implements the adk [model.LLM] interface using OpenAI SDK.
type openAILLM struct {
	client         openai.Client
	name           string
	userAgent      string
	providerParams *ProviderParams
}

var _ model.LLM = (*openAILLM)(nil)

// NewOpenAILLM creates a new OpenAI-backed LLM.
//
// If authKey is nil, the OpenAI SDK falls back to the OPENAI_API_KEY environment variable.
//
//nolint:unparam
func NewOpenAILLM(_ context.Context, apiKey, modelName string, params *ProviderParams, opts ...option.RequestOption) (model.LLM, error) {
	userAgent := version.UserAgent("openai")

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
	client := openai.NewClient(opts...)

	return &openAILLM{
		client:         client,
		name:           modelName,
		userAgent:      userAgent,
		providerParams: params,
	}, nil
}

// Name implements [model.LLM].
func (m *openAILLM) Name() string { return m.name }

// GenerateContent implements [model.LLM].
func (m *openAILLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	ctx, span := openaiTracer.Start(ctx, "gollm.openai.GenerateContent")
	req.Config = adapter.NormalizeRequest(req, m.userAgent)

	params, err := m.responseParams(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			defer func() { telemetry.End(span, err) }()
			yield(nil, err)
		}
	}

	stopSeq := req.Config.StopSequences
	count := max(req.Config.CandidateCount, 1)

	if stream {
		// Responses streaming currently returns a single candidate; fall back to one even if caller asked for more.
		return m.stream(ctx, params, stopSeq)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		var spanErr error
		defer func() { telemetry.End(span, spanErr) }()

		for range count {
			resp, err := m.client.Responses.New(ctx, *params)
			if err != nil {
				spanErr = err
				yield(nil, err)
				return
			}

			llmResp, err := adapter.OpenAIResponseToLLM(resp, stopSeq)
			if err != nil {
				spanErr = err
				yield(nil, err)
				return
			}
			if !yield(llmResp, nil) {
				return
			}
		}
	}
}

// responseParams builds OpenAI Responses parameters from the request.
//
// It converts GenAI contents to Responses input items and applies generation
// config, returning an error when unsupported options are requested.
func (m *openAILLM) responseParams(req *model.LLMRequest) (*responses.ResponseNewParams, error) {
	items, err := adapter.GenAIToResponsesInput(req.Contents)
	if err != nil {
		return nil, fmt.Errorf("convert content: %w", err)
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("no input items to send")
	}

	params := responses.ResponseNewParams{
		Model: adapter.ModelName(m.name, req),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: responses.ResponseInputParam(items),
		},
	}

	cfg := req.Config
	if cfg.Temperature != nil {
		params.Temperature = param.NewOpt(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = param.NewOpt(float64(*cfg.TopP))
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(cfg.MaxOutputTokens))
	}
	switch {
	case cfg.Logprobs != nil:
		params.TopLogprobs = param.NewOpt(int64(*cfg.Logprobs))
		params.Include = append(params.Include, responses.ResponseIncludableMessageOutputTextLogprobs)
	case cfg.ResponseLogprobs:
		params.Include = append(params.Include, responses.ResponseIncludableMessageOutputTextLogprobs)
	}
	if len(cfg.Tools) > 0 {
		tools, tc := adapter.GenAIToolsToResponses(cfg.Tools, cfg.ToolConfig)
		params.Tools = tools
		if tc != nil {
			params.ToolChoice = *tc
		}
	}

	applyOpenAIProviderParams(req, m.providerParams, &params)

	return &params, nil
}

// stream executes a streaming chat completion request and aggregates partial responses.
//
// It forwards each streamed chunk through the OpenAI aggregator and emits final output
// after the stream ends, respecting consumer backpressure.
func (m *openAILLM) stream(ctx context.Context, params *responses.ResponseNewParams, stopSeq []string) iter.Seq2[*model.LLMResponse, error] {
	span := trace.SpanFromContext(ctx)

	agg := adapter.NewOpenAIStreamAggregator(stopSeq)
	return func(yield func(*model.LLMResponse, error) bool) {
		stream := m.client.Responses.NewStreaming(ctx, *params)
		defer stream.Close()

		var spanErr error
		defer func() { telemetry.End(span, spanErr) }()

		for stream.Next() {
			event := stream.Current()

			for _, resp := range agg.Process(&event) {
				if !yield(resp, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
			spanErr = err
			yield(nil, err)
			return
		}

		if aggErr := agg.Err(); aggErr != nil {
			spanErr = aggErr
			yield(nil, aggErr)
			return
		}

		if final := agg.Final(); final != nil {
			yield(final, nil)
		}
	}
}

func applyOpenAIProviderParams(req *model.LLMRequest, defaults *ProviderParams, params *responses.ResponseNewParams) {
	pp, ok := effectiveProviderParams(req, defaults)
	if !ok || pp.OpenAI == nil {
		return
	}
	for _, mutate := range pp.OpenAI.Mutate {
		if mutate == nil {
			continue
		}
		mutate(params)
	}
}
