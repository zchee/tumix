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
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/gollm/internal/httputil"
	"github.com/zchee/tumix/internal/version"
)

// openAILLM implements the adk [model.LLM] interface using OpenAI SDK.
type openAILLM struct {
	client    openai.Client
	name      string
	userAgent string
}

var _ model.LLM = (*openAILLM)(nil)

// NewOpenAILLM creates a new OpenAI-backed LLM.
//
// If authKey is nil, the OpenAI SDK falls back to the OPENAI_API_KEY environment variable.
//
//nolint:unparam
func NewOpenAILLM(_ context.Context, authKey AuthMethod, modelName string, opts ...option.RequestOption) (model.LLM, error) {
	userAgent := version.UserAgent("openai")

	httpClient := httputil.NewClient(3 * time.Minute)
	ropts := []option.RequestOption{
		option.WithHTTPClient(httpClient),
		option.WithHeader("User-Agent", userAgent),
		option.WithMaxRetries(2),
	}
	if authKey != nil {
		ropts = append(ropts, option.WithAPIKey(authKey.value()))
	}

	// opts are allowed to override by order
	opts = append(ropts, opts...)
	client := openai.NewClient(opts...)

	return &openAILLM{
		client:    client,
		name:      modelName,
		userAgent: userAgent,
	}, nil
}

// Name implements [model.LLM].
func (m *openAILLM) Name() string { return m.name }

// GenerateContent implements [model.LLM].
func (m *openAILLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	ensureUserContent(req)
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.HTTPOptions == nil {
		req.Config.HTTPOptions = &genai.HTTPOptions{}
	}
	if req.Config.HTTPOptions.Headers == nil {
		req.Config.HTTPOptions.Headers = make(map[string][]string)
	}
	req.Config.HTTPOptions.Headers["User-Agent"] = []string{m.userAgent}

	params, err := m.chatCompletionParams(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}

	if stream {
		return m.stream(ctx, params)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.client.Chat.Completions.New(ctx, *params)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp, err := adapter.OpenAIResponseToLLM(resp)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(llmResp, nil)
	}
}

func (m *openAILLM) chatCompletionParams(req *model.LLMRequest) (*openai.ChatCompletionNewParams, error) {
	msgs, err := adapter.GenaiToOpenAIMessages(req.Contents)
	if err != nil {
		return nil, fmt.Errorf("convert content: %w", err)
	}
	if len(msgs) == 0 {
		return nil, fmt.Errorf("no messages to send")
	}

	params := openai.ChatCompletionNewParams{
		Model:    resolveModelName(req, m.name),
		Messages: msgs,
	}

	cfg := req.Config
	if cfg.Temperature != nil {
		params.Temperature = openai.Float(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = openai.Float(float64(*cfg.TopP))
	}
	if cfg.MaxOutputTokens > 0 {
		params.MaxTokens = openai.Int(int64(cfg.MaxOutputTokens))
		params.MaxCompletionTokens = openai.Int(int64(cfg.MaxOutputTokens))
	}
	if cfg.CandidateCount > 0 {
		params.N = openai.Int(int64(cfg.CandidateCount))
	}
	if len(cfg.StopSequences) > 0 {
		// OpenAI stop accepts string or []string; we set []string.
		params.Stop = openai.ChatCompletionNewParamsStopUnion{OfStringArray: cfg.StopSequences}
	}
	if cfg.Seed != nil {
		params.Seed = openai.Int(int64(*cfg.Seed))
	}
	switch {
	case cfg.Logprobs != nil:
		params.Logprobs = openai.Bool(true)
		params.TopLogprobs = openai.Int(int64(*cfg.Logprobs))
	case cfg.ResponseLogprobs:
		params.Logprobs = openai.Bool(true)
	}
	if cfg.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(float64(*cfg.FrequencyPenalty))
	}
	if cfg.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(float64(*cfg.PresencePenalty))
	}

	if len(cfg.Tools) > 0 {
		tools, tc := adapter.GenaiToolsToOpenAI(cfg.Tools, cfg.ToolConfig)
		params.Tools = tools
		if tc != nil {
			params.ToolChoice = *tc
		}
	}

	return &params, nil
}

func (m *openAILLM) stream(ctx context.Context, params *openai.ChatCompletionNewParams) iter.Seq2[*model.LLMResponse, error] {
	stream := m.client.Chat.Completions.NewStreaming(ctx, *params)
	agg := adapter.NewOpenAIStreamAggregator()

	return func(yield func(*model.LLMResponse, error) bool) {
		defer stream.Close()

		for stream.Next() {
			chunk := stream.Current()

			for _, resp := range agg.Process(&chunk) {
				if !yield(resp, nil) {
					return
				}
			}
		}

		if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
			yield(nil, err)
			return
		}

		if final := agg.Final(); final != nil {
			yield(final, nil)
		}
	}
}
