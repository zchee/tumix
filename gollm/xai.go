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
	"fmt"
	"iter"

	"google.golang.org/adk/model"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/gollm/xai"
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
	"github.com/zchee/tumix/internal/version"
)

// xaiLLM implements the adk [model.LLM] interface using xAI SDK.
type xaiLLM struct {
	client         *xai.Client
	name           string
	userAgent      string
	providerParams *ProviderParams
}

var _ model.LLM = (*xaiLLM)(nil)

// NewXAILLM creates a new xAI-backed LLM.
//
// If authKey is nil, the xAI SDK falls back to the XAI_API_KEY environment variable.
func NewXAILLM(_ context.Context, authKey AuthMethod, modelName string, params *ProviderParams, opts ...xai.ClientOption) (model.LLM, error) {
	var apiKey string
	if authKey != nil {
		apiKey = authKey.value()
	}

	client, err := xai.NewClient(apiKey, opts...)
	if err != nil {
		return nil, fmt.Errorf("new xAI client: %w", err)
	}

	// Create userAgent header value once, when the model is created
	userAgent := version.UserAgent("xai")

	return &xaiLLM{
		client:         client,
		name:           modelName,
		userAgent:      userAgent,
		providerParams: params,
	}, nil
}

// Name implements [model.LLM].
func (m *xaiLLM) Name() string { return m.name }

// GenerateContent implements [model.LLM].
func (m *xaiLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	cfg := adapter.NormalizeRequest(req, m.userAgent)

	msgs, err := xai.GenAIContentsToMessages(cfg.SystemInstruction, req.Contents)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}

	if stream {
		return m.generateStream(ctx, req, msgs)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req, msgs)
		yield(resp, err)
	}
}

// generate calls the model synchronously returning result from the first candidate.
func (m *xaiLLM) generate(ctx context.Context, req *model.LLMRequest, msgs []*xaipb.Message) (*model.LLMResponse, error) {
	opts := []xai.ChatOption{
		xai.WithMessages(msgs...),
	}
	if opt := adapter.GenAI2XAIChatOptions(req.Config); opt != nil {
		opts = append(opts, opt)
	}
	opts = appendXAIProviderOptions(req, m.providerParams, opts)
	sess := m.client.Chat.Create(adapter.ModelName(m.name, req), opts...)

	resp, err := sess.Completion(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}

	if resp.Content() == "" {
		// shouldn't happen?
		return nil, fmt.Errorf("empty response")
	}

	return adapter.XAIResponseToLLM(resp), nil
}

// generateStream returns a stream of responses from the model.
func (m *xaiLLM) generateStream(ctx context.Context, req *model.LLMRequest, msgs []*xaipb.Message) iter.Seq2[*model.LLMResponse, error] {
	aggregator := adapter.NewXAIStreamAggregator()

	return func(yield func(*model.LLMResponse, error) bool) {
		opts := []xai.ChatOption{
			xai.WithMessages(msgs...),
		}
		if opt := adapter.GenAI2XAIChatOptions(req.Config); opt != nil {
			opts = append(opts, opt)
		}
		opts = appendXAIProviderOptions(req, m.providerParams, opts)
		sess := m.client.Chat.Create(adapter.ModelName(m.name, req), opts...)

		stream, err := sess.Stream(ctx)
		if err != nil {
			yield(nil, err)
			return
		}
		for resp, err := range stream.Recv() {
			if err != nil {
				yield(nil, err)
				return
			}

			for llmResponse, err := range aggregator.Process(ctx, resp) {
				if !yield(llmResponse, err) {
					return // Consumer stopped
				}
			}
		}
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

func appendXAIProviderOptions(req *model.LLMRequest, defaults *ProviderParams, opts []xai.ChatOption) []xai.ChatOption {
	pp, ok := effectiveProviderParams(req, defaults)
	if !ok || pp.XAI == nil || len(pp.XAI.Options) == 0 {
		return opts
	}

	return append(opts, pp.XAI.Options...)
}
