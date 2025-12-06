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
	"slices"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/gollm/xai"
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

func TestProviderParams_AnthropicMutate(t *testing.T) {
	req := &model.LLMRequest{
		Contents: genai.Text("ping"),
		Config:   &genai.GenerateContentConfig{},
	}
	SetProviderParams(req, &ProviderParams{
		Anthropic: &AnthropicProviderParams{
			Mutate: []AnthropicParamMutator{
				func(p *anthropic.BetaMessageNewParams) {
					p.Metadata.UserID = param.NewOpt("user-123")
				},
			},
		},
	})

	cfg := adapter.NormalizeRequest(req, "test-agent")
	system, msgs, err := adapter.GenAIToAnthropicBetaMessages(cfg.SystemInstruction, req.Contents)
	if err != nil {
		t.Fatalf("GenAIToAnthropicBetaMessages() error = %v", err)
	}

	llm := &anthropicLLM{name: "claude-haiku-4-5", userAgent: "test-agent"}
	params, err := llm.buildParams(req, system, msgs)
	if err != nil {
		t.Fatalf("buildParams() error = %v", err)
	}

	if got := params.Metadata.UserID.Or(""); got != "user-123" {
		t.Fatalf("Metadata.UserID = %q, want %q", got, "user-123")
	}
}

func TestProviderParams_OpenAIMutate(t *testing.T) {
	req := &model.LLMRequest{
		Contents: genai.Text("hello"),
		Config:   &genai.GenerateContentConfig{},
	}
	req.Config = adapter.NormalizeRequest(req, "test-agent")

	SetProviderParams(req, &ProviderParams{
		OpenAI: &OpenAIProviderParams{
			Mutate: []OpenAIParamMutator{
				func(p *responses.ResponseNewParams) {
					p.Store = openai.Bool(true)
					p.TopP = openai.Float(0.42)
				},
			},
		},
	})

	llm := &openAILLM{name: "gpt-4o", userAgent: "test-agent"}
	params, err := llm.responseParams(req)
	if err != nil {
		t.Fatalf("chatCompletionParams() error = %v", err)
	}

	if got := params.Store.Or(false); !got {
		t.Fatalf("Store.Or(false) = %v, want true", got)
	}
	if got := params.TopP.Or(0); got != 0.42 {
		t.Fatalf("TopP.Or(0) = %v, want %v", got, 0.42)
	}
}

func TestProviderParams_XAIOptions(t *testing.T) {
	req := &model.LLMRequest{
		Contents: genai.Text("hi"),
		Config:   &genai.GenerateContentConfig{},
	}
	SetProviderParams(req, &ProviderParams{
		XAI: &XAIProviderParams{
			Options: []xai.ChatOption{
				xai.WithUser("user-abc"),
				xai.WithMaxTokens(321),
			},
		},
	})

	opts := appendXAIProviderOptions(req, nil, []xai.ChatOption{
		xai.WithMessages(&xaipb.Message{Role: xaipb.MessageRole_ROLE_USER}),
	})

	gotReq := &xaipb.GetCompletionsRequest{}
	for _, opt := range opts {
		opt(gotReq, &xai.ChatSession{})
	}

	if gotReq.GetUser() != "user-abc" {
		t.Fatalf("User = %q, want %q", gotReq.GetUser(), "user-abc")
	}
	if gotReq.GetMaxTokens() != 321 {
		t.Fatalf("MaxTokens = %v, want 321", gotReq.GetMaxTokens())
	}
}

func TestProviderParams_DefaultsAppliedToOpenAI(t *testing.T) {
	req := &model.LLMRequest{
		Contents: genai.Text("hello"),
		Config:   &genai.GenerateContentConfig{},
	}
	req.Config = adapter.NormalizeRequest(req, "test-agent")

	llm := &openAILLM{
		name:      "gpt-4o",
		userAgent: "test-agent",
		providerParams: &ProviderParams{
			OpenAI: &OpenAIProviderParams{
				Mutate: []OpenAIParamMutator{
					func(p *responses.ResponseNewParams) {
						p.Store = openai.Bool(true)
						p.TopP = openai.Float(0.33)
					},
				},
			},
		},
	}

	params, err := llm.responseParams(req)
	if err != nil {
		t.Fatalf("responseParams() error = %v", err)
	}
	if got := params.Store.Or(false); !got {
		t.Fatalf("Store = %v, want true", got)
	}
	if got := params.TopP.Or(0); got != 0.33 {
		t.Fatalf("TopP = %v, want 0.33", got)
	}
}

func TestProviderParams_DefaultsMergedWithRequest_OpenAI(t *testing.T) {
	req := &model.LLMRequest{
		Contents: genai.Text("hello"),
		Config:   &genai.GenerateContentConfig{},
	}
	req.Config = adapter.NormalizeRequest(req, "test-agent")

	SetProviderParams(req, &ProviderParams{
		OpenAI: &OpenAIProviderParams{
			Mutate: []OpenAIParamMutator{
				func(p *responses.ResponseNewParams) {
					p.Store = openai.Bool(false)
					p.TopP = openai.Float(0.12)
				},
			},
		},
	})

	llm := &openAILLM{
		name:      "gpt-4o",
		userAgent: "test-agent",
		providerParams: &ProviderParams{
			OpenAI: &OpenAIProviderParams{
				Mutate: []OpenAIParamMutator{
					func(p *responses.ResponseNewParams) {
						p.Store = openai.Bool(true)
						p.TopP = openai.Float(0.9)
					},
				},
			},
		},
	}

	params, err := llm.responseParams(req)
	if err != nil {
		t.Fatalf("responseParams() error = %v", err)
	}
	if got := params.Store.Or(true); got {
		t.Fatalf("Store = %v, want false (request override)", got)
	}
	if got := params.TopP.Or(0); got != 0.12 {
		t.Fatalf("TopP = %v, want 0.12 (request override)", got)
	}
}

func TestProviderParams_DefaultsAppliedToAnthropic(t *testing.T) {
	req := &model.LLMRequest{
		Contents: genai.Text("ping"),
		Config:   &genai.GenerateContentConfig{},
	}
	cfg := adapter.NormalizeRequest(req, "test-agent")
	system, msgs, err := adapter.GenAIToAnthropicBetaMessages(cfg.SystemInstruction, req.Contents)
	if err != nil {
		t.Fatalf("GenAIToAnthropicBetaMessages() error = %v", err)
	}

	llm := &anthropicLLM{
		name:      "claude-haiku-4-5",
		userAgent: "test-agent",
		providerParams: &ProviderParams{
			Anthropic: &AnthropicProviderParams{
				Mutate: []AnthropicParamMutator{
					func(p *anthropic.BetaMessageNewParams) {
						p.Metadata.UserID = param.NewOpt("user-default")
					},
				},
			},
		},
	}

	params, err := llm.buildParams(req, system, msgs)
	if err != nil {
		t.Fatalf("buildParams() error = %v", err)
	}

	if got := params.Metadata.UserID.Or(""); got != "user-default" {
		t.Fatalf("Metadata.UserID = %q, want %q", got, "user-default")
	}
}

func TestProviderParams_DefaultsAppliedToXAI(t *testing.T) {
	req := &model.LLMRequest{
		Contents: genai.Text("hi"),
		Config:   &genai.GenerateContentConfig{},
	}

	pp := &ProviderParams{
		XAI: &XAIProviderParams{
			Options: []xai.ChatOption{
				xai.WithUser("default-user"),
				xai.WithMaxTokens(111),
			},
		},
	}

	opts := appendXAIProviderOptions(req, pp, []xai.ChatOption{
		xai.WithMessages(&xaipb.Message{Role: xaipb.MessageRole_ROLE_USER}),
	})

	gotReq := &xaipb.GetCompletionsRequest{}
	for _, opt := range opts {
		opt(gotReq, &xai.ChatSession{})
	}

	if gotReq.GetUser() != "default-user" {
		t.Fatalf("User = %q, want %q", gotReq.GetUser(), "default-user")
	}
	if gotReq.GetMaxTokens() != 111 {
		t.Fatalf("MaxTokens = %v, want 111", gotReq.GetMaxTokens())
	}
}

func TestMergeMutatorsUsesOrderAndCopies(t *testing.T) {
	type sample struct {
		steps *[]string
	}

	base := &ProviderMutators[sample]{
		Mutate: []ProviderMutator[sample]{
			func(s *sample) { *s.steps = append(*s.steps, "base-1") },
		},
	}
	add := &ProviderMutators[sample]{
		Mutate: []ProviderMutator[sample]{
			func(s *sample) { *s.steps = append(*s.steps, "add-1") },
		},
	}

	merged := mergeMutators(copyMutators(base), copyMutators(add))
	if merged == nil {
		t.Fatal("mergeMutators() = nil, want non-nil")
	}

	var seq []string
	target := sample{steps: &seq}
	for _, mutate := range merged.Mutate {
		mutate(&target)
	}

	if !slices.Equal(seq, []string{"base-1", "add-1"}) {
		t.Fatalf("mutate order = %v, want [base-1 add-1]", seq)
	}

	base.Mutate = append(base.Mutate, func(*sample) {})
	if len(merged.Mutate) != 2 {
		t.Fatalf("merged mutate length mutated with base slice, got %d, want 2", len(merged.Mutate))
	}
}

func TestMergeOptionsNilSafeAndCopies(t *testing.T) {
	if got := mergeOptions[int](nil, nil); got != nil {
		t.Fatalf("mergeOptions(nil, nil) = %v, want nil", got)
	}

	base := &ProviderOptions[int]{Options: []int{1, 2}}
	add := &ProviderOptions[int]{Options: []int{3}}

	merged := mergeOptions(copyOptions(base), copyOptions(add))
	if merged == nil {
		t.Fatal("mergeOptions() = nil, want non-nil")
	}

	if !slices.Equal(merged.Options, []int{1, 2, 3}) {
		t.Fatalf("merged options = %v, want [1 2 3]", merged.Options)
	}

	base.Options[0] = 99
	if merged.Options[0] != 1 {
		t.Fatalf("merged options mutated after base change, got %d, want 1", merged.Options[0])
	}
}

func TestProviderParams_DefaultsAndRequestMergeForXAI(t *testing.T) {
	req := &model.LLMRequest{
		Contents: genai.Text("hi"),
		Config:   &genai.GenerateContentConfig{},
	}

	defaults := &ProviderParams{
		XAI: &XAIProviderParams{
			Options: []xai.ChatOption{
				xai.WithUser("default-user"),
				xai.WithMaxTokens(111),
			},
		},
	}
	SetProviderParams(req, &ProviderParams{
		XAI: &XAIProviderParams{
			Options: []xai.ChatOption{
				xai.WithUser("request-user"),
				xai.WithMaxTokens(222),
			},
		},
	})

	opts := appendXAIProviderOptions(req, defaults, []xai.ChatOption{
		xai.WithMessages(&xaipb.Message{Role: xaipb.MessageRole_ROLE_USER}),
	})

	gotReq := &xaipb.GetCompletionsRequest{}
	for _, opt := range opts {
		opt(gotReq, &xai.ChatSession{})
	}

	if gotReq.GetUser() != "request-user" {
		t.Fatalf("User = %q, want %q (request override)", gotReq.GetUser(), "request-user")
	}
	if gotReq.GetMaxTokens() != 222 {
		t.Fatalf("MaxTokens = %v, want 222 (request override)", gotReq.GetMaxTokens())
	}

	// Mutate defaults after merge to ensure merged slice was copied.
	defaults.XAI.Options[0] = xai.WithUser("mutated-user")
	defaults.XAI.Options[1] = xai.WithMaxTokens(999)

	gotReq2 := &xaipb.GetCompletionsRequest{}
	for _, opt := range opts {
		opt(gotReq2, &xai.ChatSession{})
	}
	if gotReq2.GetUser() != "request-user" || gotReq2.GetMaxTokens() != 222 {
		t.Fatalf("merged options mutated after defaults change, got user=%q max=%v", gotReq2.GetUser(), gotReq2.GetMaxTokens())
	}
}
