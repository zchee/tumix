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
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestLLMNames(t *testing.T) {
	t.Parallel()

	cases := map[string]string{
		"anthropic": (&anthropicLLM{name: "claude"}).Name(),
		"openai":    (&openAILLM{name: "gpt"}).Name(),
		"xai":       (&xaiLLM{name: "grok"}).Name(),
	}

	for provider, got := range cases {
		if got == "" {
			t.Fatalf("%s Name() empty", provider)
		}
	}
}

func TestAnthropicBuildParamsDefaults(t *testing.T) {
	t.Parallel()

	llm := &anthropicLLM{name: "claude-3-haiku"}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Temperature:     ptrFloat32(0.5),
			TopP:            ptrFloat32(0.9),
			TopK:            ptrFloat32(3),
			StopSequences:   []string{"END"},
			MaxOutputTokens: 0, // triggers default
		},
	}
	system := []anthropic.BetaTextBlockParam{}
	msgs := []anthropic.BetaMessageParam{{
		Role: anthropic.BetaMessageParamRoleUser,
		Content: []anthropic.BetaContentBlockParamUnion{
			anthropic.NewBetaTextBlock("hi"),
		},
	}}

	params, err := llm.buildParams(req, system, msgs)
	if err != nil {
		t.Fatalf("buildParams() error = %v", err)
	}
	if params.MaxTokens != 1024 {
		t.Fatalf("MaxTokens = %d, want default 1024", params.MaxTokens)
	}
	if params.StopSequences[0] != "END" {
		t.Fatalf("StopSequences = %v", params.StopSequences)
	}
	if !almost(params.Temperature.Or(0), 0.5) || !almost(params.TopP.Or(0), 0.9) || params.TopK.Or(0) != 3 {
		t.Fatalf("temp/top: %v %v %v", params.Temperature.Or(0), params.TopP.Or(0), params.TopK.Or(0))
	}
}

func TestAnthropicBuildParamsNoMessages(t *testing.T) {
	t.Parallel()

	llm := &anthropicLLM{name: "claude"}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
	}
	if _, err := llm.buildParams(req, nil, nil); err == nil {
		t.Fatalf("expected error for empty messages")
	}
}

func TestOpenAIResponseParamsLogprobs(t *testing.T) {
	t.Parallel()

	llm := &openAILLM{name: "gpt-4o"}
	logprobs := int32(3)
	cfg := &genai.GenerateContentConfig{
		Temperature:      ptrFloat32(0.2),
		TopP:             ptrFloat32(0.7),
		MaxOutputTokens:  12,
		Logprobs:         &logprobs,
		ResponseLogprobs: true,
	}
	req := &model.LLMRequest{
		Contents: genai.Text("ping"),
		Config:   cfg,
	}

	params, err := llm.responseParams(req)
	if err != nil {
		t.Fatalf("responseParams() error = %v", err)
	}
	if !almost(params.Temperature.Or(0), 0.2) || !almost(params.TopP.Or(0), 0.7) {
		t.Fatalf("temp/topP mismatch")
	}
	if params.MaxOutputTokens.Or(0) != 12 {
		t.Fatalf("MaxOutputTokens = %d, want 12", params.MaxOutputTokens.Or(0))
	}
	if params.TopLogprobs.Or(0) != int64(logprobs) {
		t.Fatalf("TopLogprobs = %d, want %d", params.TopLogprobs.Or(0), logprobs)
	}
	if len(params.Include) == 0 {
		t.Fatalf("Include empty, want logprobs included")
	}
}

func TestOpenAIResponseParamsErrors(t *testing.T) {
	t.Parallel()

	llm := &openAILLM{name: "gpt-4o-mini"}
	if _, err := llm.responseParams(&model.LLMRequest{
		Contents: nil,
		Config:   &genai.GenerateContentConfig{},
	}); err == nil {
		t.Fatalf("expected error on empty contents")
	}
}

func TestXAIName(t *testing.T) {
	t.Parallel()

	m := &xaiLLM{name: "grok-mini"}
	if m.Name() != "grok-mini" {
		t.Fatalf("Name = %q, want grok-mini", m.Name())
	}
}

// helper functions reused from adapter/xai_test.go.
func ptrFloat32(v float32) *float32 { return &v }

func almost(got, want float64) bool {
	if got > want {
		return got-want < 1e-6
	}
	return want-got < 1e-6
}
