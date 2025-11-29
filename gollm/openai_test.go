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

	"github.com/google/go-cmp/cmp"
	openai "github.com/openai/openai-go/v3"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestOpenAILLMChatCompletionParams(t *testing.T) {
	t.Parallel()

	temp := float32(0.25)
	topP := float32(0.6)
	maxTokens := int32(256)
	candidateCount := int32(2)
	seed := int32(11)
	logprobs := int32(3)
	freqPenalty := float32(0.1)
	presPenalty := float32(0.2)

	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "lookup_weather",
				Description: "Return the weather for a city",
				ParametersJsonSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
					"required": []string{"city"},
				},
			},
		},
	}

	tests := map[string]struct {
		req   *model.LLMRequest
		check func(t *testing.T, params *openai.ChatCompletionNewParams, err error)
	}{
		"error: empty messages": {
			req: &model.LLMRequest{
				Contents: []*genai.Content{},
				Config:   &genai.GenerateContentConfig{},
			},
			check: func(t *testing.T, params *openai.ChatCompletionNewParams, err error) {
				t.Helper()

				if err == nil {
					t.Fatalf("chatCompletionParams() error = nil, want non-nil")
				}
				if params != nil {
					t.Fatalf("params = %+v, want nil on error", params)
				}
			},
		},

		"success: maps config to params": {
			req: &model.LLMRequest{
				Contents: genai.Text("ping"),
				Config: &genai.GenerateContentConfig{
					Temperature:      &temp,
					TopP:             &topP,
					MaxOutputTokens:  maxTokens,
					CandidateCount:   candidateCount,
					StopSequences:    []string{"STOP"},
					Seed:             &seed,
					Logprobs:         &logprobs,
					FrequencyPenalty: &freqPenalty,
					PresencePenalty:  &presPenalty,
					Tools:            []*genai.Tool{tool},
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{
							Mode:                 genai.FunctionCallingConfigModeAny,
							AllowedFunctionNames: []string{"lookup_weather"},
						},
					},
				},
			},
			check: func(t *testing.T, params *openai.ChatCompletionNewParams, err error) {
				t.Helper()

				if err != nil {
					t.Fatalf("chatCompletionParams() unexpected error: %v", err)
				}
				if params == nil {
					t.Fatal("chatCompletionParams() params nil")
				}

				if params.Model != "gpt-4o" {
					t.Fatalf("model = %q, want %q", params.Model, "gpt-4o")
				}

				if got := len(params.Messages); got != 1 {
					t.Fatalf("messages len = %d, want 1", got)
				}
				user := params.Messages[0].OfUser
				if user == nil || user.Content.OfString.Value != "ping" {
					t.Fatalf("user message = %+v", user)
				}

				if !params.Temperature.Valid() || params.Temperature.Value != float64(temp) {
					t.Fatalf("temperature = %+v, want %v", params.Temperature, temp)
				}
				if !params.TopP.Valid() || params.TopP.Value != float64(topP) {
					t.Fatalf("topP = %+v, want %v", params.TopP, topP)
				}
				if !params.MaxTokens.Valid() || params.MaxTokens.Value != int64(maxTokens) {
					t.Fatalf("maxTokens = %+v, want %d", params.MaxTokens, maxTokens)
				}
				if !params.MaxCompletionTokens.Valid() || params.MaxCompletionTokens.Value != int64(maxTokens) {
					t.Fatalf("maxCompletionTokens = %+v, want %d", params.MaxCompletionTokens, maxTokens)
				}
				if !params.N.Valid() || params.N.Value != int64(candidateCount) {
					t.Fatalf("candidate count = %+v, want %d", params.N, candidateCount)
				}
				if diff := cmp.Diff([]string{"STOP"}, params.Stop.OfStringArray); diff != "" {
					t.Fatalf("stop sequences diff (-want +got):\n%s", diff)
				}
				if !params.Seed.Valid() || params.Seed.Value != int64(seed) {
					t.Fatalf("seed = %+v, want %d", params.Seed, seed)
				}
				if !params.Logprobs.Valid() || !params.TopLogprobs.Valid() || params.TopLogprobs.Value != int64(logprobs) {
					t.Fatalf("logprobs/topLogprobs = %+v/%+v, want %d", params.Logprobs, params.TopLogprobs, logprobs)
				}
				if !params.FrequencyPenalty.Valid() || params.FrequencyPenalty.Value != float64(freqPenalty) {
					t.Fatalf("frequency penalty = %+v, want %v", params.FrequencyPenalty, freqPenalty)
				}
				if !params.PresencePenalty.Valid() || params.PresencePenalty.Value != float64(presPenalty) {
					t.Fatalf("presence penalty = %+v, want %v", params.PresencePenalty, presPenalty)
				}

				if got := len(params.Tools); got != 1 {
					t.Fatalf("tools len = %d, want 1", got)
				}
				if params.ToolChoice.OfFunctionToolChoice == nil {
					t.Fatalf("tool choice not set: %+v", params.ToolChoice)
				}
				if fn := params.ToolChoice.OfFunctionToolChoice.Function.Name; fn != "lookup_weather" {
					t.Fatalf("tool choice name = %q, want %q", fn, "lookup_weather")
				}
			},
		},

		"success: uses request model override": {
			req: &model.LLMRequest{
				Model:    "gpt-4.1-mini",
				Contents: genai.Text("override"),
				Config:   &genai.GenerateContentConfig{},
			},
			check: func(t *testing.T, params *openai.ChatCompletionNewParams, err error) {
				t.Helper()

				if err != nil {
					t.Fatalf("chatCompletionParams() unexpected error: %v", err)
				}
				if params.Model != "gpt-4.1-mini" {
					t.Fatalf("model override = %q, want %q", params.Model, "gpt-4.1-mini")
				}
			},
		},
	}

	for name, tc := range tests {
		tc := tc
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			llm := &openAILLM{name: "gpt-4o"}
			params, err := llm.chatCompletionParams(tc.req)

			tc.check(t, params, err)
		})
	}
}
