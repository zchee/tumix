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
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/testing/rr"
)

func TestAnthropicBuildParamsDefaults(t *testing.T) {
	t.Parallel()

	llm := &anthropicLLM{name: "claude-3-haiku"}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{
			Temperature:     anthropic.Ptr(float32(0.5)),
			TopP:            anthropic.Ptr(float32(0.9)),
			TopK:            anthropic.Ptr(float32(3)),
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

	approxOpts := cmpopts.EquateApprox(0, 1e-6)
	if diff := cmp.Diff(params.Temperature.Or(0), 0.5, approxOpts); diff != "" {
		t.Fatalf("Temperature mismatch (-got +want):\n%s", diff)
	}
	if diff := cmp.Diff(params.TopP.Or(0), 0.9, approxOpts); diff != "" {
		t.Fatalf("TopP mismatch (-got +want):\n%s", diff)
	}
	if params.TopK.Or(0) != 3 {
		t.Fatalf("TopK = %v, want 3", params.TopK.Or(0))
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

func TestAnthropicLLM_Generate(t *testing.T) {
	tests := map[string]struct {
		modelName string
		req       *model.LLMRequest
		want      *model.LLMResponse
		wantErr   bool
	}{
		"ok": {
			modelName: "claude-haiku-4-5",
			req: &model.LLMRequest{
				Contents: genai.Text("What is the capital of France?"),
				Config:   &genai.GenerateContentConfig{},
			},
			want: &model.LLMResponse{
				Content: genai.NewContentFromText("The capital of France is Paris.", genai.RoleModel),
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     14,
					CandidatesTokenCount: 10,
					TotalTokenCount:      24,
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			httpClient, cleanup, _ := rr.NewHTTPClient(t, func(r *rr.Recorder) {
				r.RemoveRequestHeaders(
					"X-Api-Key",
				)
				r.RemoveResponseHeaders(
					"Anthropic-Organization-Id",
				)
			})
			t.Cleanup(cleanup)

			apiKey := ""
			if rr.Replaying() {
				apiKey = "test-key"
			}

			llm, err := NewAnthropicLLM(t.Context(), apiKey, tt.modelName, nil,
				option.WithHTTPClient(httpClient),
			)
			if err != nil {
				t.Fatalf("NewAnthropicLLM() error = %v", err)
			}

			var got *model.LLMResponse
			for resp, err := range llm.GenerateContent(t.Context(), tt.req, false) {
				if err != nil {
					t.Fatalf("GenerateContent() unexpected error: %v", err)
				}
				got = resp
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("GenerateContent() diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestAnthropicLLM_GenerateStream(t *testing.T) {
	tests := map[string]struct {
		modelName string
		req       *model.LLMRequest
		want      string
		wantErr   bool
	}{
		"ok": {
			modelName: "claude-haiku-4-5",
			req: &model.LLMRequest{
				Contents: genai.Text("What is the capital of France? One word."),
				Config: &genai.GenerateContentConfig{
					MaxOutputTokens: 2048,
				},
			},
			want: "Paris",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			httpClient, cleanup, _ := rr.NewHTTPClient(t, func(r *rr.Recorder) {
				r.RemoveRequestHeaders(
					"X-Api-Key",
				)
				r.RemoveResponseHeaders(
					"Anthropic-Organization-Id",
				)
			})
			t.Cleanup(cleanup)

			apiKey := ""
			if rr.Replaying() {
				apiKey = "test-key"
			}

			llm, err := NewAnthropicLLM(t.Context(), apiKey, tt.modelName, nil,
				option.WithHTTPClient(httpClient),
			)
			if err != nil {
				t.Fatalf("NewAnthropicLLM() error = %v", err)
			}

			// Transforms the stream into strings, concatenating the text value of the response parts
			got, err := readResponse(llm.GenerateContent(t.Context(), tt.req, true))
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Since we are expecting GenerateStream to aggregate partial events, the text should be the same
			if diff := cmp.Diff(tt.want, got.FinalText); diff != "" {
				t.Errorf("Model.GenerateStream().FinalText = %v, want %v\ndiff(-want +got):\n%v", got.FinalText, tt.want, diff)
			}
		})
	}
}
