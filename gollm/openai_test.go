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
	"github.com/google/go-cmp/cmp/cmpopts"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/testing/rr"
)

func TestOpenAIResponseParamsLogprobs(t *testing.T) {
	t.Parallel()

	llm := &openAILLM{name: "gpt-4o"}
	logprobs := int32(3)
	cfg := &genai.GenerateContentConfig{
		Temperature:      openai.Ptr(float32(0.2)),
		TopP:             openai.Ptr(float32(0.7)),
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

	approxOpts := cmpopts.EquateApprox(0, 1e-6)
	if diff := cmp.Diff(params.Temperature.Or(0), 0.2, approxOpts); diff != "" {
		t.Fatalf("Temperature mismatch (-got +want):\n%s", diff)
	}
	if diff := cmp.Diff(params.TopP.Or(0), 0.7, approxOpts); diff != "" {
		t.Fatalf("TopP mismatch (-got +want):\n%s", diff)
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

func TestOpenAILLM_Generate(t *testing.T) {
	tests := map[string]struct {
		modelName string
		req       *model.LLMRequest
		want      *model.LLMResponse
		wantErr   bool
	}{
		"ok": {
			modelName: "gpt-5.1",
			req: &model.LLMRequest{
				Contents: genai.Text("What is the capital of France?"),
				Config:   &genai.GenerateContentConfig{},
			},
			want: &model.LLMResponse{
				Content: genai.NewContentFromText("The capital of France is Paris.", genai.RoleModel),
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     13,
					CandidatesTokenCount: 17,
					TotalTokenCount:      30,
				},
				FinishReason: genai.FinishReasonStop,
				TurnComplete: true,
			},
		},
		"MultiCandidate": {
			modelName: "gpt-5.1",
			req: &model.LLMRequest{
				Contents: genai.Text("two please"),
				Config: &genai.GenerateContentConfig{
					CandidateCount: 2,
				},
			},
			want: &model.LLMResponse{
				Content: genai.NewContentFromText("2", genai.RoleModel),
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     8,
					CandidatesTokenCount: 11,
					TotalTokenCount:      19,
				},
				FinishReason: genai.FinishReasonStop,
				TurnComplete: true,
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
					"Openai-Organization",
					"Openai-Project",
					"Set-Cookie",
				)
			})
			t.Cleanup(cleanup)

			apiKey := ""
			if rr.Replaying() {
				apiKey = "test-key"
			}

			llm, err := NewOpenAILLM(t.Context(), apiKey, tt.modelName, nil,
				option.WithHTTPClient(httpClient),
			)
			if err != nil {
				t.Fatalf("NewOpenAILLM() error = %v", err)
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

func TestOpenAILLM_GenerateStream(t *testing.T) {
	tests := map[string]struct {
		modelName string
		req       *model.LLMRequest
		want      string
		wantErr   bool
	}{
		"ok": {
			modelName: "gpt-5.1",
			req: &model.LLMRequest{
				Contents: genai.Text("What is the capital of France? Must be One word."),
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
					"Openai-Organization",
					"Openai-Project",
					"Set-Cookie",
				)
			})
			t.Cleanup(cleanup)

			apiKey := ""
			if rr.Replaying() {
				apiKey = "test-key"
			}

			llm, err := NewOpenAILLM(t.Context(), apiKey, tt.modelName, nil,
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
