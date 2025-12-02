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

	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/testing/rr"
)

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

			apiKey := AuthMethod(nil)
			if rr.Replaying() {
				apiKey = AuthMethodAPIKey("test-key")
			}

			llm, err := NewAnthropicLLM(t.Context(), apiKey, tt.modelName,
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

			apiKey := AuthMethod(nil)
			if rr.Replaying() {
				apiKey = AuthMethodAPIKey("test-key")
			}

			llm, err := NewAnthropicLLM(t.Context(), apiKey, tt.modelName,
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
