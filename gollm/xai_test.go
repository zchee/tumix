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
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/gollm/xai"
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
	"github.com/zchee/tumix/testing/rr"
)

func TestXAILLM_Generate(t *testing.T) {
	tests := map[string]struct {
		modelName string
		req       *model.LLMRequest
		want      *model.LLMResponse
		wantErr   bool
	}{
		"ok": {
			modelName: "grok-4-1-fast-non-reasoning",
			req: &model.LLMRequest{
				Contents: genai.Text("What is the capital of France?"),
				Config:   &genai.GenerateContentConfig{},
			},
			want: &model.LLMResponse{
				Content: genai.NewContentFromText("Paris", genai.RoleModel),
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					CachedContentTokenCount: 174,
					CandidatesTokenCount:    1,
					PromptTokenCount:        175,
					ThoughtsTokenCount:      0,
					TotalTokenCount:         176,
				},
				CustomMetadata: map[string]any{
					"xai_finish_reason":      "REASON_STOP",
					"xai_system_fingerprint": "fp_174298dd8e",
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			apiKey, managementKey := xaiTestKeys(t)
			opts := xai.DefaultClientOptions()

			conn, cleanup := rr.NewGRPCConn(t, "xai", xai.APIHost, xai.BuildDialOptions(opts, apiKey)...)
			t.Cleanup(cleanup)

			mgmtConn, cleanup2 := rr.NewGRPCConn(t, "xai-management", xai.ManagementAPIHost, xai.BuildDialOptions(opts, managementKey)...)
			t.Cleanup(cleanup2)

			llm, err := NewXAILLM(t.Context(), apiKey, tt.modelName, nil,
				xai.WithAPIConn(conn),
				xai.WithManagementConn(mgmtConn),
			)
			if err != nil {
				t.Fatalf("NewXAILLM() error = %v", err)
			}

			var got *model.LLMResponse
			for resp, err := range llm.GenerateContent(t.Context(), tt.req, false) {
				if err != nil {
					if xErr, ok := xai.AsError(err); ok {
						err = xErr.Unwrap()
					}
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

func TestXAILLM_GenerateStream(t *testing.T) {
	tests := map[string]struct {
		modelName string
		req       *model.LLMRequest
		want      string
		wantErr   bool
	}{
		"ok": {
			modelName: "grok-4-1-fast-non-reasoning",
			req: &model.LLMRequest{
				Contents: genai.Text("What is the capital of France? One word."),
				Config:   &genai.GenerateContentConfig{},
			},
			want: "Paris",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			apiKey, managementKey := xaiTestKeys(t)
			opts := xai.DefaultClientOptions()

			conn, cleanup := rr.NewGRPCConn(t, "xai", xai.APIHost, xai.BuildDialOptions(opts, apiKey)...)
			t.Cleanup(cleanup)

			mgmtConn, cleanup2 := rr.NewGRPCConn(t, "xai-management", xai.ManagementAPIHost, xai.BuildDialOptions(opts, managementKey)...)
			t.Cleanup(cleanup2)

			llm, err := NewXAILLM(t.Context(), apiKey, tt.modelName, nil,
				xai.WithAPIConn(conn),
				xai.WithManagementConn(mgmtConn),
			)
			if err != nil {
				t.Fatalf("NewXAILLM() error = %v", err)
			}

			got, err := readResponse(llm.GenerateContent(t.Context(), tt.req, true))
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if diff := cmp.Diff(tt.want, got.PartialText); diff != "" {
				t.Errorf("GenerateStream() = %v, want %v\ndiff(-want +got):\n%v", got.PartialText, tt.want, diff)
			}

			// Since we are expecting GenerateStream to aggregate partial events, the text should be the same
			if diff := cmp.Diff(tt.want, got.FinalText); diff != "" {
				t.Errorf("GenerateStream() = %v, want %v\ndiff(-want +got):\n%v", got.FinalText, tt.want, diff)
			}
		})
	}
}

func xaiTestKeys(t *testing.T) (apiKey, managementKey string) {
	t.Helper()

	apiKey = os.Getenv("XAI_API_KEY")
	managementKey = os.Getenv("XAI_MANAGEMENT_KEY")

	if rr.Replaying() {
		if apiKey == "" {
			apiKey = "test-key"
		}
		if managementKey == "" {
			managementKey = apiKey
		}
		return apiKey, managementKey
	}

	if apiKey == "" {
		t.Skip("XAI_API_KEY not set; rerun with -record and valid credentials to refresh goldens")
	}
	if managementKey == "" {
		managementKey = apiKey
	}

	return apiKey, managementKey
}

func TestGenAI2XAIChatOptions(t *testing.T) {
	t.Parallel()

	temp := float32(0.7)
	topP := float32(0.8)
	maxTokens := int32(32)
	stop := []string{"END"}

	cfg := &genai.GenerateContentConfig{
		Temperature:      &temp,
		TopP:             &topP,
		MaxOutputTokens:  maxTokens,
		StopSequences:    stop,
		ResponseLogprobs: true,
	}

	req := &xaipb.GetCompletionsRequest{}
	session := &xai.ChatSession{}

	opt := adapter.GenAI2XAIChatOptions(cfg)
	if opt == nil {
		t.Fatal("genAI2XAIChatOptions returned nil")
	}

	opt(req, session)

	if req.Temperature == nil || req.GetTemperature() != temp {
		t.Fatalf("temperature = %v, want %v", req.GetTemperature(), temp)
	}
	if req.TopP == nil || req.GetTopP() != topP {
		t.Fatalf("topP = %v, want %v", req.GetTopP(), topP)
	}
	if req.MaxTokens == nil || req.GetMaxTokens() != maxTokens {
		t.Fatalf("maxTokens = %v, want %v", req.GetMaxTokens(), maxTokens)
	}
	if got := req.GetStop(); !cmp.Equal(got, stop) {
		t.Fatalf("stop sequences = %v, want %v", got, stop)
	}
	if !req.GetLogprobs() {
		t.Fatal("logprobs not set")
	}
}
