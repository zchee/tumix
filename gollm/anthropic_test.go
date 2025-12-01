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
	"net"
	"net/http"
	"strings"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/internal/adapter"
	"github.com/zchee/tumix/testing/rr"
)

func TestAnthropicLLMBuildParams(t *testing.T) {
	t.Parallel()

	temp := float32(0.4)
	topP := float32(0.7)
	topK := float32(8)

	tool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{Name: "lookup_weather", Description: "Return current weather"},
		},
	}

	tests := map[string]struct {
		req     *model.LLMRequest
		wantErr string
		check   func(t *testing.T, params *anthropic.MessageNewParams)
	}{
		"error: no messages": {
			req: &model.LLMRequest{
				Contents: nil,
				Config:   &genai.GenerateContentConfig{},
			},
			wantErr: "no messages",
		},

		"success: defaults max tokens": {
			req: &model.LLMRequest{
				Contents: genai.Text("hello"),
				Config: &genai.GenerateContentConfig{
					MaxOutputTokens: 0,
					StopSequences:   []string{"cut"},
				},
			},
			check: func(t *testing.T, params *anthropic.MessageNewParams) {
				t.Helper()

				if params.MaxTokens != 1024 {
					t.Fatalf("MaxTokens = %d, want 1024", params.MaxTokens)
				}
				if diff := cmp.Diff([]string{"cut"}, params.StopSequences); diff != "" {
					t.Fatalf("stop sequences diff (-want +got):\n%s", diff)
				}
				if string(params.Model) != "claude-3-haiku" {
					t.Fatalf("model = %q, want %q", params.Model, "claude-3-haiku")
				}
			},
		},

		"success: maps config and tools": {
			req: &model.LLMRequest{
				Model:    "claude-3-5-sonnet",
				Contents: genai.Text("hi"),
				Config: &genai.GenerateContentConfig{
					SystemInstruction: genai.NewContentFromText("stay short", "system"),
					MaxOutputTokens:   64,
					Temperature:       &temp,
					TopP:              &topP,
					TopK:              &topK,
					StopSequences:     []string{"END"},
					Tools:             []*genai.Tool{tool},
					ToolConfig: &genai.ToolConfig{
						FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone},
					},
				},
			},
			check: func(t *testing.T, params *anthropic.MessageNewParams) {
				t.Helper()

				if params.MaxTokens != 64 {
					t.Fatalf("MaxTokens = %d, want 64", params.MaxTokens)
				}
				if !params.Temperature.Valid() || params.Temperature.Value != float64(temp) {
					t.Fatalf("temperature = %+v, want %v", params.Temperature, temp)
				}
				if !params.TopP.Valid() || params.TopP.Value != float64(topP) {
					t.Fatalf("topP = %+v, want %v", params.TopP, topP)
				}
				if !params.TopK.Valid() || params.TopK.Value != int64(topK) {
					t.Fatalf("topK = %+v, want %v", params.TopK, topK)
				}
				if diff := cmp.Diff([]string{"END"}, params.StopSequences); diff != "" {
					t.Fatalf("stop sequences diff (-want +got):\n%s", diff)
				}
				if len(params.System) == 0 || params.System[0].Text != "stay short" {
					t.Fatalf("system prompt = %+v, want text 'stay short'", params.System)
				}
				if got := len(params.Tools); got != 1 {
					t.Fatalf("tools len = %d, want 1", got)
				}
				if params.ToolChoice.OfNone == nil {
					t.Fatalf("tool choice not set to none: %+v", params.ToolChoice)
				}
				if string(params.Model) != "claude-3-5-sonnet" {
					t.Fatalf("model override = %q, want %q", params.Model, "claude-3-5-sonnet")
				}
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			system, msgs, convErr := adapter.GenAIToAnthropicMessages(tc.req.Config.SystemInstruction, tc.req.Contents)
			if convErr != nil {
				t.Fatalf("GenaiToAnthropicMessages() error: %v", convErr)
			}

			params, err := (&anthropicLLM{name: "claude-3-haiku"}).buildParams(tc.req, system, msgs)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("buildParams() error = %v, want substring %q", err, tc.wantErr)
				}
				if params != nil {
					t.Fatalf("params = %+v, want nil on error", params)
				}
				return
			}

			if err != nil {
				t.Fatalf("buildParams() unexpected error: %v", err)
			}
			tc.check(t, params)
		})
	}
}

func TestAnthropicLLMRecordReplay(t *testing.T) {
	t.Parallel()

	const anthropicReplayAddr = "127.0.0.1:28081"
	const anthropicReplayBaseURL = "http://" + anthropicReplayAddr

	if *rr.Record {
		ln, err := net.Listen("tcp", anthropicReplayAddr)
		if err != nil {
			t.Skipf("unable to listen for anthropic stub: %v", err)
		}

		mux := http.NewServeMux()
		mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/messages" && r.URL.Path != "/messages" {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
  "id": "msg_rr_test",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-haiku",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 9,
    "output_tokens": 2
  },
  "content": [
    {
      "type": "text",
      "text": "Paris"
    }
  ]
}`))
		})

		srv := &http.Server{Handler: mux}
		go func() {
			_ = srv.Serve(ln)
		}()
		t.Cleanup(func() {
			_ = srv.Shutdown(context.Background())
			_ = ln.Close()
		})
	}

	httpClient, cleanup, _ := rr.NewHTTPClient(t, func(r *rr.Recorder) {})
	t.Cleanup(cleanup)

	llm, err := NewAnthropicLLM(t.Context(), AuthMethodAPIKey("test-key"), "claude-3-haiku",
		option.WithBaseURL(anthropicReplayBaseURL),
		option.WithHTTPClient(httpClient),
	)
	if err != nil {
		t.Fatalf("NewAnthropicLLM() error = %v", err)
	}

	req := &model.LLMRequest{
		Contents: genai.Text("What is the capital of France?"),
		Config:   &genai.GenerateContentConfig{},
	}

	var got *model.LLMResponse
	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent() unexpected error: %v", err)
		}
		got = resp
	}

	want := &model.LLMResponse{
		Content: genai.NewContentFromText("Paris", genai.RoleModel),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     9,
			CandidatesTokenCount: 2,
			TotalTokenCount:      11,
		},
		FinishReason: genai.FinishReasonStop,
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("GenerateContent() diff (-want +got):\n%s", diff)
	}
}
