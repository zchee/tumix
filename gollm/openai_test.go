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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3/option"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/testing/rr"
)

func TestOpenAILLMRecordReplay(t *testing.T) {
	t.Parallel()

	const openaiReplayAddr = "127.0.0.1:28082"
	const openaiReplayBaseURL = "http://" + openaiReplayAddr + "/v1"

	if *rr.Record {
		ln, err := net.Listen("tcp", openaiReplayAddr)
		if err != nil {
			t.Skipf("unable to listen for openai stub: %v", err)
		}

		mux := http.NewServeMux()
		mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/chat/completions" {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
  "id": "chatcmpl-rreplay",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Paris"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 2,
    "total_tokens": 10
  }
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

	llm, err := NewOpenAILLM(t.Context(), AuthMethodAPIKey("test-key"), "gpt-4o",
		option.WithHTTPClient(httpClient),
		option.WithBaseURL(openaiReplayBaseURL),
	)
	if err != nil {
		t.Fatalf("NewOpenAILLM() error = %v", err)
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
			PromptTokenCount:     8,
			CandidatesTokenCount: 2,
			TotalTokenCount:      10,
		},
		FinishReason: genai.FinishReasonStop,
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("GenerateContent() diff (-want +got):\n%s", diff)
	}
}
