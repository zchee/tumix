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

func TestAnthropicLLMRecordReplay(t *testing.T) {
	t.Parallel()

	const anthropicReplayAddr = "127.0.0.1:28081"
	const anthropicReplayBaseURL = "http://" + anthropicReplayAddr

	if *rr.Record {
		cleanup := startStubHTTP(t, anthropicReplayAddr, []string{"/v1/messages", "/messages"}, `{
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
}`)
		t.Cleanup(cleanup)
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
