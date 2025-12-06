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

package adapter_test

import (
	json "encoding/json/v2"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/internal/adapter"
)

func TestAnthropicBetaMessageToLLMResponse(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		raw  string
		want *model.LLMResponse
	}{
		"text and tool call": {
			raw: `{
  "id": "msg_1",
  "container": {
    "id": "cont_1",
    "expires_at": "2024-01-01T00:00:00Z",
    "skills": [
      {"skill_id": "sk1", "type": "anthropic", "version": "latest"}
    ]
  },
  "content": [
    {"type": "text", "text": "The capital is ", "citations": []},
    {"type": "tool_use", "id": "tool-1", "name": "get_weather", "input": {"city": "Paris"}}
  ],
  "context_management": {
    "applied_edits": [
      {"type": "clear_tool_uses_20250919", "cleared_tool_uses": 0, "cleared_input_tokens": 0}
    ]
  },
  "model": "claude-3-5-sonnet-20241022",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": "",
  "type": "message",
  "usage": {
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "input_tokens": 5,
    "output_tokens": 7,
    "server_tool_use": {"web_fetch_requests": 0, "web_search_requests": 0},
    "service_tier": "standard"
  }
}`,
			want: &model.LLMResponse{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						genai.NewPartFromText("The capital is "),
						{
							FunctionCall: &genai.FunctionCall{
								ID:   "tool-1",
								Name: "get_weather",
								Args: map[string]any{"city": "Paris"},
							},
						},
					},
				},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     5,
					CandidatesTokenCount: 7,
					TotalTokenCount:      12,
				},
				FinishReason: genai.FinishReasonStop,
			},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			var msg anthropic.BetaMessage
			if err := json.Unmarshal([]byte(tt.raw), &msg); err != nil {
				t.Fatalf("unmarshal anthropic beta message: %v", err)
			}

			got, err := adapter.AnthropicBetaMessageToLLMResponse(&msg)
			if err != nil {
				t.Fatalf("AnthropicBetaMessageToLLMResponse() err = %v", err)
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Fatalf("AnthropicBetaMessageToLLMResponse diff (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGenAIToAnthropicBetaMessages(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		system   *genai.Content
		contents []*genai.Content
	}{
		"system and user message": {
			system:   genai.NewContentFromText("system guidance", "system"),
			contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			systemBlocks, msgs, err := adapter.GenAIToAnthropicBetaMessages(tc.system, tc.contents)
			if err != nil {
				t.Fatalf("GenAIToAnthropicBetaMessages() error = %v", err)
			}

			if len(systemBlocks) != 1 || systemBlocks[0].Text != "system guidance" {
				t.Fatalf("system blocks got %+v", systemBlocks)
			}
			if len(msgs) != 1 {
				t.Fatalf("messages len = %d, want 1", len(msgs))
			}
			if msgs[0].Role != anthropic.BetaMessageParamRoleUser {
				t.Fatalf("msg role = %s, want user", msgs[0].Role)
			}
			if len(msgs[0].Content) != 1 {
				t.Fatalf("msg content len = %d, want 1", len(msgs[0].Content))
			}
		})
	}
}
