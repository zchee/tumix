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

package model

import (
	json "encoding/json/v2"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestAnthropicMessageToLLMResponse_TextAndToolCall(t *testing.T) {
	raw := []byte(`{
        "content": [
            {"type": "text", "text": "The capital is "},
            {"type": "tool_use", "id": "tool-1", "name": "get_weather", "input": {"city": "Paris"}}
        ],
        "model": "claude-3",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "type": "message",
        "usage": {"input_tokens": 5, "output_tokens": 7}
    }`)

	var msg anthropic.Message
	if err := json.Unmarshal(raw, &msg); err != nil {
		t.Fatalf("unmarshal anthropic message: %v", err)
	}

	got, err := anthropicMessageToLLMResponse(&msg)
	if err != nil {
		t.Fatalf("anthropicMessageToLLMResponse() err = %v", err)
	}

	want := &model.LLMResponse{
		Content: &genai.Content{
			Role: string(genai.RoleModel),
			Parts: []*genai.Part{
				genai.NewPartFromText("The capital is "),
				{FunctionCall: &genai.FunctionCall{ID: "tool-1", Name: "get_weather", Args: map[string]any{"city": "Paris"}}},
			},
		},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     5,
			CandidatesTokenCount: 7,
			TotalTokenCount:      12,
		},
		FinishReason: genai.FinishReasonStop,
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("anthropicMessageToLLMResponse diff (-want +got):\n%s", diff)
	}
}

func TestGenaiToAnthropicMessages_SystemAndUser(t *testing.T) {
	sys := genai.NewContentFromText("system guidance", "system")
	contents := []*genai.Content{
		genai.NewContentFromText("hello", genai.RoleUser),
	}

	systemBlocks, msgs, err := genaiToAnthropicMessages(sys, contents)
	if err != nil {
		t.Fatalf("genaiToAnthropicMessages err = %v", err)
	}

	if len(systemBlocks) != 1 || systemBlocks[0].Text != "system guidance" {
		t.Fatalf("system blocks got %+v", systemBlocks)
	}
	if len(msgs) != 1 {
		t.Fatalf("messages len = %d, want 1", len(msgs))
	}
	if msgs[0].Role != anthropic.MessageParamRoleUser {
		t.Fatalf("msg role = %s, want user", msgs[0].Role)
	}
	if len(msgs[0].Content) != 1 {
		t.Fatalf("msg content len = %d, want 1", len(msgs[0].Content))
	}
}
