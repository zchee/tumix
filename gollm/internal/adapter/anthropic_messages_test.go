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

package adapter

import (
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestGenAIToAnthropicMessages(t *testing.T) {
	t.Parallel()

	system := &genai.Content{Role: "system", Parts: []*genai.Part{genai.NewPartFromText("sys")}}
	contents := []*genai.Content{
		{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				genai.NewPartFromText("hi"),
				{
					FunctionCall: &genai.FunctionCall{
						Name: "lookup",
						Args: map[string]any{"q": "x"},
					},
				},
			},
		},
		{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				{
					FunctionResponse: &genai.FunctionResponse{
						Name: "lookup",
						Response: map[string]any{
							"ok": true,
						},
					},
				},
			},
		},
	}

	sysBlocks, msgs, err := GenAIToAnthropicMessages(system, contents)
	if err != nil {
		t.Fatalf("GenAIToAnthropicMessages() error = %v", err)
	}
	if len(sysBlocks) != 1 || sysBlocks[0].Text != "sys" {
		t.Fatalf("system blocks = %+v", sysBlocks)
	}
	if len(msgs) != 2 {
		t.Fatalf("messages len = %d, want 2", len(msgs))
	}
	if msgs[0].Role != anthropic.BetaMessageParamRoleUser || msgs[1].Role != anthropic.BetaMessageParamRoleAssistant {
		t.Fatalf("roles = %v %v", msgs[0].Role, msgs[1].Role)
	}
}

func TestGenAIToAnthropicMessagesErrors(t *testing.T) {
	t.Parallel()

	tests := map[string]*genai.Content{
		"function call missing name": {
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						Args: map[string]any{},
					},
				},
			},
		},
		"function response missing name": {
			Parts: []*genai.Part{
				{
					FunctionResponse: &genai.FunctionResponse{
						Response: map[string]any{},
					},
				},
			},
		},
		"unsupported part": {
			Parts: []*genai.Part{
				{
					InlineData: &genai.Blob{
						MIMEType: "text/plain",
					},
				},
			},
		},
		"empty parts": {
			Parts: []*genai.Part{},
		},
	}

	for name, content := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			if _, _, err := GenAIToAnthropicMessages(nil, []*genai.Content{content}); err == nil {
				t.Fatalf("expected error")
			}
		})
	}
}

func TestGenAIToAnthropicMessagesToolUseIDs(t *testing.T) {
	t.Parallel()

	_, msgs, err := GenAIToAnthropicMessages(nil, []*genai.Content{
		{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						Name: "fn",
						Args: map[string]any{
							"a": 1,
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("GenAIToAnthropicMessages() error = %v", err)
	}
	call := msgs[0].Content[0].OfToolUse
	if call == nil {
		t.Fatalf("tool use missing")
	}
	if call.ID == "" {
		t.Fatalf("tool id not set")
	}
	if diff := cmp.Diff("fn", call.Name); diff != "" {
		t.Fatalf("name diff: %s", diff)
	}
}
