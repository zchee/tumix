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
	json "encoding/json/v2"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

func TestGenAIToResponsesInputRolesAndTextFlush(t *testing.T) {
	t.Parallel()

	items, err := GenAIToResponsesInput([]*genai.Content{
		{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				genai.NewPartFromText("Hello, "),
				genai.NewPartFromText("world"),
			},
		},
		{
			Role: "developer",
			Parts: []*genai.Part{
				genai.NewPartFromText("dev note"),
			},
		},
	})
	if err != nil {
		t.Fatalf("GenAIToResponsesInput() error = %v", err)
	}

	if len(items) != 2 {
		t.Fatalf("items len = %d, want 2", len(items))
	}

	got := items[0].OfMessage
	if got == nil {
		t.Fatalf("first item is nil")
	}
	if got.Role != responses.EasyInputMessageRoleUser {
		t.Fatalf("role = %v, want user", got.Role)
	}
	if got.Content.OfString.Or("") != "Hello, world" {
		t.Fatalf("content = %q, want %q", got.Content.OfString.Or(""), "Hello, world")
	}

	if items[1].OfMessage == nil || items[1].OfMessage.Role != responses.EasyInputMessageRoleDeveloper {
		t.Fatalf("second role = %v, want developer", items[1].OfMessage.Role)
	}
}

func TestGenAIToResponsesInputFunctionCallAndResponse(t *testing.T) {
	t.Parallel()

	items, err := GenAIToResponsesInput([]*genai.Content{
		{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				{
					FunctionCall: &genai.FunctionCall{
						ID:   "",
						Name: "lookup",
						Args: map[string]any{"q": "Paris"},
					},
				},
				{
					FunctionResponse: &genai.FunctionResponse{
						ID:       "",
						Name:     "lookup",
						Response: map[string]any{"city": "Paris"},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("GenAIToResponsesInput() error = %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("items len = %d, want 2", len(items))
	}

	call := items[0].OfFunctionCall
	if call == nil || call.Name != "lookup" {
		t.Fatalf("function call missing: %+v", call)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
		t.Fatalf("parse args: %v", err)
	}
	if args["q"] != "Paris" {
		t.Fatalf("args = %+v", args)
	}

	resp := items[1].OfFunctionCallOutput
	if resp == nil {
		t.Fatalf("function response missing")
	}
	if resp.CallID == "" {
		t.Fatalf("CallID empty")
	}
	if out := resp.Output.OfString; out.Or("") == "" {
		t.Fatalf("output empty: %+v", resp.Output)
	}
}

func TestGenAIToResponsesInputErrors(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		contents []*genai.Content
	}{
		"missing function name": {
			contents: []*genai.Content{{
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{}}},
			}},
		},
		"missing function response name": {
			contents: []*genai.Content{{
				Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Response: map[string]any{}}}},
			}},
		},
		"unsupported part": {
			contents: []*genai.Content{{
				Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "text/plain"}}},
			}},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := GenAIToResponsesInput(tt.contents); err == nil {
				t.Fatalf("expected error")
			}
		})
	}
}

func TestToEasyRoleFallback(t *testing.T) {
	t.Parallel()

	cases := map[string]responses.EasyInputMessageRole{
		genai.RoleUser:  responses.EasyInputMessageRoleUser,
		genai.RoleModel: responses.EasyInputMessageRoleAssistant,
		"system":        responses.EasyInputMessageRoleSystem,
		"developer":     responses.EasyInputMessageRoleDeveloper,
		"unknown":       responses.EasyInputMessageRoleUser,
	}

	for role, want := range cases {
		if got := toEasyRole(role); got != want {
			t.Fatalf("toEasyRole(%q) = %v, want %v", role, got, want)
		}
	}
}
