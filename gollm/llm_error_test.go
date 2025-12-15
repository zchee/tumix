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

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestAnthropicLLM_GenerateContent_NoMessages(t *testing.T) {
	t.Parallel()

	llm := &anthropicLLM{name: "claude-haiku-4-5"}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
	}

	if _, err := llm.buildParams(req, nil, nil); err == nil {
		t.Fatalf("expected error for empty messages")
	}
}

func TestOpenAILLM_GenerateContent_NoInput(t *testing.T) {
	t.Parallel()

	llm := &openAILLM{name: "gpt-4o"}
	req := &model.LLMRequest{
		Config: &genai.GenerateContentConfig{},
	}

	var seenErr error
	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		seenErr = err
		break
	}
	if seenErr == nil {
		t.Fatalf("expected error for empty input items")
	}
}

func TestXAILLM_GenerateContent_InvalidMessages(t *testing.T) {
	t.Parallel()

	llm := &xaiLLM{name: "grok-mini"}
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "text/plain",
						},
					},
				},
			},
		},
		Config: &genai.GenerateContentConfig{},
	}

	var seenErr error
	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		seenErr = err
		break
	}
	if seenErr == nil {
		t.Fatalf("expected error converting messages, got nil")
	}
}
