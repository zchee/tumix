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
	"encoding/json"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

func TestMapAnthropicFinishReason(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		in   anthropic.BetaStopReason
		want genai.FinishReason
	}{
		"stop":       {anthropic.BetaStopReasonStopSequence, genai.FinishReasonStop},
		"end_turn":   {anthropic.BetaStopReasonEndTurn, genai.FinishReasonStop},
		"max_tokens": {anthropic.BetaStopReasonMaxTokens, genai.FinishReasonMaxTokens},
		"tool":       {anthropic.BetaStopReasonToolUse, genai.FinishReasonOther},
		"unknown":    {anthropic.BetaStopReason("mystery"), genai.FinishReasonUnspecified},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := mapAnthropicFinishReason(tt.in); got != tt.want {
				t.Fatalf("mapAnthropicFinishReason(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestAccText(t *testing.T) {
	t.Parallel()

	t.Run("nil message", func(t *testing.T) {
		if got := AccText(nil); got != "" {
			t.Fatalf("AccText(nil) = %q, want empty", got)
		}
	})

	t.Run("mixed blocks", func(t *testing.T) {
		raw := `{"content":[{"type":"tool_use"},{"type":"text","text":"Hello"},{"type":"text","text":" world"}]}`
		var msg anthropic.BetaMessage
		if err := json.Unmarshal([]byte(raw), &msg); err != nil {
			t.Fatalf("unmarshal message: %v", err)
		}

		if got := AccText(&msg); got != "Hello world" {
			t.Fatalf("AccText() = %q, want %q", got, "Hello world")
		}
	})
}

func TestAnthropicMessageToLLMResponseToolCall(t *testing.T) {
	t.Parallel()

	raw := `{"content":[{"type":"tool_use","id":"call-1","name":"search","input":{"q":"hi"}}],"usage":{"input_tokens":2,"output_tokens":3},"stop_reason":"tool_use"}`
	var msg anthropic.BetaMessage
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	got, err := AnthropicMessageToLLMResponse(&msg)
	if err != nil {
		t.Fatalf("AnthropicMessageToLLMResponse error: %v", err)
	}
	if len(got.Content.Parts) != 1 || got.Content.Parts[0].FunctionCall == nil {
		t.Fatalf("function call missing: %+v", got.Content.Parts)
	}
	if got.FinishReason != genai.FinishReasonOther {
		t.Fatalf("finish reason = %v, want other", got.FinishReason)
	}
	if got.UsageMetadata == nil || got.UsageMetadata.PromptTokenCount != 2 || got.UsageMetadata.CandidatesTokenCount != 3 {
		t.Fatalf("usage metadata mismatch: %+v", got.UsageMetadata)
	}
}
