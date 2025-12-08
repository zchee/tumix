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
	"errors"
	"fmt"
	"slices"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// AnthropicMessageToLLMResponse converts an Anthropic beta message into an ADK LLM response.
func AnthropicMessageToLLMResponse(msg *anthropic.BetaMessage) (*model.LLMResponse, error) {
	if msg == nil {
		return nil, errors.New("nil anthropic beta message")
	}

	parts := make([]*genai.Part, 0, len(msg.Content))
	for block := range slices.Values(msg.Content) {
		switch v := block.AsAny().(type) {
		case anthropic.BetaTextBlock:
			parts = append(parts, genai.NewPartFromText(v.Text))
		case anthropic.BetaToolUseBlock:
			args := map[string]any{}
			switch inp := v.Input.(type) {
			case nil:
			case map[string]any:
				args = inp
			default:
				raw, err := json.Marshal(inp)
				if err != nil {
					return nil, fmt.Errorf("marshal tool input: %w", err)
				}
				if err := json.Unmarshal(raw, &args); err != nil {
					return nil, fmt.Errorf("unmarshal tool input: %w", err)
				}
			}
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   v.ID,
					Name: v.Name,
					Args: args,
				},
			})
		}
	}

	usage := msg.Usage
	llmUsage := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(usage.InputTokens),
		CandidatesTokenCount: int32(usage.OutputTokens),
		TotalTokenCount:      int32(usage.InputTokens + usage.OutputTokens),
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  genai.RoleModel,
			Parts: parts,
		},
		UsageMetadata: llmUsage,
		FinishReason:  mapAnthropicFinishReason(msg.StopReason),
	}, nil
}

func mapAnthropicFinishReason(reason anthropic.BetaStopReason) genai.FinishReason {
	switch reason {
	case anthropic.BetaStopReasonStopSequence, anthropic.BetaStopReasonEndTurn:
		return genai.FinishReasonStop
	case anthropic.BetaStopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	case anthropic.BetaStopReasonToolUse:
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonUnspecified
	}
}

// AccText concatenates all text blocks from an Anthropic Beta message.
func AccText(msg *anthropic.BetaMessage) string {
	if msg == nil {
		return ""
	}
	var sb strings.Builder
	for block := range slices.Values(msg.Content) {
		if v, ok := block.AsAny().(anthropic.BetaTextBlock); ok {
			sb.WriteString(v.Text)
		}
	}
	return sb.String()
}
