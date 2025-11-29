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
	"cmp"
	json "encoding/json/v2"
	"fmt"
	"slices"
	"strings"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func toolCallsFromMessage(msg *openai.ChatCompletionMessage) []openai.ChatCompletionMessageToolCallUnion {
	if msg == nil {
		return nil
	}

	if len(msg.ToolCalls) > 0 {
		return msg.ToolCalls
	}

	raw := msg.JSON.FunctionCall.Raw()
	if !msg.JSON.FunctionCall.Valid() || raw == "" {
		return nil
	}

	var legacyFunctionCall struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(raw), &legacyFunctionCall); err != nil {
		return nil
	}
	if legacyFunctionCall.Name == "" {
		return nil
	}

	return []openai.ChatCompletionMessageToolCallUnion{
		{
			Type: string(constant.ValueOf[constant.Function]()),
			Function: openai.ChatCompletionMessageFunctionToolCallFunction{
				Name:      legacyFunctionCall.Name,
				Arguments: legacyFunctionCall.Arguments,
			},
		},
	}
}

func OpenAIResponseToLLM(resp *openai.ChatCompletion) (*model.LLMResponse, error) {
	if resp == nil {
		return nil, fmt.Errorf("nil openai response")
	}
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("empty choices")
	}

	msg := resp.Choices[0].Message
	parts := make([]*genai.Part, 0, 2)
	if msg.Content != "" {
		parts = append(parts, genai.NewPartFromText(msg.Content))
	}

	toolCalls := toolCallsFromMessage(&msg)
	for tc := range slices.Values(toolCalls) {
		var fn openai.ChatCompletionMessageFunctionToolCallFunction
		switch v := tc.AsAny().(type) {
		case openai.ChatCompletionMessageFunctionToolCall:
			fn = v.Function
		default:
			fn = tc.Function
		}
		if fn.Name == "" {
			fn = tc.Function
		}

		parts = append(parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   tc.ID,
				Name: fn.Name,
				Args: parseArgs(fn.Arguments),
			},
		})
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  genai.RoleModel,
		},
		UsageMetadata: openAIUsage(&resp.Usage),
		FinishReason:  mapOpenAIFinishReason(resp.Choices[0].FinishReason),
	}, nil
}

type toolCallState struct {
	id    string
	name  string
	index int64
	args  strings.Builder
}

type OpenAIStreamAggregator struct {
	text         strings.Builder
	toolCalls    []*toolCallState
	finishReason string
	usage        *openai.CompletionUsage
}

func NewOpenAIStreamAggregator() *OpenAIStreamAggregator {
	return &OpenAIStreamAggregator{}
}

func (a *OpenAIStreamAggregator) Process(chunk *openai.ChatCompletionChunk) []*model.LLMResponse {
	if chunk == nil || len(chunk.Choices) == 0 {
		return nil
	}

	choice := chunk.Choices[0]
	if choice.FinishReason != "" {
		a.finishReason = choice.FinishReason
	}
	if chunk.JSON.Usage.Valid() {
		a.usage = openai.Ptr(chunk.Usage)
	}

	var out []*model.LLMResponse
	delta := choice.Delta

	if delta.Content != "" {
		a.text.WriteString(delta.Content)
		out = append(out, &model.LLMResponse{
			Content: &genai.Content{
				Role:  string(genai.RoleModel),
				Parts: []*genai.Part{genai.NewPartFromText(delta.Content)},
			},
			Partial: true,
		})
	}

	for tc := range slices.Values(delta.ToolCalls) {
		state := a.ensureToolCall(tc.Index, tc.ID)
		if tc.Function.Name != "" {
			state.name = tc.Function.Name
		}
		if tc.Function.Arguments != "" {
			state.args.WriteString(tc.Function.Arguments)
		}
		if tc.ID != "" {
			state.id = tc.ID
		}
	}

	return out
}

func (a *OpenAIStreamAggregator) Final() *model.LLMResponse {
	if a.text.Len() == 0 && len(a.toolCalls) == 0 && a.finishReason == "" && a.usage == nil {
		return nil
	}

	parts := make([]*genai.Part, 0, 1+len(a.toolCalls))
	if a.text.Len() > 0 {
		parts = append(parts, genai.NewPartFromText(a.text.String()))
	}

	if len(a.toolCalls) > 0 {
		slices.SortFunc(a.toolCalls, func(a, b *toolCallState) int {
			return cmp.Compare(a.index, b.index)
		})

		for _, tc := range a.toolCalls {
			args := parseArgs(tc.args.String())
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tc.id,
					Name: tc.name,
					Args: args,
				},
			})
		}
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  genai.RoleModel,
		},
		UsageMetadata: openAIUsage(a.usage),
		TurnComplete:  a.finishReason != "",
		FinishReason:  mapOpenAIFinishReason(a.finishReason),
	}
}

func (a *OpenAIStreamAggregator) ensureToolCall(idx int64, id string) *toolCallState {
	for _, tc := range a.toolCalls {
		if tc.index == idx || (id != "" && tc.id == id) {
			return tc
		}
	}

	tc := &toolCallState{
		id:    id,
		index: idx,
	}
	a.toolCalls = append(a.toolCalls, tc)

	return tc
}

func parseArgs(raw string) map[string]any {
	if strings.TrimSpace(raw) == "" {
		return map[string]any{}
	}

	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return map[string]any{
			"raw": raw,
		}
	}

	return out
}

func openAIUsage(u *openai.CompletionUsage) *genai.GenerateContentResponseUsageMetadata {
	if u == nil {
		return nil
	}

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(u.PromptTokens),
		CandidatesTokenCount: int32(u.CompletionTokens),
		TotalTokenCount:      int32(u.TotalTokens),
	}
}

// mapOpenAIFinishReason maps [openai.ChatCompletionChoice.FinishReason] to [genai.FinishReason].
//
// The [openai.ChatCompletionChoice.FinishReason] is any of:
//
//   - stop
//   - length
//   - tool_calls
//   - content_filter
//   - function_call (deprecated)
func mapOpenAIFinishReason(reason string) genai.FinishReason {
	switch strings.ToLower(reason) {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	case "tool_calls", "function_call":
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonUnspecified
	}
}
