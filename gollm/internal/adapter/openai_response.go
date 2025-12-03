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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// OpenAIResponseToLLM converts an OpenAI Responses payload into an ADK LLMResponse,
// returning an error when the payload is nil or contains no output items.
func OpenAIResponseToLLM(resp *responses.Response) (*model.LLMResponse, error) {
	if resp == nil {
		return nil, fmt.Errorf("nil openai response")
	}
	if len(resp.Output) == 0 {
		return nil, fmt.Errorf("empty output")
	}

	parts := make([]*genai.Part, 0, len(resp.Output))

	for _, item := range resp.Output {
		typ := strings.ToLower(strings.TrimSpace(item.Type))
		switch typ {
		case "message":
			if len(item.Content) == 0 {
				continue
			}
			for _, c := range item.Content {
				ctype := strings.ToLower(strings.TrimSpace(c.Type))
				switch ctype {
				case "output_text":
					if c.Text != "" {
						parts = append(parts, genai.NewPartFromText(c.Text))
					}
				case "refusal":
					if c.Refusal != "" {
						parts = append(parts, genai.NewPartFromText(c.Refusal))
					}
				}
			}

		case "function_call":
			fnID := item.CallID
			if fnID == "" {
				fnID = item.ID
			}
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   fnID,
					Name: item.Name,
					Args: parseArgs(item.Arguments),
				},
			})
		}
	}

	if len(parts) == 0 {
		return nil, fmt.Errorf("no convertible output items")
	}

	usage := openAIUsage(&resp.Usage)
	finish := mapResponseFinishReason(resp.Status, resp.IncompleteDetails)

	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  genai.RoleModel,
			Parts: parts,
		},
		UsageMetadata: usage,
		TurnComplete:  resp.Status == responses.ResponseStatusCompleted || resp.Status == responses.ResponseStatusIncomplete,
		FinishReason:  finish,
	}, nil
}

type toolCallState struct {
	id    string
	name  string
	index int64
	args  strings.Builder
}

// OpenAIStreamAggregator aggregates Responses streaming events into LLM responses.
type OpenAIStreamAggregator struct {
	text      strings.Builder
	toolCalls []*toolCallState
	usage     *responses.ResponseUsage
	status    responses.ResponseStatus
	final     *model.LLMResponse
	err       error
}

// NewOpenAIStreamAggregator constructs a streaming aggregator for Responses events.
func NewOpenAIStreamAggregator() *OpenAIStreamAggregator {
	return &OpenAIStreamAggregator{}
}

// Process consumes a streaming event and emits any partial LLM responses produced by it.
func (a *OpenAIStreamAggregator) Process(event responses.ResponseStreamEventUnion) []*model.LLMResponse {
	switch event.Type {
	case "response.output_text.delta":
		if event.Delta == "" {
			return nil
		}
		a.text.WriteString(event.Delta)
		return []*model.LLMResponse{
			{
				Content: &genai.Content{
					Role:  genai.RoleModel,
					Parts: []*genai.Part{genai.NewPartFromText(event.Delta)},
				},
				Partial: true,
			},
		}

	case "response.function_call_arguments.delta":
		state := a.ensureToolCall(event.OutputIndex, event.ItemID)
		if event.Delta != "" {
			state.args.WriteString(event.Delta)
		}
		return nil

	case "response.function_call_arguments.done":
		state := a.ensureToolCall(event.OutputIndex, event.ItemID)
		if event.Name != "" {
			state.name = event.Name
		}
		if event.Arguments != "" {
			state.args.Reset()
			state.args.WriteString(event.Arguments)
		}
		if event.ItemID != "" {
			state.id = event.ItemID
		}
		return nil

	case "response.completed":
		a.status = responses.ResponseStatusCompleted
		a.usage = &event.Response.Usage
		llm, err := OpenAIResponseToLLM(&event.Response)
		if err == nil {
			a.final = llm
			return []*model.LLMResponse{llm}
		}
		// fall through to fallback aggregation if conversion failed.
		return nil

	case "response.incomplete":
		a.status = responses.ResponseStatusIncomplete
		a.usage = &event.Response.Usage
		llm, err := OpenAIResponseToLLM(&event.Response)
		if err == nil {
			a.final = llm
			return []*model.LLMResponse{llm}
		}
		return nil

	case "response.failed", "error":
		msg := strings.TrimSpace(event.Message)
		if msg == "" {
			msg = "openai response failed"
		}
		a.err = fmt.Errorf("%s", msg)
		return nil
	}

	return nil
}

// Final returns the terminal aggregated LLM response, or nil when nothing was accumulated.
func (a *OpenAIStreamAggregator) Final() *model.LLMResponse {
	if a.final != nil {
		return a.final
	}
	if a.err != nil {
		return nil
	}
	if a.text.Len() == 0 && len(a.toolCalls) == 0 {
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
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   tc.id,
					Name: tc.name,
					Args: parseArgs(tc.args.String()),
				},
			})
		}
	}

	finish := mapResponseFinishReason(a.status, responses.ResponseIncompleteDetails{})
	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  genai.RoleModel,
			Parts: parts,
		},
		UsageMetadata: openAIUsage(a.usage),
		TurnComplete:  a.status == responses.ResponseStatusCompleted || a.status == responses.ResponseStatusIncomplete,
		FinishReason:  finish,
	}
}

// Err returns the terminal error captured during stream aggregation.
func (a *OpenAIStreamAggregator) Err() error {
	return a.err
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
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]any{}
	}

	dec := argsDecoderPool.Get().(*jsontext.Decoder)
	defer argsDecoderPool.Put(dec)
	dec.Reset(strings.NewReader(trimmed))

	var out map[string]any
	if err := json.UnmarshalDecode(dec, &out); err != nil {
		return map[string]any{"raw": raw}
	}

	return out
}

var argsDecoderPool = sync.Pool{
	New: func() any {
		return jsontext.NewDecoder(strings.NewReader(""))
	},
}

func openAIUsage(u *responses.ResponseUsage) *genai.GenerateContentResponseUsageMetadata {
	if u == nil {
		return nil
	}

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(u.InputTokens),
		CandidatesTokenCount: int32(u.OutputTokens),
		TotalTokenCount:      int32(u.TotalTokens),
	}
}

// mapResponseFinishReason maps [responses.ResponseStatus] and incomplete details to [genai.FinishReason].
func mapResponseFinishReason(status responses.ResponseStatus, details responses.ResponseIncompleteDetails) genai.FinishReason {
	switch status {
	case responses.ResponseStatusCompleted:
		return genai.FinishReasonStop
	case responses.ResponseStatusIncomplete:
		switch strings.ToLower(details.Reason) {
		case "max_output_tokens":
			return genai.FinishReasonMaxTokens
		case "content_filter":
			return genai.FinishReasonSafety
		default:
			return genai.FinishReasonOther
		}
	case responses.ResponseStatusFailed:
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonUnspecified
	}
}
