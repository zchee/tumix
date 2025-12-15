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
	"context"
	"fmt"
	"iter"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/xai"
)

// XAIStreamAggregator accumulates streaming xAI responses into coherent LLM responses.
type XAIStreamAggregator struct {
	text        strings.Builder
	thoughtText strings.Builder
	response    *model.LLMResponse
	role        string
}

// NewXAIStreamAggregator constructs a streaming aggregator for xAI responses.
func NewXAIStreamAggregator() *XAIStreamAggregator {
	return &XAIStreamAggregator{}
}

// Process ingests a streaming xAI response and yields partial or complete LLM responses.
func (s *XAIStreamAggregator) Process(_ context.Context, xaiResp *xai.Response) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if xaiResp.Content() == "" {
			// shouldn't happen?
			yield(nil, fmt.Errorf("empty response"))
			return
		}

		resp := XAIResponseToLLM(xaiResp)
		resp.TurnComplete = mapXAIFinishReason(xaiResp.FinishReason()) != ""
		// Aggregate the response and check if an intermediate event to yield was created
		if aggrResp := s.aggregateResponse(resp); aggrResp != nil {
			if !yield(aggrResp, nil) {
				return // Consumer stopped
			}
		}
		// Yield the processed response
		if !yield(resp, nil) {
			return // Consumer stopped
		}
	}
}

func (s *XAIStreamAggregator) aggregateResponse(llmResponse *model.LLMResponse) *model.LLMResponse {
	s.response = llmResponse

	var part0 *genai.Part
	if llmResponse.Content != nil && len(llmResponse.Content.Parts) > 0 {
		part0 = llmResponse.Content.Parts[0]
		s.role = llmResponse.Content.Role
	}

	// If part is text append it
	if part0 != nil && part0.Text != "" {
		appendDeltaToBuilder(part0.Text, func() *strings.Builder {
			if part0.Thought {
				return &s.thoughtText
			}
			return &s.text
		}())
		llmResponse.Partial = true
		return nil
	}

	// gemini 3 in streaming returns a last response with an empty part. We need to filter it out.
	// TODO(zchee): This logic for the gemini 3.
	if isZeroPart(part0) {
		llmResponse.Partial = true
		return nil
	}

	// If there is aggregated text and there is no content or parts return aggregated response
	if (s.thoughtText.Len() != 0 || s.text.Len() != 0) &&
		(llmResponse.Content == nil ||
			len(llmResponse.Content.Parts) == 0 ||
			// don't yield the merged text event when receiving audio data
			(len(llmResponse.Content.Parts) > 0 && llmResponse.Content.Parts[0].InlineData == nil)) {
		return s.Close()
	}

	return nil
}

// Close returns the final aggregated LLM response and resets the aggregator state.
func (s *XAIStreamAggregator) Close() *model.LLMResponse {
	if (s.text.Len() != 0 || s.thoughtText.Len() != 0) && s.response != nil {
		var parts []*genai.Part
		if s.thoughtText.Len() != 0 {
			parts = append(parts, &genai.Part{Text: s.thoughtText.String(), Thought: true})
		}
		if s.text.Len() != 0 {
			parts = append(parts, &genai.Part{Text: s.text.String(), Thought: false})
		}

		response := &model.LLMResponse{
			Content:           &genai.Content{Parts: parts, Role: s.role},
			ErrorCode:         s.response.ErrorCode,
			ErrorMessage:      s.response.ErrorMessage,
			UsageMetadata:     s.response.UsageMetadata,
			GroundingMetadata: s.response.GroundingMetadata,
			FinishReason:      s.response.FinishReason,
		}
		s.clear()
		return response
	}
	s.clear()
	return nil
}

func (s *XAIStreamAggregator) clear() {
	s.response = nil
	s.text.Reset()
	s.thoughtText.Reset()
	s.role = ""
}

func appendDeltaToBuilder(incoming string, acc *strings.Builder) string {
	if acc == nil || incoming == "" {
		return incoming
	}

	// Avoid quadratic concatenation by appending only the new suffix when the incoming
	// text already contains the previous accumulated text as a prefix.
	existing := acc.String()
	delta := incoming
	if len(incoming) >= len(existing) && strings.HasPrefix(incoming, existing) {
		delta = incoming[len(existing):]
	}
	if delta == "" {
		return ""
	}

	acc.WriteString(delta)
	return delta
}

func isZeroPart(p *genai.Part) bool {
	if p == nil {
		return false
	}
	return p.MediaResolution == nil &&
		p.CodeExecutionResult == nil &&
		p.ExecutableCode == nil &&
		p.FileData == nil &&
		p.FunctionCall == nil &&
		p.FunctionResponse == nil &&
		p.InlineData == nil &&
		p.Text == "" &&
		!p.Thought &&
		len(p.ThoughtSignature) == 0 &&
		p.VideoMetadata == nil
}
