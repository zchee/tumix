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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"iter"
	"reflect"
	"slices"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/xai"
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

func GenAI2XAIChatOptions(config *genai.GenerateContentConfig) xai.ChatOption {
	if config == nil {
		return nil
	}
	// ... (rest of function)

	sb := new(strings.Builder)
	enc := jsontext.NewEncoder(sb)
	marshalJSON := func(v any) (string, bool) {
		if v == nil {
			return "", false
		}

		sb.Reset()
		enc.Reset(sb)
		if err := json.MarshalEncode(enc, v); err != nil {
			return "", false
		}

		return sb.String(), true
	}

	var (
		temperature      *float32
		topP             *float32
		maxTokens        *int32
		seed             *int32
		stop             []string
		logprobs         bool
		topLogprobs      *int32
		frequencyPenalty *float32
		presencePenalty  *float32
		tools            []*xaipb.Tool
		toolChoice       *xaipb.ToolChoice
		responseFormat   *xaipb.ResponseFormat
	)

	if config.Temperature != nil {
		v := *config.Temperature
		temperature = &v
	}
	if config.TopP != nil {
		v := *config.TopP
		topP = &v
	}
	if config.MaxOutputTokens > 0 {
		v := config.MaxOutputTokens
		maxTokens = &v
	}
	if config.Seed != nil {
		v := *config.Seed
		seed = &v
	}
	if len(config.StopSequences) > 0 {
		stop = slices.Clone(config.StopSequences)
	}
	if config.Logprobs != nil {
		v := *config.Logprobs
		topLogprobs = &v
	}
	logprobs = config.ResponseLogprobs

	if config.FrequencyPenalty != nil {
		v := *config.FrequencyPenalty
		frequencyPenalty = &v
	}
	if config.PresencePenalty != nil {
		v := *config.PresencePenalty
		presencePenalty = &v
	}

	for tool := range slices.Values(config.Tools) {
		if tool == nil {
			continue
		}

		if len(tool.FunctionDeclarations) > 0 {
			for decl := range slices.Values(tool.FunctionDeclarations) {
				if decl == nil || decl.Name == "" {
					continue
				}

				params := ""
				switch {
				case decl.ParametersJsonSchema != nil:
					if raw, ok := marshalJSON(decl.ParametersJsonSchema); ok {
						params = raw
					}
				case decl.Parameters != nil:
					if raw, ok := marshalJSON(decl.Parameters); ok {
						params = raw
					}
				}

				tools = append(tools, &xaipb.Tool{
					Tool: &xaipb.Tool_Function{
						Function: &xaipb.Function{
							Name:        decl.Name,
							Description: decl.Description,
							Parameters:  params,
						},
					},
				})
			}
		}
		if tool.CodeExecution != nil {
			tools = append(tools, xai.CodeExecutionTool())
		}
	}

	if tc := config.ToolConfig; tc != nil && tc.FunctionCallingConfig != nil {
		fc := tc.FunctionCallingConfig
		switch fc.Mode {
		case genai.FunctionCallingConfigModeNone:
			toolChoice = &xaipb.ToolChoice{ToolChoice: &xaipb.ToolChoice_Mode{Mode: xaipb.ToolMode_TOOL_MODE_NONE}}
		case genai.FunctionCallingConfigModeAny:
			if len(fc.AllowedFunctionNames) == 1 && fc.AllowedFunctionNames[0] != "" {
				toolChoice = &xaipb.ToolChoice{ToolChoice: &xaipb.ToolChoice_FunctionName{FunctionName: fc.AllowedFunctionNames[0]}}
			} else {
				toolChoice = &xaipb.ToolChoice{ToolChoice: &xaipb.ToolChoice_Mode{Mode: xaipb.ToolMode_TOOL_MODE_REQUIRED}}
			}
		case genai.FunctionCallingConfigModeAuto:
			toolChoice = &xaipb.ToolChoice{ToolChoice: &xaipb.ToolChoice_Mode{Mode: xaipb.ToolMode_TOOL_MODE_AUTO}}
		}
	}

	switch {
	case config.ResponseJsonSchema != nil:
		if schema, ok := marshalJSON(config.ResponseJsonSchema); ok {
			responseFormat = &xaipb.ResponseFormat{
				FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA,
				Schema:     &schema,
			}
		}
	case config.ResponseSchema != nil:
		if schema, ok := marshalJSON(config.ResponseSchema); ok {
			responseFormat = &xaipb.ResponseFormat{
				FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_SCHEMA,
				Schema:     &schema,
			}
		}
	default:
		mime := strings.ToLower(strings.TrimSpace(config.ResponseMIMEType))
		if strings.HasPrefix(mime, "application/json") {
			responseFormat = &xaipb.ResponseFormat{FormatType: xaipb.FormatType_FORMAT_TYPE_JSON_OBJECT}
		}
	}

	if logprobs || topLogprobs != nil {
		logprobs = true
	}

	hasEffect := temperature != nil || topP != nil || maxTokens != nil || seed != nil || len(stop) > 0 || logprobs || topLogprobs != nil || frequencyPenalty != nil || presencePenalty != nil || len(tools) > 0 || toolChoice != nil || responseFormat != nil
	if !hasEffect {
		return nil
	}

	opt := xai.ChatOption(func(req *xaipb.GetCompletionsRequest, _ *xai.ChatSession) {
		if maxTokens != nil {
			req.MaxTokens = maxTokens
		}
		if seed != nil {
			req.Seed = seed
		}
		if temperature != nil {
			req.Temperature = temperature
		}
		if topP != nil {
			req.TopP = topP
		}
		if len(stop) > 0 {
			req.Stop = append(req.Stop, stop...)
		}
		if logprobs {
			req.Logprobs = true
		}
		if topLogprobs != nil {
			req.TopLogprobs = topLogprobs
		}
		if frequencyPenalty != nil {
			req.FrequencyPenalty = frequencyPenalty
		}
		if presencePenalty != nil {
			req.PresencePenalty = presencePenalty
		}
		if len(tools) > 0 {
			req.Tools = append(req.Tools, tools...)
		}
		if toolChoice != nil {
			req.ToolChoice = toolChoice
		}
		if responseFormat != nil {
			req.ResponseFormat = responseFormat
		}
	})

	return opt
}

func XAI2LLMResponse(resp *xai.Response) *model.LLMResponse {
	if resp == nil {
		return &model.LLMResponse{
			ErrorCode:    "NIL_RESPONSE",
			ErrorMessage: "xAI response is nil",
		}
	}

	usage := resp.Usage()
	var usageMetadata *genai.GenerateContentResponseUsageMetadata
	if usage != nil {
		usageMetadata = &genai.GenerateContentResponseUsageMetadata{
			CachedContentTokenCount: usage.GetCachedPromptTextTokens(),
			CandidatesTokenCount:    usage.GetCompletionTokens(),
			PromptTokenCount:        usage.GetPromptTokens(),
			ThoughtsTokenCount:      usage.GetReasoningTokens(),
			TotalTokenCount:         usage.GetTotalTokens(),
		}
	}

	parts := make([]*genai.Part, 0, 3)

	if reasoning := resp.ReasoningContent(); reasoning != "" {
		parts = append(parts, &genai.Part{Text: reasoning, Thought: true})
	}

	if content := resp.Content(); content != "" {
		parts = append(parts, genai.NewPartFromText(content))
	}

	var argErrors []string
	if toolCalls := resp.ToolCalls(); len(toolCalls) > 0 { //nolint:nestif // TODO(zchee): fix nolint
		dec := jsontext.NewDecoder(nil)
		for _, call := range toolCalls {
			fc := call.GetFunction()
			if fc == nil {
				continue
			}

			args := map[string]any{}
			rawArgs := fc.GetArguments()
			if rawArgs != "" {
				dec.Reset(strings.NewReader(rawArgs))

				var obj map[string]any
				if err := json.UnmarshalDecode(dec, &obj); err == nil {
					args = obj
				} else {
					var generic any
					if err2 := json.UnmarshalDecode(dec, &generic); err2 == nil {
						args = map[string]any{"value": generic}
					} else {
						args["raw"] = rawArgs
						argErrors = append(argErrors, err.Error())
					}
				}
			}

			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   call.GetId(),
					Name: fc.GetName(),
					Args: args,
				},
			})
		}
	}

	role := strings.ToLower(strings.TrimPrefix(resp.Role(), "ROLE_"))
	// NOTE(zchee): genai support only "user" and "model" roles.
	switch role {
	case "user":
		role = genai.RoleUser
	default:
		role = genai.RoleModel
	}

	finishReason := mapXAIFinishReason(resp.FinishReason())

	custom := map[string]any{}
	if raw := resp.FinishReason(); raw != "" {
		custom["xai_finish_reason"] = raw
	}
	if fp := resp.SystemFingerprint(); fp != "" {
		custom["xai_system_fingerprint"] = fp
	}
	if citations := resp.Citations(); len(citations) > 0 {
		custom["xai_citations"] = slices.Clone(citations)
	}
	if len(argErrors) > 0 {
		custom["tool_call_args_errors"] = slices.Clone(argErrors)
	}
	if len(custom) == 0 {
		custom = nil
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  role,
			Parts: parts,
		},
		CustomMetadata: custom,
		UsageMetadata:  usageMetadata,
		FinishReason:   finishReason,
	}
}

type XAIStreamAggregator struct {
	text        string
	thoughtText string
	response    *model.LLMResponse
	role        string
}

func NewXAIStreamAggregator() *XAIStreamAggregator {
	return &XAIStreamAggregator{}
}

func (s *XAIStreamAggregator) Process(_ context.Context, xaiResp *xai.Response) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if xaiResp.Content() == "" {
			// shouldn't happen?
			yield(nil, fmt.Errorf("empty response"))
			return
		}

		resp := XAI2LLMResponse(xaiResp)
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
		delta := appendDelta(part0.Text, func() *string {
			if part0.Thought {
				return &s.thoughtText
			}
			return &s.text
		}())
		if part0.Thought {
			s.thoughtText += delta
		} else {
			s.text += delta
		}
		llmResponse.Partial = true
		return nil
	}

	// gemini 3 in streaming returns a last response with an empty part. We need to filter it out.
	// TODO(zchee): This logic for the gemini 3.
	if part0 != nil && reflect.ValueOf(*part0).IsZero() {
		llmResponse.Partial = true
		return nil
	}

	// If there is aggregated text and there is no content or parts return aggregated response
	if (s.thoughtText != "" || s.text != "") &&
		(llmResponse.Content == nil ||
			len(llmResponse.Content.Parts) == 0 ||
			// don't yield the merged text event when receiving audio data
			(len(llmResponse.Content.Parts) > 0 && llmResponse.Content.Parts[0].InlineData == nil)) {
		return s.Close()
	}

	return nil
}

func (s *XAIStreamAggregator) Close() *model.LLMResponse {
	if (s.text != "" || s.thoughtText != "") && s.response != nil {
		var parts []*genai.Part
		if s.thoughtText != "" {
			parts = append(parts, &genai.Part{Text: s.thoughtText, Thought: true})
		}
		if s.text != "" {
			parts = append(parts, &genai.Part{Text: s.text, Thought: false})
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
	s.text = ""
	s.thoughtText = ""
	s.role = ""
}

func appendDelta(incoming string, acc *string) string {
	if acc == nil || incoming == "" {
		return incoming
	}

	if after, ok := strings.CutPrefix(incoming, *acc); ok {
		return after
	}

	// If the incoming text is not a superset, treat it as fresh chunk to avoid data loss.
	return incoming
}

func mapXAIFinishReason(fr string) genai.FinishReason {
	switch strings.TrimPrefix(strings.ToUpper(fr), "REASON_") {
	case "", "INVALID":
		return genai.FinishReasonUnspecified
	case "STOP":
		return genai.FinishReasonStop
	case "MAX_LEN", "MAX_CONTEXT":
		return genai.FinishReasonMaxTokens
	case "TOOL_CALLS":
		return genai.FinishReasonOther
	case "TIME_LIMIT":
		return genai.FinishReasonOther
	default:
		return genai.FinishReasonOther
	}
}
