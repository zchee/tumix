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
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"slices"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/xai"
)

// XAIResponseToLLM converts an xAI response into an ADK LLMResponse, preserving usage and metadata.
func XAIResponseToLLM(resp *xai.Response) *model.LLMResponse {
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
		dec := jsontext.NewDecoder(strings.NewReader(""))
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
