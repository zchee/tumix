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

	"google.golang.org/genai"

	"github.com/zchee/tumix/gollm/xai"
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

// GenAI2XAIChatOptions builds a ChatOption that maps GenAI generation config into xAI request fields.
func GenAI2XAIChatOptions(config *genai.GenerateContentConfig) xai.ChatOption {
	if config == nil {
		return nil
	}

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
