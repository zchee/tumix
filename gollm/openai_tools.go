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
	"encoding/json/v2"
	"fmt"
	"strings"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"
)

func genaiToolsToOpenAI(tools []*genai.Tool, cfg *genai.ToolConfig) (params []openai.ChatCompletionToolUnionParam, choiceOpt *openai.ChatCompletionToolChoiceOptionUnionParam) {
	if len(tools) == 0 {
		return nil, nil
	}

	params = make([]openai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, t := range tools {
		for _, decl := range t.FunctionDeclarations {
			if decl == nil || decl.Name == "" {
				continue
			}

			fn := shared.FunctionDefinitionParam{
				Name: decl.Name,
			}
			if desc := strings.TrimSpace(decl.Description); desc != "" {
				fn.Description = openai.String(desc)
			}

			if params, err := toFunctionParameters(decl.ParametersJsonSchema); err == nil && params != nil {
				fn.Parameters = params
			} else if params, err := toFunctionParameters(decl.Parameters); err == nil && params != nil {
				fn.Parameters = params
			}

			params = append(params, openai.ChatCompletionFunctionTool(fn))
		}
	}

	if cfg != nil && cfg.FunctionCallingConfig != nil {
		switch cfg.FunctionCallingConfig.Mode {
		case genai.FunctionCallingConfigModeNone:
			choiceOpt = &openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("none"),
			}

		case genai.FunctionCallingConfigModeAny:
			if len(cfg.FunctionCallingConfig.AllowedFunctionNames) == 1 {
				choiceOpt = &openai.ChatCompletionToolChoiceOptionUnionParam{
					OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
						Function: openai.ChatCompletionNamedToolChoiceFunctionParam{
							Name: cfg.FunctionCallingConfig.AllowedFunctionNames[0],
						},
						// [openai.ChatCompletionToolChoiceOptionUnionParam.Type] can be elided but just in case
						Type: constant.ValueOf[constant.Function](),
					},
				}
			}
		}
	}

	return params, choiceOpt
}

func toFunctionParameters(src any) (shared.FunctionParameters, error) {
	if src == nil {
		return nil, nil //nolint:nilnil
	}

	raw, err := json.Marshal(src)
	if err != nil {
		return nil, fmt.Errorf("marshal json: %w", err)
	}

	var params map[string]any
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, fmt.Errorf("unmarshal json: %w", err)
	}

	return params, nil
}
