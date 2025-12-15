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
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"
)

// GenAIToolsToResponses maps GenAI tool declarations into Responses tool parameters and choice options.
func GenAIToolsToResponses(tools []*genai.Tool, cfg *genai.ToolConfig) (params []responses.ToolUnionParam, choiceOpt *responses.ResponseNewParamsToolChoiceUnion) {
	if len(tools) == 0 {
		return nil, nil
	}

	params = make([]responses.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		for _, decl := range t.FunctionDeclarations {
			if decl == nil || decl.Name == "" {
				continue
			}

			fn := responses.ToolParamOfFunction(decl.Name, nil, true)
			if desc := strings.TrimSpace(decl.Description); desc != "" {
				fn.OfFunction.Description = param.NewOpt(desc)
			}

			if paramsJSON, err := toFunctionParameters(decl.ParametersJsonSchema); err == nil && paramsJSON != nil {
				fn.OfFunction.Parameters = paramsJSON
			} else if paramsJSON, err := toFunctionParameters(decl.Parameters); err == nil && paramsJSON != nil {
				fn.OfFunction.Parameters = paramsJSON
			}

			params = append(params, fn)
		}
	}

	if cfg != nil && cfg.FunctionCallingConfig != nil {
		switch cfg.FunctionCallingConfig.Mode {
		case genai.FunctionCallingConfigModeNone:
			choiceOpt = &responses.ResponseNewParamsToolChoiceUnion{
				OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsNone),
			}

		case genai.FunctionCallingConfigModeAny:
			if len(cfg.FunctionCallingConfig.AllowedFunctionNames) == 1 {
				choiceOpt = &responses.ResponseNewParamsToolChoiceUnion{
					OfFunctionTool: &responses.ToolChoiceFunctionParam{
						Name: cfg.FunctionCallingConfig.AllowedFunctionNames[0],
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

	switch v := src.(type) {
	case map[string]any:
		return v, nil
	case []byte:
		var params map[string]any
		if err := json.Unmarshal(v, &params); err != nil {
			return nil, fmt.Errorf("unmarshal json: %w", err)
		}
		return params, nil
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
