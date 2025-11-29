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
	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"google.golang.org/genai"
)

// Convert ADK tool declarations to Anthropics definitions.
func GenaiToolsToAnthropic(tools []*genai.Tool, cfg *genai.ToolConfig) ([]anthropic.ToolUnionParam, *anthropic.ToolChoiceUnionParam) {
	if len(tools) == 0 {
		return nil, nil
	}

	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		for _, decl := range t.FunctionDeclarations {
			if decl == nil || decl.Name == "" {
				continue
			}
			out = append(out, anthropic.ToolUnionParam{
				OfTool: &anthropic.ToolParam{
					Name:        decl.Name,
					Description: param.NewOpt(decl.Description),
					InputSchema: anthropic.ToolInputSchemaParam{
						Type:       constant.ValueOf[constant.Object](),
						Properties: decl.Parameters,
					},
					Type: anthropic.ToolTypeCustom,
				},
			})
		}
	}

	var tc *anthropic.ToolChoiceUnionParam
	if cfg != nil && cfg.FunctionCallingConfig != nil {
		switch cfg.FunctionCallingConfig.Mode {
		case genai.FunctionCallingConfigModeNone:
			none := anthropic.NewToolChoiceNoneParam()
			tc = &anthropic.ToolChoiceUnionParam{OfNone: &none}
		case genai.FunctionCallingConfigModeAny, genai.FunctionCallingConfigModeAuto:
			tc = &anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{Type: constant.ValueOf[constant.Auto]()}}
		}
	}

	return out, tc
}
