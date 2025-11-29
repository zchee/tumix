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
	"github.com/openai/openai-go/v3/shared/constant"
	"google.golang.org/genai"
)

func genaiToOpenAIMessages(contents []*genai.Content) ([]openai.ChatCompletionMessageParamUnion, error) {
	var msgs []openai.ChatCompletionMessageParamUnion
	var text strings.Builder

	for i, c := range contents {
		if c == nil {
			continue
		}

		text.Reset()
		toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(c.Parts))
		for j, part := range c.Parts {
			if part == nil {
				continue
			}

			switch {
			case part.Text != "":
				text.WriteString(part.Text)

			case part.FunctionCall != nil:
				fc := part.FunctionCall
				if fc.Name == "" {
					return nil, fmt.Errorf("content[%d] part[%d]: function call missing name", i, j)
				}
				argsJSON, err := json.Marshal(fc.Args)
				if err != nil {
					return nil, fmt.Errorf("content[%d] part[%d]: marshal function args: %w", i, j, err)
				}

				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
					OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
						ID:   toolID(fc.ID, i, j),
						Type: constant.ValueOf[constant.Function](),
						Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name:      fc.Name,
							Arguments: string(argsJSON),
						},
					},
				})

			case part.FunctionResponse != nil:
				fr := part.FunctionResponse
				if fr.Name == "" {
					return nil, fmt.Errorf("content[%d] part[%d]: function response missing name", i, j)
				}
				data, err := json.Marshal(fr.Response)
				if err != nil {
					return nil, fmt.Errorf("content[%d] part[%d]: marshal function response: %w", i, j, err)
				}
				msgs = append(msgs,
					openai.ToolMessage(string(data), toolID(fr.ID, i, j)),
				)

			default:
				return nil, fmt.Errorf("content[%d] part[%d]: unsupported part", i, j)
			}
		}

		role := strings.ToLower(strings.TrimSpace(c.Role))
		switch role {
		case genai.RoleUser:
			msgs = append(msgs, openai.UserMessage(text.String()))

		case genai.RoleModel:
			var modelParam openai.ChatCompletionAssistantMessageParam
			if text.Len() > 0 {
				modelParam.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(text.String()),
				}
			}
			if len(toolCalls) > 0 {
				modelParam.ToolCalls = toolCalls
			}
			// [openai.ChatCompletionAssistantMessageParam.Role] can be elided but just in case
			modelParam.Role = constant.ValueOf[constant.Assistant]()

			msgs = append(msgs,
				openai.ChatCompletionMessageParamUnion{
					OfAssistant: &modelParam,
				},
			)

		default:
			return nil, fmt.Errorf("content[%d]: unsupported role %q", i, role)
		}
	}

	return msgs, nil
}
