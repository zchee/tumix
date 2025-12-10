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

	"github.com/openai/openai-go/v3/responses"
	"google.golang.org/genai"
)

// GenAIToResponsesInput converts GenAI content slices into OpenAI Responses input items.
func GenAIToResponsesInput(contents []*genai.Content) ([]responses.ResponseInputItemUnionParam, error) {
	var items []responses.ResponseInputItemUnionParam

	for i, c := range contents {
		if c == nil {
			continue
		}

		role := toEasyRole(c.Role)
		var text strings.Builder

		flush := func() {
			if text.Len() == 0 {
				return
			}
			items = append(items, responses.ResponseInputItemParamOfMessage(text.String(), role))
			text.Reset()
		}

		for j, part := range c.Parts {
			if part == nil {
				continue
			}

			switch {
			case part.Text != "":
				text.WriteString(part.Text)

			case part.FunctionCall != nil:
				flush()
				fc := part.FunctionCall
				if fc.Name == "" {
					return nil, fmt.Errorf("content[%d] part[%d]: function call missing name", i, j)
				}
				argsJSON, err := json.Marshal(fc.Args)
				if err != nil {
					return nil, fmt.Errorf("content[%d] part[%d]: marshal function args: %w", i, j, err)
				}

				items = append(items,
					responses.ResponseInputItemParamOfFunctionCall(string(argsJSON), toolID(fc.ID, i, j), fc.Name),
				)

			case part.FunctionResponse != nil:
				flush()
				fr := part.FunctionResponse
				if fr.Name == "" {
					return nil, fmt.Errorf("content[%d] part[%d]: function response missing name", i, j)
				}
				data, err := json.Marshal(fr.Response)
				if err != nil {
					return nil, fmt.Errorf("content[%d] part[%d]: marshal function response: %w", i, j, err)
				}

				items = append(items,
					responses.ResponseInputItemParamOfFunctionCallOutput(toolID(fr.ID, i, j), string(data)),
				)

			default:
				return nil, fmt.Errorf("content[%d] part[%d]: unsupported part", i, j)
			}
		}

		flush()
	}

	return items, nil
}

func toEasyRole(role string) responses.EasyInputMessageRole {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case genai.RoleUser:
		return responses.EasyInputMessageRoleUser
	case genai.RoleModel:
		return responses.EasyInputMessageRoleAssistant
	case "system":
		return responses.EasyInputMessageRoleSystem
	case "developer":
		return responses.EasyInputMessageRoleDeveloper
	default:
		return responses.EasyInputMessageRoleUser
	}
}
