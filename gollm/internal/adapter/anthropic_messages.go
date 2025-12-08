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

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"google.golang.org/genai"
)

// GenAIToAnthropicMessages converts the GenAI contents into Anthropic beta message parameters.
func GenAIToAnthropicMessages(system *genai.Content, contents []*genai.Content) ([]anthropic.BetaTextBlockParam, []anthropic.BetaMessageParam, error) {
	var systemBlocks []anthropic.BetaTextBlockParam
	if system != nil {
		text := joinTextParts(system.Parts)
		if text != "" {
			systemBlocks = append(systemBlocks, anthropic.BetaTextBlockParam{
				Type: constant.ValueOf[constant.Text](),
				Text: text,
			})
		}
	}

	msgs := make([]anthropic.BetaMessageParam, 0, len(contents))
	for idx, c := range contents {
		if c == nil {
			continue
		}
		role := strings.ToLower(c.Role)
		mp := anthropic.BetaMessageParam{
			Content: make([]anthropic.BetaContentBlockParamUnion, 0, len(c.Parts)),
		}
		if role == genai.RoleUser {
			mp.Role = anthropic.BetaMessageParamRoleUser
		} else {
			mp.Role = anthropic.BetaMessageParamRoleAssistant
		}

		for pi, part := range c.Parts {
			if part == nil {
				continue
			}
			switch {
			case part.Text != "":
				mp.Content = append(mp.Content, anthropic.NewBetaTextBlock(part.Text))

			case part.FunctionCall != nil:
				fc := part.FunctionCall
				if fc.Name == "" {
					return nil, nil, fmt.Errorf("content[%d] part[%d]: function call missing name", idx, pi)
				}
				args := fc.Args
				if args == nil {
					args = map[string]any{}
				}
				mp.Content = append(mp.Content, anthropic.BetaContentBlockParamUnion{
					OfToolUse: &anthropic.BetaToolUseBlockParam{
						ID:    toolID(fc.ID, idx, pi),
						Name:  fc.Name,
						Input: args,
						Type:  constant.ValueOf[constant.ToolUse](),
					},
				})

			case part.FunctionResponse != nil:
				fr := part.FunctionResponse
				if fr.Name == "" {
					return nil, nil, fmt.Errorf("content[%d] part[%d]: function response missing name", idx, pi)
				}
				contentJSON, err := json.Marshal(fr.Response)
				if err != nil {
					return nil, nil, fmt.Errorf("marshal json: %w", err)
				}
				mp.Content = append(mp.Content, anthropic.BetaContentBlockParamUnion{
					OfToolResult: &anthropic.BetaToolResultBlockParam{
						ToolUseID: toolID(fr.ID, idx, pi),
						Content: []anthropic.BetaToolResultBlockParamContentUnion{
							{OfText: &anthropic.BetaTextBlockParam{Type: constant.ValueOf[constant.Text](), Text: string(contentJSON)}},
						},
						Type: constant.ValueOf[constant.ToolResult](),
					},
				})

			default:
				return nil, nil, fmt.Errorf("content[%d] part[%d]: unsupported part", idx, pi)
			}
		}

		if len(mp.Content) == 0 {
			return nil, nil, fmt.Errorf("content[%d]: empty parts", idx)
		}
		msgs = append(msgs, mp)
	}

	return systemBlocks, msgs, nil
}
