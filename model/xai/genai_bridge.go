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

package xai

import (
	json "encoding/json/v2"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/genai"

	xaipb "github.com/zchee/tumix/model/xai/api/v1"
)

// GenAIContentsToMessages converts a GenerateContent request payload into xAI chat messages.
// It maps system instruction (when present) and each content entry into the closest xAI shape,
// preserving roles and tool calling information. Unsupported parts return an error so callers
// can fail fast instead of silently dropping context.
func GenAIContentsToMessages(system *genai.Content, contents []*genai.Content) ([]*xaipb.Message, error) {
	msgs := make([]*xaipb.Message, 0, len(contents)+1)

	if system != nil {
		sysMsg, err := genaiContentToMessage(system, xaipb.MessageRole_ROLE_SYSTEM)
		if err != nil {
			return nil, fmt.Errorf("system instruction: %w", err)
		}
		msgs = append(msgs, sysMsg)
	}

	for i, c := range contents {
		if c == nil {
			continue
		}
		msg, err := genaiContentToMessage(c, xaipb.MessageRole_INVALID_ROLE)
		if err != nil {
			return nil, fmt.Errorf("content[%d]: %w", i, err)
		}
		msgs = append(msgs, msg)
	}

	if len(msgs) == 0 {
		return nil, errors.New("no messages to send")
	}

	return msgs, nil
}

func genaiContentToMessage(c *genai.Content, overrideRole xaipb.MessageRole) (*xaipb.Message, error) {
	if c == nil {
		return nil, errors.New("nil content")
	}

	role := overrideRole
	if role == xaipb.MessageRole_INVALID_ROLE {
		var err error
		role, err = mapGenAIRole(c.Role)
		if err != nil {
			return nil, err
		}
	}

	msg := &xaipb.Message{
		Role: role,
	}

	for pi, part := range c.Parts {
		if part == nil {
			continue
		}

		switch {
		case part.Text != "":
			msg.Content = append(msg.Content, TextContent(part.Text))

		case part.FunctionCall != nil:
			tc, err := functionCallToToolCall(part.FunctionCall)
			if err != nil {
				return nil, fmt.Errorf("part[%d] function_call: %w", pi, err)
			}
			msg.ToolCalls = append(msg.ToolCalls, tc)

		case part.FunctionResponse != nil:
			payload, err := encodeFunctionResponse(part.FunctionResponse)
			if err != nil {
				return nil, fmt.Errorf("part[%d] function_response: %w", pi, err)
			}
			if msg.Role == xaipb.MessageRole_INVALID_ROLE {
				msg.Role = xaipb.MessageRole_ROLE_TOOL
			}
			msg.Content = append(msg.Content, TextContent(payload))

		case part.FileData != nil:
			msg.Content = append(msg.Content, FileContent(part.FileData.FileURI))

		case part.InlineData != nil:
			return nil, fmt.Errorf("part[%d]: inline data is not supported by xAI chat", pi)

		default:
			return nil, fmt.Errorf("part[%d]: unsupported part", pi)
		}
	}

	if len(msg.Content) == 0 && len(msg.ToolCalls) == 0 {
		return nil, errors.New("message has neither content nor tool calls")
	}

	return msg, nil
}

func mapGenAIRole(role string) (xaipb.MessageRole, error) {
	switch strings.ToLower(role) {
	case "", string(genai.RoleUser):
		return xaipb.MessageRole_ROLE_USER, nil
	case string(genai.RoleModel), "assistant":
		return xaipb.MessageRole_ROLE_ASSISTANT, nil
	case "system":
		return xaipb.MessageRole_ROLE_SYSTEM, nil
	case "tool", "function":
		return xaipb.MessageRole_ROLE_TOOL, nil
	default:
		return xaipb.MessageRole_INVALID_ROLE, fmt.Errorf("unsupported role %q", role)
	}
}

func functionCallToToolCall(fc *genai.FunctionCall) (*xaipb.ToolCall, error) {
	if fc == nil {
		return nil, errors.New("nil function call")
	}
	if fc.Name == "" {
		return nil, errors.New("function call name is required")
	}
	if len(fc.PartialArgs) > 0 {
		return nil, errors.New("partial function call arguments are not supported for xAI chat")
	}

	argsJSON := ""
	if len(fc.Args) > 0 {
		raw, err := json.Marshal(fc.Args)
		if err != nil {
			return nil, fmt.Errorf("marshal function call args: %w", err)
		}
		argsJSON = string(raw)
	}

	return &xaipb.ToolCall{
		Id: fc.ID,
		Tool: &xaipb.ToolCall_Function{
			Function: &xaipb.FunctionCall{
				Name:      fc.Name,
				Arguments: argsJSON,
			},
		},
	}, nil
}

func encodeFunctionResponse(fr *genai.FunctionResponse) (string, error) {
	if fr == nil {
		return "", errors.New("nil function response")
	}
	if fr.Name == "" {
		return "", errors.New("function response name is required")
	}

	payload := map[string]any{
		"name": fr.Name,
	}

	if fr.ID != "" {
		payload["tool_call_id"] = fr.ID
	}
	if fr.Response != nil {
		payload["response"] = fr.Response
	}
	if fr.Scheduling != "" {
		payload["scheduling"] = fr.Scheduling
	}
	if fr.WillContinue != nil {
		payload["will_continue"] = *fr.WillContinue
	}
	if len(fr.Parts) > 0 {
		payload["parts"] = fr.Parts
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal function response: %w", err)
	}

	return string(raw), nil
}
