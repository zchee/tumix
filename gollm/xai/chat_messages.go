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
	xaipb "github.com/zchee/tumix/gollm/xai/api/v1"
)

// User creates a user message with text or content parts.
func User(parts ...any) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_USER, parts...)
}

// System creates a system message.
func System(parts ...any) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_SYSTEM, parts...)
}

// Assistant creates an assistant message.
func Assistant(parts ...any) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_ASSISTANT, parts...)
}

// ToolResult creates a tool result message.
func ToolResult(result string) *xaipb.Message {
	return newMessage(xaipb.MessageRole_ROLE_TOOL, result)
}

func newMessage(role xaipb.MessageRole, parts ...any) *xaipb.Message {
	contents := make([]*xaipb.Content, 0, len(parts))
	for _, part := range parts {
		switch v := part.(type) {
		case string:
			contents = append(contents, TextContent(v))
		case *xaipb.Content:
			contents = append(contents, v)
		default:
			panic("unsupported content type")
		}
	}

	return &xaipb.Message{
		Role:    role,
		Content: contents,
	}
}

func buildMessageFromCompletion(out *xaipb.CompletionOutput) *xaipb.Message {
	var reasoning *string
	if out.GetMessage().GetReasoningContent() != "" {
		rc := out.GetMessage().GetReasoningContent()
		reasoning = &rc
	}

	return &xaipb.Message{
		Role: out.GetMessage().GetRole(),
		Content: []*xaipb.Content{
			TextContent(out.GetMessage().GetContent()),
		},
		ReasoningContent: reasoning,
		EncryptedContent: out.GetMessage().GetEncryptedContent(),
		ToolCalls:        out.GetMessage().GetToolCalls(),
	}
}

// TextContent wraps plain text into a Content message.
func TextContent(text string) *xaipb.Content {
	return &xaipb.Content{
		Content: &xaipb.Content_Text{
			Text: text,
		},
	}
}

// FileContentWithName references an uploaded file and provides a display name.
func FileContentWithName(fileID, name string) *xaipb.Content {
	_ = name // name not supported in current proto; kept for parity but ignored
	return FileContent(fileID)
}

// ImageContent creates an image content entry with optional detail.
func ImageContent(url string, detail xaipb.ImageDetail) *xaipb.Content {
	return &xaipb.Content{
		Content: &xaipb.Content_ImageUrl{
			ImageUrl: &xaipb.ImageUrlContent{
				ImageUrl: url,
				Detail:   detail,
			},
		},
	}
}

// FileContent references an uploaded file (id only).
func FileContent(fileID string) *xaipb.Content {
	return &xaipb.Content{
		Content: &xaipb.Content_File{
			File: &xaipb.FileContent{
				FileId: fileID,
			},
		},
	}
}
