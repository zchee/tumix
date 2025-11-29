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
	"fmt"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// toolID returns a unique identifier for a tool call.
//
// If id is empty, it generates one based on the content and part indices.
func toolID(id string, contentIdx, partIdx int) string {
	if strings.TrimSpace(id) != "" {
		return id
	}
	return fmt.Sprintf("tool_%d_%d", contentIdx, partIdx)
}

// ensureUserContent aligns with ADK behavior of ending with a user turn.
//
// It appends a default user message if the request content is empty or
// if the last message is not from the user.
func ensureUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", genai.RoleUser))
		return
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != genai.RoleUser {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", genai.RoleUser))
	}
}

// resolveModelName returns the model name to use for the request.
//
// It prefers the model name in the request if provided, otherwise falls back to the default name.
func resolveModelName(req *model.LLMRequest, defaultName string) string {
	if req != nil {
		if name := strings.TrimSpace(req.Model); name != "" {
			return name
		}
	}
	return defaultName
}
